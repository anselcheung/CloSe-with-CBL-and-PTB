# Implementation Brief: Boundary-Aware CloSeNet

**Goal.** Improve CloSeNet's 3D clothing segmentation near garment boundaries (collar lines, sleeve hems, waistbands, shoe–pant transitions) by transferring the auxiliary supervision and guided-propagation ideas from *Push-the-Boundary* (PTB, Du et al., WACV 2023, arXiv:2212.12402) onto the CloSeNet architecture (Antić et al., 3DV 2024).

**Base repos.** `anticdimi/CloSe` (model to modify) and `shenglandu/PushBoundary` (reference implementation, `PointNet2_Backbone/` is the cleaner PyTorch reference).

---

## 1. What each method actually does

**CloSeNet** segments a colored scan point cloud into 18 clothing classes. Per forward pass it takes `points` (N × 9: xyz, rgb, normals), SMPL canonical-pose coordinates per point (`canon_pose_coords`, the body-prior "part encoder", which is just an identity passthrough of 3 dims), and a per-scan garment one-hot vector. The point encoder is a **DGCNN**: three EdgeConv blocks (k = 40, kNN built in *feature space*, max-pooling over neighbors), whose outputs are concatenated (192-d) with a repeated 1024-d global feature. A garment-embedding cross-attention module (queries = last EdgeConv features, keys/values = learned embeddings of the garments present in the scan) produces a 64-d garment prior. Everything is concatenated (1283-d per point) and pushed through a shared **per-point MLP decoder** → 18 logits. Training uses a single unweighted cross-entropy loss on 2048-point subsamples.

**PTB** attaches, to one shared FCN encoder, three parallel decoding streams: (i) a binary **boundary head** (is this point on a semantic boundary?), supervised with weighted BCE; (ii) a **direction head** regressing a unit vector pointing from the nearest boundary into the object's interior, supervised with MSE; (iii) the semantic head. The predicted boundary probability P_b and direction d are then fused into the semantic decoder's upsampling interpolation weights ("guided feature propagation", GFP):

```
w(x_j) = max(0,  exp(−‖x_j − x_i‖ / r)  +  α · exp(P_b(x_i) − 1) · cos(x_j − x_i, d_i))
```

so that a boundary point preferentially inherits features from neighbors lying in its interior direction, and interior points fall back to plain distance weighting. Their KPConv ablation on S3DIS is instructive for us: boundary head alone +1.0 mIoU, all three heads with *standard* propagation +0.7, full network with GFP +1.6. Roughly two-thirds of the benefit comes from the multi-task supervision itself, before any propagation mechanism.

## 2. The architectural mismatch you must design around

PTB's GFP is an *upsampling* operator: it modifies how features travel from sparse deep layers back to dense shallow layers in an encoder–decoder FCN. **CloSeNet has no such stage.** The DGCNN encoder never subsamples — every EdgeConv runs on all N points — and the segmentation "decoder" is a pointwise 1×1-conv MLP with no spatial neighborhood at all. There is literally no interpolation step to replace.

This has two consequences. First, the parts of PTB that transfer directly are the **auxiliary heads, their losses, and the GT generation pipeline** — these attach to any per-point feature tensor. Second, the guided-propagation idea must be *re-homed*: instead of guiding decoder upsampling, we use P_b and d to guide a spatial neighborhood aggregation that CloSeNet currently lacks. Ironically, CloSeNet's pointwise decoder is one plausible cause of its boundary blur (the other being the DGCNN feature-space kNN, which happily mixes features across a garment seam whenever two sides look similar), so adding a guided spatial aggregation step is not just a port — it fills a genuine gap.

## 3. Proposed design, in three phases of increasing risk

### Phase 1 — Auxiliary boundary + direction heads (pure PTB transfer, low risk)

Add two heads next to `segm_dec` in `lib/closenet/model.py`, consuming the same 1283-d concatenated per-point encoding (or a cheaper subset — see §6):

```python
self.boundary_head = MLPDecoder([segm_input, 256, 128, 2],  dropout, slope)
self.direction_head = MLPDecoder([segm_input, 256, 128, 3], dropout, slope)
# direction output: L2-normalize, e.g. F.normalize(d_raw, dim=1, eps=1e-8)
```

Losses, mirroring PTB's hyperparameters as starting points: weighted BCE (or CE over 2 classes) with boundary-class weight β = 0.6 for the boundary head; MSE against unit GT vectors for the direction head; total loss `L = L_seg + λ₁ L_B + λ₂ L_D` with λ₁ = 3.0, λ₂ = 0.3. The CloSe trainer already sums arbitrary weighted losses via `cfg.training.loss_weights`, so this is a config entry plus two lines in `CloSeNetTrainer.step` — a clean hook.

The mechanism of benefit here is representation shaping: forcing the shared DGCNN + attention features to be boundary-discriminative sharpens them exactly where CloSeNet currently smears. Given PTB's ablation, Phase 1 alone is a defensible expected win and is the cheapest experiment to run first.

A CloSe-specific twist worth trying immediately: feed the direction head the **canonical-pose coordinates** rather than (or in addition to) posed xyz, and define GT directions in canonical space. Garment boundaries (necklines, hems) live at nearly fixed places on the canonical body across subjects and poses, so directions in that frame should be far easier to regress than in posed world space. This exploits a prior PTB never had access to.

### Phase 2 — Boundary-guided refinement module (adapted GFP)

Since there is no upsampling to guide, insert a **Boundary-aware Refinement Module (BRM)** between the encoder concat and the final classifier (or between the penultimate decoder layer and the logits). For each point x_i, gather a small *spatial* kNN (k = 4–8, Euclidean, matching PTB's k = 4 in the guided interpolation) and re-aggregate neighbor features with exactly PTB's weight:

```python
# f: (B, N, C) pre-logit features; xyz: (B, N, 3)
# pb: boundary prob from Phase-1 head; d: predicted unit directions
w_s = torch.exp(-dist / r)                                  # spatial term
cos = ((nbr_xyz - xyz.unsqueeze(2)) * d.unsqueeze(2)).sum(-1) / (dist + 1e-8)
w_c = torch.exp(pb - 1.0).unsqueeze(-1) * cos               # guidance term
w   = (w_s + alpha * w_c).clamp(min=0)
w   = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
f_refined = (w.unsqueeze(-1) * nbr_feats).sum(2)            # then MLP + skip
```

Start with PTB's constants (α = 1.0, r = 0.125) but note their r was tuned for room-scale S3DIS blocks; CloSe scans are unit-scale humans, so r must be rescaled — a sensible initialization is r ≈ 2–4× the median nearest-neighbor distance of a 2048-point sample, and it's worth making r learnable per layer. As in PTB, interior points (P_b ≈ 0) degrade gracefully to distance-weighted smoothing, and the whole term is differentiable, so it trains end-to-end. Note the reference implementation additionally divides the cosine by distance and uses `exp(−c(1−P_b))` — functionally the paper's Eq. 2; copy from `pointnet2_utils_boundary.py` lines ~250–280 when in doubt.

An alternative placement — a **boundary-aware EdgeConv** that reweights or masks neighbors inside the encoder — is tempting but riskier: EdgeConv's kNN is in feature space and aggregates via max, not a weighted sum, so injecting PTB's weights changes the operator semantics. Keep this as a stretch experiment; the post-decoder BRM is the faithful adaptation.

### Phase 3 (optional, zero-training baseline) — SegFix-style test-time relabeling

PTB's direction scheme descends from SegFix. Before touching training code at all, you can build a post-processing baseline: after Phase-1 training, for every point with P_b > τ (say 0.7), walk along d_i by a step s (≈ local point spacing) and copy the predicted label of the nearest point at x_i + s·d_i (optionally iterate 2–3 steps). This costs one kNN query at inference, gives you a lower bound on what boundary/direction information is worth, and is a good sanity check that the heads learned something meaningful before you invest in the BRM.

## 4. Ground-truth generation for CloSe-D

Port `preprocessing_boundaries.py` from the PTB repo almost verbatim into a new `prep_boundaries.py`. For every scan npz: build a KDTree on the full-resolution points, mark point i as boundary if any of its k = 4 nearest neighbors carries a different label (treating background −1 and unlabeled points carefully — either exclude them from the neighbor set or you'll mark every scan border as a garment boundary); then for every point, find its nearest boundary point and store the normalized vector from that boundary point to the point as GT direction. Cache `boundary` (N,) and `direction` (N, 3) arrays back into the npz, and extend `load_scan_npz` / the dataset `__getitem__` to surface them.

Two details matter more here than in S3DIS. **Compute GT at full scan resolution, before the 2048-point training subsample.** If you compute label-disagreement boundaries on the subsample, the boundary set becomes sampling-dependent noise; instead, precompute on the full scan and let the sampled points carry their precomputed labels/directions. Second, if you go the canonical-space route from Phase 1, compute the direction GT from `canon_pose` coordinates instead of posed points (boundary *membership* can be computed in either frame — labels are the same).

Also decide explicitly which boundaries count. For clothing, the garment–skin transitions and garment–garment seams are precisely the regions of interest, so the naive "any label disagreement" definition is right; but multi-layer outfits (open jacket over a shirt) create geometrically thin double boundaries where direction GT is ambiguous — expect noisy direction supervision there, and consider masking L_D on points whose nearest boundary is closer than the local point spacing.

## 5. Evaluation: measure the thing you're fixing

Global mIoU will move little (PTB saw +1.6 on KPConv); the story is at boundaries. Add to `lib/utils/metrics.py` a **boundary-region mIoU/accuracy**: restrict evaluation to points within radius ρ of any GT boundary point (report ρ at, say, 1 cm and 2.5 cm on the unit-normalized scans), following the boundary-IoU protocol popularized by Contrastive Boundary Learning. Report per-class IoU for boundary-heavy classes (shoes, socks, gloves, collars/hoodies vs. shirts). Replicate PTB's ablation table: baseline / +B / +D / +B+D with plain decoder / full with BRM — this tells you whether the refinement module earns its complexity on this architecture or whether Phase 1 already saturates the gain.

## 6. Concrete change map

| File | Change |
|---|---|
| `prep_boundaries.py` (new) | GT boundary + direction precomputation per scan, cached into npz (port of PTB `preprocessing_boundaries.py`) |
| `lib/closed/dataset.py` | Load `boundary`, `direction` fields; include in sampled batch; respect `filter_bg` |
| `lib/closenet/model.py` | Add `boundary_head`, `direction_head`; return `boundary_logits`, `pred_dirs`; Phase 2: insert BRM before `segm_dec` final layer |
| `lib/closenet/nets/` | New `brm.py` implementing guided aggregation (spatial kNN + PTB weights) |
| `lib/closenet/trainers/closenet_trainer.py` | Add `boundary_loss` (weighted BCE, β = 0.6) and `direction_loss` (MSE) to `step`; wire through `loss_weights` |
| `cfg/closenet.yaml` | `loss_weights: {segm_loss: 1.0, boundary_loss: 3.0, direction_loss: 0.3}`; BRM hyperparams (k, r, α); head channel specs |
| `lib/utils/metrics.py` | Boundary-region mIoU at radii ρ |

## 7. Risks and open questions

The **2048-point subsample** is the biggest threat: at that density a human scan has ~3–5 cm point spacing, so "boundary" becomes a fuzzy band and direction vectors are quantized. If Phase 1 results are flat, the first knob to turn is training density (CloSe's DGCNN is full-resolution anyway, so 4–8k points is mostly a memory question), or biasing the sampler to oversample near GT boundaries. Second, MSE direction loss competes with segmentation for shared-encoder capacity — PTB themselves note the three-task encoder is strained; monitor whether λ₂ > 0 ever hurts and consider a short warm-up (train segmentation-only for the first ~20 epochs, then enable auxiliary losses). Third, the garment-attention module already injects a strong class prior; there's a chance boundary supervision is partially redundant with it — the +B-only ablation will reveal this cheaply. Finally, keep an eye on PTB's own failure mode: guided propagation can *propagate* errors when boundary/direction predictions are wrong (their board and water classes regressed), so gate the BRM with predicted P_b confidence if you see it.

## 8. Suggested execution order

Week 1: GT preprocessing + dataset plumbing + boundary metrics (everything testable without training). Week 2: Phase 1 training runs + the ablation grid (baseline, +B, +D, +B+D), plus the Phase 3 relabeling baseline on the best checkpoint. Week 3: BRM implementation and r/α/k sweep. Decision point after week 2: if +B+D with the plain decoder already delivers most of the boundary-mIoU gain (as PTB's Table 3 hints it might), ship Phase 1 and treat BRM as a paper-style ablation rather than a product dependency.
