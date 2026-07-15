# Implementation Brief: Closing the PTB Gaps (Phase 2 BRM + Phase 3 postproc fix)

**Context.** `closenet_ptb_boundary_brief.md` staged the PTB port in three phases. Phase 1
(auxiliary boundary/direction heads + GT generation + losses) is implemented and correct. Phase 2
(the guided-propagation mechanism PTB is actually named for) was never implemented — only
sketched. Phase 3 (`lib/closenet/postprocess.py`) is implemented but (a) is a repo-original
heuristic, not a PushBoundary port, and (b) has a real bug: its `n_iters` loop re-applies a single
precomputed neighbour lookup instead of walking further each iteration. This brief specifies both
fixes concretely against the current code.

---

## Part A — Phase 2: Boundary-aware Refinement Module (BRM)

### A.1 Where it plugs in

`lib/closenet/model.py::CloSeNet._decode` currently does:

```python
logits, decoder_features = self.segm_dec(feat, return_features=True, **kwargs)
```

`MLPDecoder` (`lib/closenet/nets/mlp.py`) builds its conv stack from
`[segm_input] + segm_dec_cfg.channels + [n_classes]`. With the shipped config
(`cfg/closenet_ptb.yaml`, `segm_dec.channels: [512,512,512,512,256]`), `decoder_features` is the
**256-d tensor fed into the final classifier layer** (`self.segm_dec.layers[-1]`, a
`Conv1d(256, 18) + BN + LeakyReLU`) — this is exactly the "pre-logit feature" insertion point the
original brief's §Phase 2 pseudocode calls for ("between the penultimate decoder layer and the
final classifier").

BRM sits between `decoder_features` and that final layer:

```python
logits, decoder_features = self.segm_dec(feat, return_features=True, **kwargs)
out = EasierDict(logits=logits, decoder_features=decoder_features)
if self.use_aux_heads:
    out.boundary_logits = self.boundary_head(feat, **kwargs)
    out.pred_dirs = F.normalize(self.direction_head(feat, **kwargs), dim=1, eps=1e-8)
    if self.use_brm:
        pb = F.softmax(out.boundary_logits, dim=1)[:, 1, :]          # (B, N)
        xyz = feat_xyz  # points[..., :3], passed in from forward()
        refined = self.brm(decoder_features, xyz, pb, out.pred_dirs)  # (B, 256, N)
        out.logits = self.segm_dec.layers[-1](refined)                # reuse the trained classifier
```

Reusing `segm_dec.layers[-1]` (rather than adding a parallel head) mirrors PTB: GFP replaces what
feeds the *existing* classifier, it doesn't add a second one. `_decode` needs `xyz` threaded in —
currently only `feat` is passed; add `points_xyz=data.points[..., :3]` to the call from `forward`.

### A.2 New module: `lib/closenet/nets/brm.py`

```python
import torch
from torch import nn
import torch.nn.functional as F


class BoundaryRefinementModule(nn.Module):
    """Boundary-aware Refinement Module — CloSeNet's stand-in for PTB's guided feature
    propagation (GFP). CloSeNet has no upsampling stage to guide (unlike PointNet++/KPConv),
    so instead of reweighting an interpolation, this reweights a spatial-kNN aggregation of
    the pre-logit decoder features, using the same predicted-boundary / predicted-direction
    guidance signal as PTB Eq. 2 (`pointnet2_utils_boundary.py` lines ~262-278).
    """

    def __init__(self, channels: int, k: int = 6, alpha: float = 1.0, r_init: float = 0.05):
        super().__init__()
        self.k = k
        self.alpha = alpha
        # r is scale-sensitive (PTB tuned it for room-scale S3DIS in meters; CloSe scans are
        # unit-bbox normalised), so make it a learnable, per-module scalar seeded near
        # eval.boundary_radii's ~2.5cm figure rather than copying PTB's r=0.125 verbatim.
        self.log_r = nn.Parameter(torch.log(torch.tensor(float(r_init))))
        self.merge = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, feat, xyz, boundary_prob, pred_dirs):
        """
        feat:          (B, C, N) pre-logit decoder features
        xyz:           (B, N, 3) point positions
        boundary_prob: (B, N)    predicted P(boundary)
        pred_dirs:     (B, 3, N) predicted unit interior directions
        Returns:       (B, C, N) refined features (residual-merged with input)
        """
        B, C, N = feat.shape
        k = min(self.k, N)
        r = torch.exp(self.log_r).clamp(min=1e-4)

        dist = torch.cdist(xyz, xyz)                       # (B, N, N)
        nn_dist, nn_idx = dist.topk(k, dim=-1, largest=False)  # (B, N, k)

        nbr_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, N, -1, -1), 2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )                                                    # (B, N, k, 3)
        rel = nbr_xyz - xyz.unsqueeze(2)                      # (B, N, k, 3)
        cos = (rel * pred_dirs.permute(0, 2, 1).unsqueeze(2)).sum(-1)
        cos = cos / (nn_dist + 1e-8)                          # (B, N, k), matches PTB's dir_cos/dists

        w_spatial = torch.exp(-nn_dist / r)
        w_guide = torch.exp(boundary_prob - 1.0).unsqueeze(-1) * cos
        w = (w_spatial + self.alpha * w_guide).clamp(min=0.0)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)       # (B, N, k)

        feat_t = feat.permute(0, 2, 1)                        # (B, N, C)
        nbr_feat = torch.gather(
            feat_t.unsqueeze(1).expand(-1, N, -1, -1), 2,
            nn_idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )                                                      # (B, N, k, C)
        agg = (w.unsqueeze(-1) * nbr_feat).sum(2).permute(0, 2, 1)  # (B, C, N)

        return self.merge(agg) + feat                          # residual skip, per §Phase 2
```

Notes on faithfulness to PTB:
- `w_spatial + alpha * w_guide`, `clamp(min=0)`, L1-normalize, `cos` divided by distance — matches
  `pointnet2_utils_boundary.py::PointNetFeaturePropagation.forward`'s guided branch exactly, modulo
  operating on a fixed-radius spatial kNN instead of a distance-sorted upsampling kNN (there is no
  upsampling here, per the original brief's §2).
- **Do not** copy PTB's `pb = softmax(boundaries1, ...)` line literally: in the reference code
  `boundaries1` is already the *log-softmax'd* boundary output when it reaches `fp1`, so PTB
  applies softmax to already-log-softmaxed values — a quirk of their code, not part of the
  documented method. Use raw boundary logits → one `softmax` (as in `A.1`) to get a proper
  probability.
- `torch.cdist` on `(B, N, N)` is O(N²) — fine at `N=2048` (CloSe's training/eval subsample) but
  would need chunking if scans are ever run at full resolution.

### A.3 Config additions (`cfg/closenet_ptb.yaml`, new `cfg/closenet_ptb_brm.yaml`)

```yaml
model_arch:
  brm:
    enabled: true       # requires aux_heads.enabled: true
    k: 6
    alpha: 1.0
    r_init: 0.05        # ~4x median NN spacing of a 2048-pt unit-bbox scan; keep learnable
```

`model.py::__init__` should raise if `brm.enabled` and not `use_aux_heads` (same pattern as the
existing `postprocess`-vs-`aux_heads` guard in `closenet_trainer.py:115-120`), since BRM consumes
`boundary_logits`/`pred_dirs`.

### A.4 Verification

- Ablation grid from the original brief (§5, §8) is the right test: `baseline / +B+D (Phase 1) /
  +B+D+BRM`. Expect most of the boundary-mIoU gain from Phase 1 per PTB's own ablation numbers
  quoted in the brief (§1: "+0.7" for heads alone vs "+1.6" full GFP on KPConv/S3DIS) — BRM should
  close roughly the remaining third.
- Sanity check before any training run: with `alpha=0` the BRM must reduce to a plain
  distance-weighted local smoothing (no directional bias) — assert this numerically on a random
  batch (`w_guide` term zeroed out) to catch a broken formula early.
- Watch the failure mode the original brief flags (§7): guided aggregation can propagate wrong
  labels when `P_b`/direction predictions are themselves wrong. Log `mIoU_boundary` alongside
  overall mIoU during the sweep, not just after.

---

## Part B — Phase 3 postproc bug fix (`lib/closenet/postprocess.py`)

### B.1 The bug

```python
targets = xyz[mask] + step * dirs[mask]
_, nn_idx = tree.query(targets, k=1)
move_idx = np.nonzero(mask)[0]

out = labels.copy()
for _ in range(max(1, n_iters)):
    out[move_idx] = out[nn_idx]          # nn_idx is fixed for every iteration
```

`targets`/`nn_idx` are computed once from the *original* positions. Repeating the assignment only
changes anything when some entry of `nn_idx` also appears in `move_idx` (a relabeled point's own
neighbour target happens to be another boundary point) — otherwise `n_iters > 1` is a silent no-op
on top of `n_iters == 1`. The docstring and `closenet_ptb_boundary_brief.md` §Phase 3 both describe
"walk along d_i ... optionally iterate 2-3 steps," i.e. each iteration should advance the
walked-to position further along the direction and re-query, not repeat the same lookup.

### B.2 The fix

```python
def _relabel_one(labels, xyz, boundary_prob, dirs, threshold, step, n_iters):
    n = labels.shape[0]
    if n == 0:
        return labels

    tree = cKDTree(xyz)

    if step is None:
        nn_dist, _ = tree.query(xyz, k=min(2, n))
        if nn_dist.ndim == 2 and nn_dist.shape[1] > 1:
            median_spacing = float(np.median(nn_dist[:, 1]))
        else:
            median_spacing = 0.0
        step = 2.0 * median_spacing if median_spacing > 0 else 0.0

    mask = boundary_prob > threshold
    if not mask.any() or step == 0.0:
        return labels

    move_idx = np.nonzero(mask)[0]
    pos = xyz[mask].copy()
    out = labels.copy()

    # Walk further along the predicted direction each iteration and re-query the tree,
    # so n_iters actually controls how far (n_iters * step) a boundary point's label
    # search travels, and chains of adjacent boundary points can relabel each other
    # across iterations via the evolving `out` array.
    for _ in range(max(1, n_iters)):
        pos = pos + step * dirs[mask]
        _, nn_idx = tree.query(pos, k=1)
        out[move_idx] = out[nn_idx]

    return out
```

Key differences from the current code:
- `pos` accumulates `step * dirs[mask]` every iteration, so total walked distance after `i`
  iterations is `i * step` (matches "iterate 2-3 steps" meaning 2-3 hops of `step`, not one hop
  repeated).
- `tree.query` is re-run each iteration against the *moving* virtual position, not the fixed
  original `targets`.
- `out[nn_idx]` on iteration `i` reads labels as of iteration `i-1`, so a chain of adjacent
  boundary points can hand labels down the line across iterations — this was presumably the
  intended effect of `n_iters > 1` all along.
- `n_iters == 1` is unchanged in behavior/semantics from before (same first hop), so this is
  backward-compatible for any reported `n_iters=1` numbers; only `n_iters >= 2` results change.

### B.3 Verification

Add a small regression test (e.g. `tests/test_postprocess.py` if a test dir exists, else inline in
the module's `if __name__ == '__main__':`) with a synthetic 1-D chain of points where boundary
points sit between two clearly-labeled interior regions:

1. Construct points along a line, labels `[0,0,0,B,B,1,1,1]` where `B` are marked boundary with
   direction pointing toward the `1`-side.
2. Assert `n_iters=1` reproduces today's single-hop output exactly (no regression).
3. Assert `n_iters=2` on the fixed version actually differs from `n_iters=1` when a boundary
   point's first-hop target is itself another boundary point (the case the current code silently
   fails to handle) — this is the test that would have caught the original bug.
4. Re-run whatever eval numbers were previously reported for `cfg/closenet_test_ptb_postproc.yaml`
   (`n_iters: 2`) — expect them to shift now that the second iteration does real work.

---

## Suggested order of work

1. Fix `postprocess.py` (Part B) first — it's a 10-line, self-contained, low-risk change and
   corrects existing eval numbers.
2. Re-run the Phase-3 eval (`evaluate_closenet_ckpts.py --config cfg/closenet_test_ptb_postproc.yaml`)
   on the existing PTB checkpoint to get corrected baseline numbers before investing in Phase 2.
3. Implement `brm.py` + config wiring (Part A), run the ablation grid from §A.4.
