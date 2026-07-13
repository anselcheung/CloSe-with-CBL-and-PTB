# CloSeNet ablations

This branch (`dev`) combines two independent additions to CloSeNet:

- **PTB** (Push-the-Boundary, `closenet_ptb_boundary_brief.md`): auxiliary boundary +
  direction heads, plus an optional SegFix-style test-time post-processing step.
- **CBL** (Contrastive Boundary Loss, arXiv:2203.05272, `cfg/README.md`): a contrastive
  loss that pulls same-label neighbor features together and pushes different-label
  ones apart.

Both are config-gated additions to the same base architecture (`cfg/closenet.yaml`'s
`model_arch`) — every ablation below is just `python train_closenet.py cfg/<name>.yaml`
with a different config, no code changes. Everything runs through the same two
entrypoints:

```bash
python train_closenet.py cfg/<config>.yaml                             # train
python evaluate_closenet_ckpts.py --config cfg/<config>.yaml --ckpt <ckpt>   # single-checkpoint eval
python evaluate_closenet_ckpts.py --base-config ... --cbl-config ...   # paired eval
```

## 1. Running training with sbatch

There's no single generic sbatch script — copy whichever of these is closest to what
you're running and point it at your config:

| Template | What it does |
|---|---|
| [`train_base.sbatch`](train_base.sbatch) | Trains one hardcoded config (`cfg/closenet.yaml`), copies the best checkpoint to `pretrained/closenet_base.pt`. |
| [`train_cbl.sbatch`](train_cbl.sbatch) | Same pattern for `cfg/closenet_cbl.yaml` -> `pretrained/closenet_cbl.pt`. |
| [`train_cbl_sweep.sbatch`](train_cbl_sweep.sbatch) | Generic: takes a config path as `$1`, trains it, copies the best checkpoint to `pretrained/<config-stem>.pt`. Use this for any new ablation instead of writing a new sbatch file. |

```bash
sbatch train_cbl_sweep.sbatch cfg/closenet_ptb.yaml
sbatch train_cbl_sweep.sbatch cfg/closenet_ptb_cbl.yaml
```

For evaluation, use [`evaluate_single.sbatch`](evaluate_single.sbatch) (single checkpoint
against any config — baseline / PTB / PTB+postproc / CBL / PTB+CBL) or one of the other
`evaluate_*.sbatch` files for paired/batched test-split scoring:

| Template | What it does |
|---|---|
| [`evaluate_single.sbatch <config> <ckpt> [output_json]`](evaluate_single.sbatch) | Generic: scores one checkpoint against one config, e.g. `sbatch evaluate_single.sbatch cfg/closenet_test_ptb_postproc.yaml closenet_ptb_train/checkpoints/<BEST>.pt`. |
| [`evaluate.sbatch`](evaluate.sbatch) | Scores the best base + best CBL checkpoint on the test split, writes one JSON. |
| [`evaluate_sweep.sbatch`](evaluate_sweep.sbatch) | Same, hardcoded to the `w001`/`w003`/`w005` weight sweep. |
| [`evaluate_variants.sbatch <tag>...`](evaluate_variants.sbatch) | Generic: `sbatch evaluate_variants.sbatch enc_w001 allstages_w003 ...` scores any list of `cfg/closenet_cbl_<tag>.yaml` variants. |

All sbatch templates `cd` to the repo root, activate the `close` conda env, and
`unset MKL_INTERFACE_LAYER` (needed for `set -eo pipefail` to survive the conda
activation hook on TCML) — copy that boilerplate as-is for any new script.

**Prerequisite for anything involving PTB** (boundary/direction heads, `+B`, `+D`,
post-processing): the scans need precomputed `boundary`/`direction`/`boundary_dist`
fields, added in-place by `prep_boundaries.py`. Run once, before training, via
[`prep_boundaries.sbatch`](prep_boundaries.sbatch):

```bash
sbatch prep_boundaries.sbatch cfg/data_split.json           # CloSe-Di
sbatch prep_boundaries.sbatch cfg/data_split_thuman.json     # THuman2.0-derived split
```

CBL and the baseline need no such step (though running it on their splits too is
harmless and lets `evaluate_closenet_ckpts.py` report `boundary_mIoU@rho` for them as a
reference number). The THuman2.0-derived split itself is built by
[`prep_thuman.sbatch`](prep_thuman.sbatch) (see `thuman2_preprocessing_brief.md`).

## 2. What gets tracked, and where

**During training** (`BaseTrainer.train_model`, `lib/closenet/trainers/base_trainer.py`):
- Every `val_it` iterations, `compute_val_loss` runs a full validation pass and logs to
  TensorBoard under `<exp_logs_path>/summary` (`tensorboard --logdir <exp_logs_path>/summary`):
  - `val/<loss_name>` for every entry in `loss_dict` (`segm_loss`, and whichever of
    `boundary_loss`, `direction_loss`, `cbl_loss`, `boundary_iou` are enabled)
  - `val/mIoU`, `val/freq_IoU`, `val/IoU/<class_idx>`
  - `val/boundary_mIoU@<rho>` for each radius in `eval.boundary_radii`, **only when the
    scans carry `boundary_dist`** (i.e. `prep_boundaries.py` has been run) — reported
    even for the baseline/CBL-only runs as a reference number, not just PTB runs
  - `train/<loss_name>` once per epoch (mean over the epoch)
- A checkpoint is written to `<exp_logs_path>/checkpoints/valmin_<epoch>_<it>_val=<total_loss>_<metric>=<value>...pt`
  every time a metric in `training.selection_metric` (comma-separated, e.g.
  `mIoU,freq_IoU` or `mIoU,freq_IoU,boundary_iou`) improves — `training.selection_mode`
  (`maximize`/`minimize`) decides the direction. This is what `find_best_checkpoint`
  (in `evaluate_closenet_ckpts.py` and the sbatch "copy best checkpoint" steps) parses
  by `mIoU=` in the filename, so **always keep `mIoU` in `selection_metric`**.

**At test time** — one script, two modes, both routed through
`BaseTrainer.evaluate_model()` so every ablation gets the same full metric set:
`segm_loss_mean`, `mIoU`, `IoU_per_class`, `freq_IoU`, `boundary_iou_mean`,
`mIoU_boundary_mean`, `mIoU_inner_mean` (all reported regardless of ablation — B-IoU/
mIoU@boundary/mIoU@inner only need points+preds+GT labels, not CBL), `cbl_loss_mean`
(only when `training.cbl.enabled`), and any `boundary_mIoU@<rho>` keys (whenever the
scans have `boundary_dist`):
- `evaluate_closenet_ckpts.py --config cfg/<test_config>.yaml --ckpt <ckpt>` —
  single checkpoint against one config (via `evaluate_single.sbatch`). Use the matching
  `cfg/closenet_test_*.yaml` for the checkpoint's mode (see the table in §3) — this is
  what carries PTB aux-head reconstruction and SegFix post-processing.
- `evaluate_closenet_ckpts.py --base-config ... --cbl-config ...` — paired base-vs-CBL
  comparison (via `evaluate.sbatch`/`evaluate_sweep.sbatch`/`evaluate_variants.sbatch`),
  written to a JSON plus a printed summary. `--cbl-config` swaps in any
  `cfg/closenet_cbl_*.yaml` variant.

## 3. Ablation matrix

### Core ablations (architecture-level)

| Ablation | Config | Key(s) to set | sbatch |
|---|---|---|---|
| Baseline | `cfg/closenet.yaml` | `model_arch.aux_heads` absent, `training.cbl.enabled: false` | `train_base.sbatch` |
| PTB, boundary head only (+B) | copy `cfg/closenet_ptb.yaml`, set `training.loss_weights.direction_loss: 0.0` | head still exists, but only `boundary_loss` shapes the encoder | `train_cbl_sweep.sbatch cfg/closenet_ptb_bonly.yaml` |
| PTB, direction head only (+D) | copy `cfg/closenet_ptb.yaml`, set `training.loss_weights.boundary_loss: 0.0` | only `direction_loss` shapes the encoder | `train_cbl_sweep.sbatch cfg/closenet_ptb_donly.yaml` |
| PTB, both heads (+B+D) | `cfg/closenet_ptb.yaml` | `model_arch.aux_heads.enabled: true`, `training.loss_weights.{boundary_loss: 3.0, direction_loss: 0.3}` | `train_cbl_sweep.sbatch cfg/closenet_ptb.yaml` |
| PTB + SegFix post-processing | train with `cfg/closenet_ptb.yaml`; **evaluate** with `cfg/closenet_test_ptb_postproc.yaml` | `post_processing.{enabled: true, threshold: 0.7, step: null, n_iters: 2}` — eval-only, no retraining | `python evaluate_closenet_ckpts.py --config cfg/closenet_test_ptb_postproc.yaml --ckpt <ptb_ckpt>.pt` |
| CBL | `cfg/closenet_cbl.yaml` | `training.cbl.enabled: true`, `training.loss_weights.cbl_loss: 0.1` | `train_cbl.sbatch` or `train_cbl_sweep.sbatch cfg/closenet_cbl.yaml` |
| PTB + CBL combined | `cfg/closenet_ptb_cbl.yaml` | both `model_arch.aux_heads` and `training.cbl` blocks set together | `train_cbl_sweep.sbatch cfg/closenet_ptb_cbl.yaml` |

`+B`/`+D`-only variants and the `_bonly`/`_donly` yamls don't exist yet as files —
copy `cfg/closenet_ptb.yaml`, rename, zero the one loss weight, and give it a unique
`exp_logs_path` (see `train_cbl_sweep.sbatch <config>` above for the run recipe).

Reproducing PTB's own ablation table (`closenet_ptb_boundary_brief.md` §5): baseline /
+B / +D / +B+D / +B+D+postproc, comparing `boundary_mIoU@0.0056`/`@0.014` and
`boundary_iou` across the five.

### CBL-side ablations

Full detail and current findings in [`cfg/README.md`](cfg/README.md); summary:

- **`training.cbl.feature_source`** — which feature the contrastive loss is computed
  on (`lib/utils/metrics.py:select_cbl_features`):
  - `decoder` (default) — MLPDecoder's second-to-last layer, per-point-independent
  - `encoder_stage3` — DGCNN's last EdgeConv stage output (real neighbor aggregation)
  - `encoder_all_stages` — all 3 EdgeConv stages, losses summed (paper's Eq. 7 form)
- **`training.loss_weights.cbl_loss`** — weight sweep, existing configs cover
  0.01/0.03/0.05/0.10/0.20 depending on `feature_source` (see the file table in
  `cfg/README.md`)
- **`training.cbl.{k, radius, temperature}`** — k-NN neighborhood size, radius cap,
  and InfoNCE temperature (defaults 40 / 0.1 / 1.0), not yet swept in any existing config
- Naming convention for a new sweep point: `closenet_cbl_<feature>_w<NNN>.yaml`
  (`<feature>` omitted for `decoder`; `w<NNN>` = weight × 100, zero-padded to 3 digits);
  add `_rerun` for a repeat with a fresh `exp_logs_path` (no seed is fixed, see
  `cfg/README.md`)

Run any variant via `sbatch train_cbl_sweep.sbatch cfg/closenet_cbl_<variant>.yaml`,
then score it with `sbatch evaluate_variants.sbatch <variant>` (or add it to
`evaluate_sweep.sbatch`/`evaluate_variants.sbatch` for a batched run).

### Other useful knobs (any ablation)

- `data.split_file` — `cfg/data_split.json` (CloSe-Di) vs. `cfg/data_split_thuman.json`
  (THuman2.0-derived, needed for PTB training data volume — see
  `thuman2_preprocessing_brief.md`)
- `data.pointcloud_samples` — training subsample density (default 2048); PTB's brief
  flags this as the first knob to try if boundary results are flat
  (`closenet_ptb_boundary_brief.md` §7)
- `training.boundary_beta` — up-weight of the boundary class in the boundary-head CE
  (default 0.6)
- `training.dir_mask_dist` — drop the direction loss for points within this distance
  of a boundary (default `null`, i.e. off)
- `eval.boundary_radii` — radii (in unit-bbox units) for `boundary_mIoU@rho` reporting
  (default `[0.0056, 0.014]`, ≈1cm/2.5cm on a unit-normalized human)
- `exp_logs_path` — **always** set a unique value per run; checkpoints/TensorBoard logs
  live there and a shared path will collide across runs
