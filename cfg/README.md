# Config files

All training configs share the same architecture (`cfg/closenet.yaml`'s `model_arch`)
and only differ in whether the contrastive boundary loss (CBL, arXiv:2203.05272) is
enabled, which feature it's applied to, and its loss weight. Each one is meant to be
run with `python train_closenet.py cfg/<name>.yaml` or `sbatch train_cbl_sweep.sbatch
cfg/<name>.yaml` (the latter also copies the best checkpoint to `pretrained/<name>.pt`).

## Naming convention

`closenet_cbl_<feature>_w<NNN>.yaml`, where:
- `<feature>` is omitted for the plain decoder-feature variant, `enc` for
  `encoder_stage3`, or `allstages` for `encoder_all_stages` (see below).
- `w<NNN>` encodes the `cbl_loss` weight as `weight * 100`, zero-padded to 3 digits
  (e.g. `w001` = 0.01, `w010` = 0.10, `w020` = 0.20).
- `_rerun` suffix marks an independent re-run of the same hyperparameters (different
  random init/data order - training never fixes a seed, see `lib/utils/misc.py:fix_seeds`
  which is only called by `demo.py`/`interactive_tool.py`, not `train_closenet.py`) with
  a fresh `exp_logs_path` so it doesn't resume from the first run's checkpoints.

## `training.cbl.feature_source`

Controls which model feature(s) `contrastive_boundary_loss` is computed on
(`lib/utils/metrics.py:select_cbl_features`):

| value | feature | notes |
|---|---|---|
| `decoder` (default if unset) | `MLPDecoder`'s second-to-last layer output | per-point-independent (`Conv1d(kernel_size=1)`, no neighbor mixing) |
| `encoder_stage3` | DGCNN encoder's 3rd/last `EdgeConv` stage output | produced via k-NN neighbor aggregation, unlike the decoder feature |
| `encoder_all_stages` | all 3 `EdgeConv` stage outputs, losses summed | mirrors the paper's Eq. 7 multi-scale form (`L = L_segm + weight * sum_n L_CBL^n`) |

## Files

| File | `cbl.enabled` | `feature_source` | `cbl_loss` weight | Purpose |
|---|---|---|---|---|
| `closenet.yaml` | false | - | - | Baseline, no boundary loss. Used by `train_base.sbatch`. |
| `closenet_cbl.yaml` | true | `decoder` | 0.1 (paper default) | Original CBL config as specified by the paper. Used by `train_cbl.sbatch`. |
| `closenet_cbl_w001.yaml` | true | `decoder` | 0.01 | Weight sweep: is 0.1 too strong for this architecture? |
| `closenet_cbl_w003.yaml` | true | `decoder` | 0.03 | Weight sweep. |
| `closenet_cbl_w005.yaml` | true | `decoder` | 0.05 | Weight sweep. |
| `closenet_cbl_enc_w001.yaml` | true | `encoder_stage3` | 0.01 | Does CBL work better on a feature with real neighbor aggregation? |
| `closenet_cbl_enc_w003.yaml` | true | `encoder_stage3` | 0.03 | " |
| `closenet_cbl_enc_w005.yaml` | true | `encoder_stage3` | 0.05 | " |
| `closenet_cbl_enc_w010.yaml` | true | `encoder_stage3` | 0.10 | " |
| `closenet_cbl_enc_w020.yaml` | true | `encoder_stage3` | 0.20 | " |
| `closenet_cbl_allstages_w001.yaml` | true | `encoder_all_stages` | 0.01 | Paper-style multi-scale CBL (all 3 encoder stages). |
| `closenet_cbl_allstages_w003.yaml` | true | `encoder_all_stages` | 0.03 | " |
| `closenet_cbl_allstages_w001_rerun.yaml` | true | `encoder_all_stages` | 0.01 | Reproducibility check for the w001 result above. |
| `closenet_cbl_allstages_w003_rerun.yaml` | true | `encoder_all_stages` | 0.03 | Reproducibility check for the w003 result above. |
| `data_split.json` | - | - | - | Train/val/test scan file lists for CloSe-Di. |
| `data_split_thuman.json` | - | - | - | Train/val/test split for the THuman2.0-derived, CloSe-Di-schema dataset built by `prep_thuman.py` (`data/CloSe-Di-THuman/`). |

## Findings so far (test-split, CloSe-Di)

All deltas are CBL vs. that run's own paired baseline evaluation (baseline itself
varies by ~0.003-0.005 across repeated evals due to eval-time point resampling, so
treat anything smaller than that as noise):

- **`decoder`**: harmful at every tested weight; harm grows monotonically with weight
  (0.10 is worst: ΔmIoU -0.0046, Δboundary_iou -0.0327). Decoder features are
  per-point-independent, so the loss has no neighbor-aggregation mechanism to act
  through.
- **`encoder_stage3`**: roughly noise-level at every weight (never clearly better or
  worse than baseline); peaks (least-bad) around weight 0.03, degrades again above it.
- **`encoder_all_stages`**: the only configuration with results clearly above the noise
  floor - weight 0.01 gives Δboundary_iou +0.0069, weight 0.03 gives ΔmIoU +0.0063.
