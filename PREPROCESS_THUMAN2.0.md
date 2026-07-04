# Preprocessing THuman2.0 into the CloSe-Di format

This guide walks you through turning the raw **THuman2.0** scans into per-scan `.npz`
files that match the **CloSe-Di** schema, and folding them into a combined
train/val/test split so THuman2.0 can augment CloSeNet training.

Everything is driven by a single script, [`prep_thuman.py`](prep_thuman.py), which
*wraps* the existing preprocessing functions in [`prep_scan.py`](prep_scan.py) — no
existing method, module, or config is modified. The original `cfg/data_split.json`
is never touched; a **new** split file is written.

> **TL;DR** — once the data below is in place and the `close` conda env is active:
> ```bash
> python prep_thuman.py --bm_dir_path /home/<user>/CloSe-with-CBL-and-PTB/models --stage all
> ```
> produces `data/THuman2.0_preprocessed/*.npz` and `cfg/data_split_thuman.json`.
> Point your training config's `split_file` at `cfg/data_split_thuman.json` to use it.

---

## 1. Environment

Preprocessing needs the same GPU stack as `prep_scan.py`: **CUDA + PyTorch +
PyTorch3D + smplx + trimesh**. Use the repo's `close` conda env (defined in
[`env.yml`](env.yml)); it already pins `pytorch=1.11 (cu113)`, `pytorch3d=0.6.2`,
`smplx==0.1.28`, `trimesh=3.20.1`.

```bash
conda env create -f env.yml     # first time only
conda activate close
```

> ⚠️ This must run on a **CUDA GPU node** (e.g. the TCML cluster). `prep_scan.py`'s
> internal texture-color step has a hardcoded `.cuda()` call, so `--device cpu` dry
> runs will fail there. There is a ready-made SLURM wrapper:
> [`tcml_job_prep_thuman.sh`](tcml_job_prep_thuman.sh).

---

## 2. Required data & models

Place everything under `data/` and `models/` as shown. **Sizes are approximate.**
Note: `models/` is git-ignored (SMPL's license forbids redistribution), so it never
arrives via `git pull` — you must download and place it yourself.

| What | Where it goes | Source |
|------|---------------|--------|
| **SMPL neutral body model** (`SMPL_NEUTRAL.pkl`) | `models/smpl/SMPL_NEUTRAL.pkl` | [SMPL website](https://smpl.is.tue.mpg.de/) → download **"SMPL for Python users" v1.0.0**, then rename `basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl` → `SMPL_NEUTRAL.pkl`. See [smplx setup docs](https://github.com/vchoutas/smplx#model-loading). |
| **THuman2.0 scans** (`.obj` + `material0.jpeg` + `.mtl`, one folder per scan `0000`…`0525`) | `data/THuman2.0_Release_copy/<id>/` | [THuman2.0 dataset](https://github.com/ytrock/THuman2.0-Dataset) — fill out their [request form](https://github.com/ytrock/THuman2.0-Dataset#agreement) to get the download link. |
| **THuman2.0 SMPL fits** (`<id>_smpl.pkl`, keys: `betas`,`body_pose`,`global_orient`,`transl`,`scale`) | `data/THuman2.0_smpl/` | Same [THuman2.0 repo](https://github.com/ytrock/THuman2.0-Dataset) — the **SMPL** (not SMPL-X) fittings download. |
| **THuman2.0 labels** (`scan_id → (N,) int label array`) | `data/CloSe-D++_updated/THuman2.0_labels.npz` | [CloSe project page](https://virtualhumans.mpi-inf.mpg.de/close3dv24/) / the CloSe-D++ release. |
| **CloSe-Di preprocessed scans** + existing split `cfg/data_split.json` | `data/CloSe-Di/` (already in repo) | [CloSe project page](https://virtualhumans.mpi-inf.mpg.de/close3dv24/). |

### About `--bm_dir_path` (the SMPL model directory)

This is the **only required argument** and the one most people get wrong.
`prep_scan.py` calls `smplx.create(model_path=<bm_dir_path>, model_type='smpl')`,
which expects `bm_dir_path` to be the **parent of a `smpl/` folder**:

```
models/                    ← this is --bm_dir_path
└── smpl/
    └── SMPL_NEUTRAL.pkl
```

So pass the `models/` directory, **not** `models/smpl/`. On the cluster use the
**absolute path** inside your repo clone, e.g.
`/home/<user>/CloSe-with-CBL-and-PTB/models`. If the path is wrong, smplx fails with
a confusing `Unknown model type <basename>, exiting!` — that means it couldn't find
the directory and fell back to treating the last path segment as a model type.

### Expected layout after setup

```
CloSe-with-CBL-and-PTB/
├── models/
│   └── smpl/SMPL_NEUTRAL.pkl
├── data/
│   ├── THuman2.0_Release_copy/0000/{0000.obj, material0.jpeg, material0.mtl} … 0525/
│   ├── THuman2.0_smpl/0000_smpl.pkl … 0525_smpl.pkl
│   ├── CloSe-D++_updated/THuman2.0_labels.npz
│   └── CloSe-Di/…              # existing CloSe-Di scans
├── cfg/data_split.json         # existing CloSe-Di split (never modified)
└── prep_thuman.py
```

Of the 526 scans, **~500 have labels**; the ~26 without a label entry are skipped
and logged automatically — that is expected, not an error.

---

## 3. How to run

The script has three stages via `--stage`:

- `prep` — preprocess scans → one `.npz` per scan in `--output_dir`.
- `split` — scan `--output_dir` for `.npz` files and write the combined split file.
- `all` — do `prep` then `split` (default).

### Step 0 — Single-scan sanity check (recommended first)

Confirms the env, paths, and SMPL model are wired up, and that vertex/label counts
match, before committing to the full batch:

```bash
python prep_thuman.py \
  --bm_dir_path /home/<user>/CloSe-with-CBL-and-PTB/models \
  --scan_ids 0000 --stage prep --overwrite
```

Then spot-check the output:

```python
import numpy as np
d = np.load('data/THuman2.0_preprocessed/0000.npz')
print(d['points'].shape, d['labels'].shape)   # -> (302021, 3) (302021,)  (must match)
print(d['garments'])                            # -> 18-vector, 1s at present classes
```

### Step 1 — Small batch smoke test (optional)

```bash
python prep_thuman.py \
  --bm_dir_path /home/<user>/CloSe-with-CBL-and-PTB/models \
  --limit 20 --stage prep --overwrite
```

Inspect the summary line in `data/THuman2.0_preprocessed/prep_thuman_log.txt`, e.g.
`SUMMARY: {'ok': 19, 'skip_no_labels': 1} over 20 scans in 207.0s`. (~10 s/scan, so
the full 526 takes ~1.5 h.)

### Step 2 — Full run

```bash
python prep_thuman.py \
  --bm_dir_path /home/<user>/CloSe-with-CBL-and-PTB/models \
  --stage all
```

Or submit it on the cluster (edit `BM_DIR_PATH` inside first):

```bash
sbatch tcml_job_prep_thuman.sh
```

Reruns are **idempotent** — scans whose `.npz` already exists are skipped unless you
pass `--overwrite`. So you can safely re-run after fixing a few scans.

### Rebuilding just the split

If preprocessing is already done and you only want to (re)build the split file:

```bash
python prep_thuman.py \
  --bm_dir_path /home/<user>/CloSe-with-CBL-and-PTB/models \
  --stage split
```

> Note: `--bm_dir_path` is required by the argument parser even for `--stage split`
> (which doesn't load the SMPL model), so you still have to pass it. `split` discovers
> preprocessed scans by scanning `--output_dir` for `.npz` files, so run it **after**
> the full `prep`, or your split will only contain whatever has been preprocessed so far.

---

## 4. Outputs

- **`data/THuman2.0_preprocessed/<id>.npz`** — one per successfully processed scan.
  Fields match the CloSe-Di schema and load directly through the existing
  `lib/closed/dataset.py` loader:

  | key | dtype / shape |
  |-----|---------------|
  | `points` | f64 `(N,3)` |
  | `colors` | f32 `(N,3)` |
  | `normals` | f32 `(N,3)` |
  | `labels` | i64 `(N,)` |
  | `canon_pose` | f64 `(N,3)` |
  | `garments` | i32 `(18,)` — 1 at each class present in `labels` |
  | `faces` | i64 `(F,3)` |
  | `betas` | f32 `(10,)` |
  | `pose` | f32 `(72,)` |
  | `trans` | f32 `(3,)` |
  | `scale` | f64 scalar |

- **`cfg/data_split_thuman.json`** — same 3-key schema as `cfg/data_split.json`
  (`train`/`val`/`test`). All original CloSe-Di entries are preserved verbatim; the
  labeled THuman scans are added on top with a fixed-seed (42) **80/10/10** split.
  Expect roughly `+400 / +50 / +50` over the original `1164 / 146 / 146`.

- **`data/THuman2.0_preprocessed/prep_thuman_log.txt`** — timestamped log with a
  per-status summary. Statuses: `ok`, `skip_exists`, `skip_no_labels`,
  `skip_no_smpl`, `skip_mismatch`, `skip_error`.

To train on the combined dataset, set `split_file: cfg/data_split_thuman.json` in
your training config (e.g. a copy of `cfg/closenet.yaml`) — no loader changes needed.

---

## 5. All CLI options

| flag | default | purpose |
|------|---------|---------|
| `--bm_dir_path` | *(required)* | SMPL body model dir (parent of `smpl/SMPL_NEUTRAL.pkl`) |
| `--thuman_root` | `data/THuman2.0_Release_copy` | raw scan folders |
| `--smpl_dir` | `data/THuman2.0_smpl` | per-scan SMPL fits |
| `--labels_file` | `data/CloSe-D++_updated/THuman2.0_labels.npz` | labels npz |
| `--output_dir` | `data/THuman2.0_preprocessed` | where `.npz` are written |
| `--closedi_split` | `cfg/data_split.json` | existing split to extend (read-only) |
| `--out_split` | `cfg/data_split_thuman.json` | combined split to write |
| `--stage` | `all` | `prep` / `split` / `all` |
| `--device` | `cuda` | passed to the mesh/SMPL loaders |
| `--seed` | `42` | THuman 80/10/10 shuffle seed |
| `--scan_ids` | *(all)* | process only these IDs (e.g. `0000 0001`) |
| `--limit` | *(none)* | cap number of scans (quick tests) |
| `--overwrite` | off | reprocess even if the `.npz` exists |
| `--log_file` | `<output_dir>/prep_thuman_log.txt` | log path |

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Unknown model type <x>, exiting!` | `--bm_dir_path` is wrong. Point it at the **parent** of `smpl/` (i.e. `.../models`), using an absolute path on the cluster. Verify `<bm_dir_path>/smpl/SMPL_NEUTRAL.pkl` exists. |
| `run conda init before conda activate` in SLURM logs | Harmless, but the provided `.sh` files avoid it by using `eval "$(conda shell.bash hook)"` instead of `source ~/.bashrc`. |
| Many `skip_no_labels` | Expected — only ~500/526 scans have labels. |
| `skip_mismatch` on a scan | Its label array length ≠ mesh vertex count; that scan is skipped (geometry must not be remeshed/decimated — vertex order has to match the labels). |
| `models/` missing after `git pull` | It's git-ignored by design (SMPL license). Download and place `SMPL_NEUTRAL.pkl` yourself (§2). |
| CPU run fails | Preprocessing requires a CUDA GPU (see §1). |
