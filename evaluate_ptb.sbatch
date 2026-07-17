#!/bin/bash
#SBATCH --job-name=CloSeDi_eval_ptb
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o logs/evaluate_ptb_%j.out
#SBATCH -e logs/evaluate_ptb_%j.err

# Evaluates the PTB, PTB+CBL, PTB+BRM, and PTB+BRM+CBL checkpoints on the CloSe-Di test
# split (data_split_thuman.json), the PTB-family equivalent of evaluate.sbatch (which
# pairs baseline vs cbl). Runs four passes:
#   1. PTB vs PTB+CBL, no post-processing (paired mode, mirrors evaluate.sbatch)      -> results/closenet_di_test_metrics_ptb.json
#   2. PTB      + SegFix post-processing (single mode, same checkpoint as pass 1)     -> results/closenet_di_test_metrics_ptb_postproc.json
#   3. PTB+CBL  + SegFix post-processing (single mode, same checkpoint as pass 1)     -> results/closenet_di_test_metrics_ptb_cbl_postproc.json
#   4. PTB+BRM vs PTB+BRM+CBL, no post-processing (paired mode)                       -> results/closenet_di_test_metrics_ptb_brm.json
# model_arch (incl. aux_heads) is identical between cfg/closenet_ptb.yaml and
# cfg/closenet_ptb_cbl.yaml, so cfg/closenet_test_ptb_postproc.yaml is reused for both
# checkpoints in passes 2-3 - only --ckpt changes. BRM has no separate postproc config
# (cfg/closenet_test_ptb_brm.yaml already sets post_processing.enabled: false and is
# meant to be reused across both BRM checkpoints per its own header comment), so pass 4
# covers both PTB+BRM and PTB+BRM+CBL directly against their training configs, the same
# way pass 1 does for PTB/PTB+CBL.

set -eo pipefail

# SLURM sets this to the directory `sbatch` was invoked from - no hardcoded path needed.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Resolve the best-mIoU checkpoint in a checkpoints/ dir (same logic as
# evaluate_closenet_ckpts.py's find_best_checkpoint), so passes 1-3 all evaluate the
# exact same checkpoint per variant.
resolve_best_ckpt() {
    python - "$1" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
scored = []
for ckpt in ckpt_dir.glob('valmin_*.pt'):
    match = re.search(r'mIoU=([0-9.]+)', ckpt.name)
    if match:
        scored.append((float(match.group(1)), ckpt))
if not scored:
    raise SystemExit(f'No scored (valmin_*mIoU=*.pt) checkpoints found in {ckpt_dir}')
print(max(scored, key=lambda x: x[0])[1])
PY
}

PTB_CKPT="$(resolve_best_ckpt closenet_ptb_train/checkpoints)"
PTB_CBL_CKPT="$(resolve_best_ckpt closenet_ptb_cbl_train/checkpoints)"
PTB_BRM_CKPT="$(resolve_best_ckpt closenet_ptb_brm_train/checkpoints)"
PTB_BRM_CBL_CKPT="$(resolve_best_ckpt closenet_ptb_brm_cbl_train/checkpoints)"
echo "PTB checkpoint:         $PTB_CKPT"
echo "PTB+CBL checkpoint:     $PTB_CBL_CKPT"
echo "PTB+BRM checkpoint:     $PTB_BRM_CKPT"
echo "PTB+BRM+CBL checkpoint: $PTB_BRM_CBL_CKPT"

echo "=== 1/4: PTB vs PTB+CBL (no post-processing) ==="
python -u evaluate_closenet_ckpts.py \
    --base-config cfg/closenet_ptb.yaml --base-ckpt "$PTB_CKPT" \
    --cbl-config cfg/closenet_ptb_cbl.yaml --cbl-ckpt "$PTB_CBL_CKPT" \
    --output-json results/closenet_di_test_metrics_ptb.json

echo "=== 2/4: PTB + SegFix post-processing ==="
python -u evaluate_closenet_ckpts.py \
    --config cfg/closenet_test_ptb_postproc.yaml --ckpt "$PTB_CKPT" \
    --output-json results/closenet_di_test_metrics_ptb_postproc.json

echo "=== 3/4: PTB+CBL + SegFix post-processing ==="
python -u evaluate_closenet_ckpts.py \
    --config cfg/closenet_test_ptb_postproc.yaml --ckpt "$PTB_CBL_CKPT" \
    --output-json results/closenet_di_test_metrics_ptb_cbl_postproc.json

echo "=== 4/4: PTB+BRM vs PTB+BRM+CBL (no post-processing) ==="
python -u evaluate_closenet_ckpts.py \
    --base-config cfg/closenet_ptb_brm.yaml --base-ckpt "$PTB_BRM_CKPT" \
    --cbl-config cfg/closenet_ptb_brm_cbl.yaml --cbl-ckpt "$PTB_BRM_CBL_CKPT" \
    --output-json results/closenet_di_test_metrics_ptb_brm.json

echo 'PTB evaluation finished'
