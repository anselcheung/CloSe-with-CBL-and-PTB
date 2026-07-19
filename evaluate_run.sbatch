#!/bin/bash
#SBATCH --job-name=CloSeDi_eval_run
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/evaluate_run_%j.out
#SBATCH -e logs/evaluate_run_%j.err

# Evaluates a single training run (runs/<experiment_name>/<date_time>/, as produced by
# train_run.sbatch) against a chosen eval config on the CloSe-Di test split. This is the
# single-run analogue of evaluate_ptb.sbatch: same best-mIoU checkpoint auto-detection
# (resolve_best_ckpt below is copied verbatim from evaluate_ptb.sbatch), same underlying
# evaluate_closenet_ckpts.py --config/--ckpt call, but you pick exactly which run
# directory and which eval config to use instead of it always evaluating all four
# hardcoded PTB/PTB+CBL/PTB+BRM/PTB+BRM+CBL variants.
#
# Usage: sbatch evaluate_run.sbatch <run_dir> <eval_config.yaml> [output_json]
# Examples:
#   sbatch evaluate_run.sbatch runs/ptb/20260718_140322 cfg/closenet_test_ptb.yaml
#   sbatch evaluate_run.sbatch runs/ptb/20260718_140322 cfg/closenet_test_ptb_postproc.yaml
#   sbatch evaluate_run.sbatch runs/ptb_brm/20260718_141900 cfg/closenet_test_ptb_brm.yaml results/ptb_brm_run1.json

set -eo pipefail

RUN_DIR="${1:?Usage: sbatch evaluate_run.sbatch <run_dir> <eval_config.yaml> [output_json]}"
EVAL_CONFIG="${2:?Usage: sbatch evaluate_run.sbatch <run_dir> <eval_config.yaml> [output_json]}"

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
# evaluate_closenet_ckpts.py's find_best_checkpoint / evaluate_ptb.sbatch's copy of it).
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

CKPT="$(resolve_best_ckpt "$RUN_DIR/checkpoints")"
echo "Run directory:  $RUN_DIR"
echo "Eval config:    $EVAL_CONFIG"
echo "Checkpoint:     $CKPT"

EXP_NAME="$(basename "$(dirname "$RUN_DIR")")"
TIMESTAMP="$(basename "$RUN_DIR")"
CONFIG_STEM="$(basename "${EVAL_CONFIG%.yaml}")"
OUTPUT_JSON="${3:-results/${EXP_NAME}_${TIMESTAMP}_${CONFIG_STEM}.json}"

python -u evaluate_closenet_ckpts.py \
    --config "$EVAL_CONFIG" \
    --ckpt "$CKPT" \
    --output-json "$OUTPUT_JSON"

echo "Single-run evaluation finished ($RUN_DIR, $EVAL_CONFIG) -> $OUTPUT_JSON"
