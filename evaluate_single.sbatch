#!/bin/bash
#SBATCH --job-name=CloSeDi_eval_single
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/evaluate_single_%j.out
#SBATCH -e logs/evaluate_single_%j.err

# Evaluates one checkpoint against one config on the CloSe-Di test split - covers
# baseline / PTB / PTB+SegFix-postproc (and CBL / PTB+CBL, though evaluate.sbatch /
# evaluate_variants.sbatch are the paired-comparison equivalents for those). Routes
# through evaluate_closenet_ckpts.py's --config/--ckpt single-checkpoint mode, which
# reports mIoU, freq_IoU, boundary_iou, mIoU@boundary, mIoU@inner, and (when the scans
# carry boundary_dist, i.e. prep_boundaries.py has been run) boundary_mIoU@rho.
#
# Usage: sbatch evaluate_single.sbatch <config.yaml> <checkpoint.pt> [<output_json>]
#
# Examples (mirrors the three modes the old tcml_job_eval.sh cycled through):
#   sbatch evaluate_single.sbatch cfg/closenet_test_baseline.yaml pretrained/closenet.pth
#   sbatch evaluate_single.sbatch cfg/closenet_test_ptb.yaml closenet_ptb_train/checkpoints/<BEST>.pt
#   sbatch evaluate_single.sbatch cfg/closenet_test_ptb_postproc.yaml closenet_ptb_train/checkpoints/<BEST>.pt

set -eo pipefail

CONFIG="${1:?Usage: sbatch evaluate_single.sbatch <config.yaml> <checkpoint.pt> [<output_json>]}"
CHECKPOINT="${2:?Usage: sbatch evaluate_single.sbatch <config.yaml> <checkpoint.pt> [<output_json>]}"
OUTPUT_JSON="${3:-results/$(basename "${CONFIG%.yaml}")_eval.json}"

# SLURM sets this to the directory `sbatch` was invoked from - no hardcoded path needed.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u evaluate_closenet_ckpts.py \
    --config "$CONFIG" \
    --ckpt "$CHECKPOINT" \
    --output-json "$OUTPUT_JSON"

echo "Single-checkpoint evaluation finished ($CONFIG, $CHECKPOINT)"
