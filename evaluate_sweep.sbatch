#!/bin/bash
#SBATCH --job-name=CloSeDi_eval_sweep
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/evaluate_sweep_%j.out
#SBATCH -e logs/evaluate_sweep_%j.err

# Evaluates the three cbl_loss-weight sweep checkpoints (w001, w003, w005) on the
# CloSe-Di test split, one results JSON per variant. All three fit in a single short
# job since each is just a forward pass over ~146 test shapes.

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

for w in w001 w003 w005; do
    echo "=== evaluating $w ==="
    python -u evaluate_closenet_ckpts.py \
        --cbl-config "cfg/closenet_cbl_${w}.yaml" \
        --output-json "results/closenet_di_test_metrics_${w}.json"
done

echo 'Sweep evaluation finished'
