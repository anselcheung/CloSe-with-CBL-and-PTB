#!/bin/bash
#SBATCH --job-name=CloSeDi_eval
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/evaluate_%j.out
#SBATCH -e logs/evaluate_%j.err

# Evaluates the best base and CBL checkpoints (by validation mIoU) on the CloSe-Di
# test split and writes results/closenet_di_test_metrics.json. Meant to be submitted
# with --dependency=afterok:<train_base_jobid>:<train_cbl_jobid> so it only runs once
# both training jobs have finished successfully.

set -eo pipefail

cd /home/stud102/CloSe-with-CBL-and-PTB
mkdir -p logs results

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u evaluate_closenet_ckpts.py --output-json results/closenet_di_test_metrics.json

echo 'Test-split evaluation finished'
