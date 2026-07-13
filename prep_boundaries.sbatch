#!/bin/bash
#SBATCH --job-name=CloSeDi_prep_boundaries
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --time=12:00:00
#SBATCH -o logs/prep_boundaries_%j.out
#SBATCH -e logs/prep_boundaries_%j.err

# Precomputes GT boundary/direction/boundary_dist fields (prep_boundaries.py) in-place
# on every scan referenced by a split file. Required once before training/evaluating
# any PTB config (aux heads need boundary/direction; boundary_mIoU@rho needs
# boundary_dist) - CBL and the baseline need no such step, though running this on their
# splits too is harmless and lets evaluate_closenet_ckpts.py report boundary_mIoU@rho
# for them as a reference number.
#
# Usage: sbatch prep_boundaries.sbatch <split_file.json>
#
# Examples:
#   sbatch prep_boundaries.sbatch cfg/data_split.json           # CloSe-Di
#   sbatch prep_boundaries.sbatch cfg/data_split_thuman.json    # THuman2.0-derived split

set -eo pipefail

SPLIT="${1:?Usage: sbatch prep_boundaries.sbatch <split_file.json>}"

# SLURM sets this to the directory `sbatch` was invoked from - no hardcoded path needed.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u prep_boundaries.py --split "$SPLIT"

echo "Boundary precomputation finished for $SPLIT"
