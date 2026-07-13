#!/bin/bash
#SBATCH --job-name=CloSeDi_prep_thuman
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o logs/prep_thuman_%j.out
#SBATCH -e logs/prep_thuman_%j.err

# Preprocesses THuman2.0 into the CloSe-Di npz schema (prep_thuman.py --stage all),
# then folds the results into cfg/data_split_thuman.json (cfg/data_split.json is left
# untouched). Skipped/failed scans are logged to
# data/THuman2.0_preprocessed/prep_thuman_log.txt. Needed once before training/evaluating
# any config that points data.split_file at cfg/data_split_thuman.json.

set -eo pipefail

# SLURM sets this to the directory `sbatch` was invoked from - no hardcoded path needed.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Fill in: path to the SMPL (not SMPL-X) body model directory, i.e. the parent of a
# "smpl/" folder containing SMPL_NEUTRAL.pkl etc.
BM_DIR_PATH=./models

# --- Debugging / trial run ---
# Before submitting the full job, it's worth running a small trial to confirm the
# environment, paths, and SMPL model are all set up correctly, and to gauge how long a
# single scan takes (to size --time for the full run). Uncomment ONE of the lines below
# (and comment out the full run below it):
#
# 1) Single-scan sanity check:
# python prep_thuman.py --bm_dir_path "$BM_DIR_PATH" --scan_ids 0000 --stage prep --overwrite
#
# 2) Small batch trial (first 20 scans found in data/THuman2.0_Release_copy/):
# python prep_thuman.py --bm_dir_path "$BM_DIR_PATH" --limit 20 --stage split

python -u prep_thuman.py --bm_dir_path "$BM_DIR_PATH" --stage all

echo 'THuman2.0 preprocessing finished'
