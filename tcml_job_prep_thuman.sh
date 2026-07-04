#!/bin/bash
#SBATCH --job-name=close_prep_thuman
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt

# --- Resources ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --time=12:00:00

# --- GPU ---
# Standard nodes: 4x GTX 1080Ti (11GB) or 8x RTX 2080Ti (11GB)
# New nodes (Pons-Moll group has access): 8x L40S (48GB)
#SBATCH --gres=gpu:1
# If you have L40S access, uncomment and use:
# #SBATCH --gres=gpu:l40s:1

# --- Notification (optional) ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ansel-heng-yu.cheung@student.uni-tuebingen.de

# -----------------------------------------------

# Create log dir if it doesn't exist
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

# Activate your environment
# Option A: conda
# Note: sourcing ~/.bashrc + `conda activate` alone can print
# "run conda init before conda activate" in non-interactive SLURM shells,
# because .bashrc's early-return-if-non-interactive guard skips past the
# conda init block. Calling the shell hook directly avoids that.
eval "$(conda shell.bash hook)"
conda activate close

# --- Fill this in: path to the SMPL (not SMPL-X) body model directory, ---
# --- i.e. the parent of a "smpl/" folder containing SMPL_NEUTRAL.pkl etc. ---
BM_DIR_PATH=/path/to/smpl/models

# --- Your command ---
# Runs both stages: preprocess all labeled THuman2.0 scans into
# data/THuman2.0_preprocessed/, then fold them into cfg/data_split_thuman.json
# (cfg/data_split.json is left untouched). Skipped/failed scans are logged to
# data/THuman2.0_preprocessed/prep_thuman_log.txt.

# --- Debugging / trial run ---
# Before submitting the full job above, it's worth running a small trial to
# confirm the environment, paths, and SMPL model are all set up correctly,
# and to gauge how long a single scan takes (to size --time for the full run).
# Uncomment ONE of the lines below (and comment out the full run below it):
#
# 1) Single-scan sanity check (mirrors the brief's Step 1, scan 0000):
# python prep_thuman.py --bm_dir_path "$BM_DIR_PATH" --scan_ids 0000 --stage prep --overwrite
#
# 2) Small batch trial (first 20 scans found in data/THuman2.0_Release_copy/):
# python prep_thuman.py --bm_dir_path "$BM_DIR_PATH" --limit 20 --stage prep --overwrite
#
# Both write into the same data/THuman2.0_preprocessed/ output dir and log file
# as the full run, so inspect prep_thuman_log.txt afterwards, then re-run with
# --stage all (no --scan_ids/--limit) once you're satisfied it works.

python prep_thuman.py --bm_dir_path "$BM_DIR_PATH" --stage all

echo "End: $(date)"
