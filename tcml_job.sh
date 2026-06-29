#!/bin/bash
#SBATCH --job-name=close_test
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt

# --- Resources ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=day
#SBATCH --time=04:00:00

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
source ~/.bashrc
conda activate close

# --- Your command ---
python test_closenet.py cfg/closenet_test.yaml pretrained/closenet.pth

echo "End: $(date)"