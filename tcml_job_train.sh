#!/bin/bash
#SBATCH --job-name=close_ptb_train
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt

# --- Resources ---
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=day
# Full 200-epoch runs can exceed the 'day' partition limit; bump --time and/or
# switch --partition (e.g. week) if a run gets killed for hitting the wall clock.
#SBATCH --time=23:00:00

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
# Note: sourcing ~/.bashrc + `conda activate` alone can print
# "run conda init before conda activate" in non-interactive SLURM shells,
# because .bashrc's early-return-if-non-interactive guard skips past the
# conda init block. Calling the shell hook directly avoids that.
eval "$(conda shell.bash hook)"
conda activate close

# ============================================================================
# IMPORTANT ARGUMENT — the training config selects the ablation/mode.
# Uncomment exactly ONE assignment below (or point CONFIG at your own copy).
# ============================================================================
# CONFIG="cfg/closenet.yaml"              # (1) baseline CloSeNet (segm loss only)
# CONFIG="cfg/closenet_ptb.yaml"        # (2) CloSeNet + PTB aux heads (+boundary +direction)
CONFIG="cfg/closenet_ptb_cbl.yaml"

echo "Training config: $CONFIG"

# ----------------------------------------------------------------------------
# PREREQUISITE (PTB modes only): boundary/direction/boundary_dist GT must exist
# in the scan npz files before training with cfg/closenet_ptb.yaml. Run ONCE
# (cheap, CPU-only) over every scan referenced by the split, then comment out.
# ----------------------------------------------------------------------------
# python prep_boundaries.py --split cfg/data_split.json
# python prep_boundaries.py --split cfg/data_split.json --overwrite   # recompute
# python prep_boundaries.py --data_dir data/THuman2.0_preprocessed    # or a dir

# ============================================================================
# EXAMPLE ABLATION RUNS — uncomment ONE. The default active command at the
# bottom uses $CONFIG above; the blocks here are ready-to-go alternatives.
# ============================================================================
#
# (1) Baseline:
# python train_closenet.py cfg/closenet.yaml
#
# (2) CloSeNet + PTB (both heads, PTB loss weights from the yaml):
# python train_closenet.py cfg/closenet_ptb.yaml
#
# (3) Ablation grid (+B only / +D only): copy cfg/closenet_ptb.yaml and, in the
#     copy, zero the loss weight of the head you want to disable (the head still
#     exists but does not shape the encoder):
#       +B only  -> training.loss_weights.direction_loss: 0.0
#       +D only  -> training.loss_weights.boundary_loss:  0.0
# python train_closenet.py cfg/closenet_ptb_bonly.yaml
# python train_closenet.py cfg/closenet_ptb_donly.yaml
#
# Other knobs worth sweeping live in the yaml:
#   training.loss_weights.{boundary_loss,direction_loss}   (defaults 3.0 / 0.3)
#   training.boundary_beta                                 (default 0.6)
#   training.dir_mask_dist                                 (default null)
#   data.pointcloud_samples                                (default 2048)
#   exp_logs_path      -> set a UNIQUE dir per run so checkpoints don't collide.
# ============================================================================

# --- Active command ---
python train_closenet.py "$CONFIG"

echo "End: $(date)"
