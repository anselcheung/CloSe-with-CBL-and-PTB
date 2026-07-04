#!/bin/bash
#SBATCH --job-name=close_eval
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

# Activate your environment (see note in tcml_job_train.sh about conda in SLURM).
eval "$(conda shell.bash hook)"
conda activate close

# ============================================================================
# IMPORTANT ARGUMENTS
#   CONFIG     — test config; selects the eval mode (aux metrics / post-proc).
#   CHECKPOINT — model weights: the shipped pretrained/closenet.pth, OR a .pt
#                produced by training (its embedded config restores the aux
#                heads automatically, so a PTB checkpoint "just works" here).
# Uncomment exactly ONE (CONFIG, CHECKPOINT) pair below.
# ============================================================================

# (1) Baseline CloSeNet — no aux heads, no post-processing:
CONFIG="cfg/closenet_test_baseline.yaml";      CHECKPOINT="pretrained/closenet.pth"

# (2) CloSeNet + PTB — aux heads active, reports boundary_mIoU@rho:
# CONFIG="cfg/closenet_test_ptb.yaml";           CHECKPOINT="<PATH_TO_PTB_CHECKPOINT>.pt"

# (3) CloSeNet + PTB + SegFix post-processing (needs the SAME PTB checkpoint):
# CONFIG="cfg/closenet_test_ptb_postproc.yaml";  CHECKPOINT="<PATH_TO_PTB_CHECKPOINT>.pt"

echo "Eval config:     $CONFIG"
echo "Eval checkpoint: $CHECKPOINT"

# ----------------------------------------------------------------------------
# NOTES
# * boundary_mIoU@rho is reported whenever the eval scans carry boundary_dist
#   (i.e. prep_boundaries.py has been run over them) — including for the
#   baseline checkpoint, giving a reference number to compare the PTB modes to.
#   If boundary metrics are missing, run:  python prep_boundaries.py --split cfg/data_split.json
# * exp_logs_path in each test yaml points at a FRESH dir so the trainer does
#   not auto-load a stray checkpoint over the weights you pass in. Keep it empty.
# * Post-processing knobs live under `post_processing:` in the yaml
#   (threshold, step=null -> auto per-cloud spacing, n_iters).
#   Enabling post-processing on a NON-PTB checkpoint raises a clear error.
# ----------------------------------------------------------------------------

# ============================================================================
# EXAMPLE CALLS (equivalent to the pairs above) — the active command at the
# bottom uses $CONFIG / $CHECKPOINT. Usage: python test_closenet.py <cfg> <ckpt>
# ============================================================================
# python test_closenet.py cfg/closenet_test_baseline.yaml     pretrained/closenet.pth
# python test_closenet.py cfg/closenet_test_ptb.yaml          closenet_ptb_train/checkpoints/<BEST>.pt
# python test_closenet.py cfg/closenet_test_ptb_postproc.yaml closenet_ptb_train/checkpoints/<BEST>.pt
# ============================================================================

# --- Active command ---
python test_closenet.py "$CONFIG" "$CHECKPOINT"

echo "End: $(date)"
