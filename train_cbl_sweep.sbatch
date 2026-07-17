#!/bin/bash
#SBATCH --job-name=CloSeDi_cbl_sweep
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o logs/train_cbl_sweep_%j.out
#SBATCH -e logs/train_cbl_sweep_%j.err

# Trains CloSeNet with the CBL loss using whichever config is passed as $1
# (e.g. cfg/closenet_cbl_w001.yaml). Meant for sweeping training.loss_weights.cbl_loss
# without duplicating train_cbl.sbatch per variant. exp_logs_path/checkpoints dir comes
# from the config itself, so each variant writes to its own directory.
#
# Usage: sbatch train_cbl_sweep.sbatch cfg/closenet_cbl_w001.yaml

set -eo pipefail

CFG="${1:?Usage: sbatch train_cbl_sweep.sbatch <config.yaml>}"

# SLURM sets this to the directory `sbatch` was invoked from - no hardcoded path needed.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs pretrained

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u train_closenet.py "$CFG"

# Copy the checkpoint with the highest validation mIoU to pretrained/<variant>.pt.
CFG="$CFG" python - <<'PY'
import os
import re
import shutil
import yaml
from pathlib import Path

cfg_path = Path(os.environ['CFG'])
variant = cfg_path.stem
cfg = yaml.load(cfg_path.read_text(), Loader=yaml.FullLoader)

ckpt_dir = Path(cfg['exp_logs_path']) / 'checkpoints'
ckpts = [p for p in ckpt_dir.glob('valmin_*.pt') if re.search(r'mIoU=([0-9.]+)', p.name)]
if not ckpts:
    raise SystemExit(f'No scored checkpoints found in {ckpt_dir}')

best = max(ckpts, key=lambda p: float(re.search(r'mIoU=([0-9.]+)', p.name).group(1)))
target = Path('pretrained') / f'{variant}.pt'
shutil.copy2(best, target)
print(f'Best checkpoint for {variant}: {best} (copied to {target})')
PY

echo "CBL sweep training finished for $CFG"
