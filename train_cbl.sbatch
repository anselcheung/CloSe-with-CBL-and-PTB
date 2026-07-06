#!/bin/bash
#SBATCH --job-name=CloSeDi_cbl
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH -o logs/train_cbl_%j.out
#SBATCH -e logs/train_cbl_%j.err

# Trains CloSeNet on CloSe-Di WITH the contrastive boundary loss (cbl.enabled: true
# in cfg/closenet_cbl.yaml). Same data split and architecture as train_base.sbatch,
# so the only difference between the two runs is the CBL loss term. Checkpoints land
# in closenet_train_cbl/checkpoints/.

set -eo pipefail

cd /home/stud102/CloSe-with-CBL-and-PTB
mkdir -p logs pretrained

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u train_closenet.py cfg/closenet_cbl.yaml

# Copy the checkpoint with the highest validation mIoU to pretrained/ for easy access.
python - <<'PY'
import re
import shutil
from pathlib import Path

ckpt_dir = Path('closenet_train_cbl/checkpoints')
ckpts = [p for p in ckpt_dir.glob('valmin_*.pt') if re.search(r'mIoU=([0-9.]+)', p.name)]
if not ckpts:
    raise SystemExit(f'No scored checkpoints found in {ckpt_dir}')

best = max(ckpts, key=lambda p: float(re.search(r'mIoU=([0-9.]+)', p.name).group(1)))
target = Path('pretrained/closenet_cbl.pt')
shutil.copy2(best, target)
print(f'Best CBL checkpoint: {best} (copied to {target})')
PY

echo 'CBL training finished'
