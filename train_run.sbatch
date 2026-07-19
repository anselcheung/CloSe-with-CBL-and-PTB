#!/bin/bash
#SBATCH --job-name=CloSeDi_train_run
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o logs/train_run_%j.out
#SBATCH -e logs/train_run_%j.err

# Generic seeded, timestamped training launcher. Unlike train_base.sbatch/train_cbl.sbatch/
# train_cbl_sweep.sbatch (which all write into a single fixed exp_logs_path baked into the
# config file, so re-running the same config resumes/overwrites the previous run), this
# script gives every submission its own runs/<experiment_name>/<date_time>/ directory:
#   runs/<experiment_name>/<date_time>/
#     config.yaml      # exact resolved config used for this run (exp_logs_path + seed baked in)
#     checkpoints/      # written by CheckpointIO, same as always
#     summary/           # tensorboardX event files
#     debug.log            # Logger output
#     logs/                  # symlinks to this job's SLURM stdout/stderr
#
# It does this without touching train_closenet.py's training logic: it just generates a
# copy of the given config with exp_logs_path (and, if given, seed) overridden, and hands
# that copy to train_closenet.py unchanged. Seeding itself is the one piece that does live
# in Python - train_closenet.py calls fix_seeds(cfg.get('seed', 42)) before building the
# dataset/model, so the same config + seed reproduces the same run.
#
# Usage: sbatch train_run.sbatch <config.yaml> <experiment_name> [seed]
# Examples:
#   sbatch train_run.sbatch cfg/closenet_ptb.yaml ptb
#   sbatch train_run.sbatch cfg/closenet_ptb.yaml ptb_seed1 1
#   sbatch train_run.sbatch cfg/closenet_ptb_cbl.yaml ptb_cbl_seed2 2

set -eo pipefail

CONFIG="${1:?Usage: sbatch train_run.sbatch <config.yaml> <experiment_name> [seed]}"
EXP_NAME="${2:?Usage: sbatch train_run.sbatch <config.yaml> <experiment_name> [seed]}"
SEED="${3:-}"

# SLURM sets this to the directory `sbatch` was invoked from - no hardcoded path needed.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${EXP_NAME}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

RUN_CONFIG="$RUN_DIR/config.yaml"

# Resolve the run's config: same content as $CONFIG, but exp_logs_path repointed at
# RUN_DIR and seed overridden if one was passed on the command line. Saved into RUN_DIR
# so it travels with the checkpoints/logs it produced.
CONFIG="$CONFIG" RUN_DIR="$RUN_DIR" RUN_CONFIG="$RUN_CONFIG" SEED="$SEED" python - <<'PY'
import os

import yaml

cfg_path = os.environ['CONFIG']
run_dir = os.environ['RUN_DIR']
run_config_path = os.environ['RUN_CONFIG']
seed = os.environ.get('SEED')

with open(cfg_path) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg['exp_logs_path'] = f'./{run_dir}'
if seed:
    cfg['seed'] = int(seed)

with open(run_config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f'Base config:    {cfg_path}')
print(f'Run config:     {run_config_path}')
print(f'Run directory:  {run_dir}')
print(f"Seed:           {cfg.get('seed', '<none>')}")
PY

python -u train_closenet.py "$RUN_CONFIG"

# Symlink (not copy) this job's SLURM stdout/stderr into the run dir, so the full log -
# including anything written after this point - always lives alongside the checkpoints
# and config, without duplicating the file.
mkdir -p "$RUN_DIR/logs"
ln -sf "$(pwd)/logs/train_run_${SLURM_JOB_ID}.out" "$RUN_DIR/logs/slurm_${SLURM_JOB_ID}.out"
ln -sf "$(pwd)/logs/train_run_${SLURM_JOB_ID}.err" "$RUN_DIR/logs/slurm_${SLURM_JOB_ID}.err"

echo "Training finished: $RUN_DIR"
