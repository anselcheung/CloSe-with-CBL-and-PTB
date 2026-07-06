#!/bin/bash
#SBATCH --job-name=CloSeDi_eval_variants
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/evaluate_variants_%j.out
#SBATCH -e logs/evaluate_variants_%j.err

# Evaluates an arbitrary list of cbl config variant tags (e.g. enc_w001, enc_w003,
# allstages_w001) on the CloSe-Di test split, one results JSON per variant. Generalizes
# evaluate_sweep.sbatch (which hardcodes w001/w003/w005) to any cfg/closenet_cbl_<tag>.yaml.
#
# Usage: sbatch evaluate_variants.sbatch enc_w001 enc_w003 allstages_w001 allstages_w003

set -eo pipefail

cd /home/stud102/CloSe-with-CBL-and-PTB
mkdir -p logs results

# The close env's conda activation hook (libblas_mkl_activate.sh) references
# MKL_INTERFACE_LAYER without a default, which trips `set -u` as unbound.
unset MKL_INTERFACE_LAYER

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate close
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

for tag in "$@"; do
    echo "=== evaluating $tag ==="
    python -u evaluate_closenet_ckpts.py \
        --cbl-config "cfg/closenet_cbl_${tag}.yaml" \
        --output-json "results/closenet_di_test_metrics_${tag}.json"
done

echo 'Variant evaluation finished'
