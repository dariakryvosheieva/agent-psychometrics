#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --job-name=irt_prior
#SBATCH --output=logs/irt_prior_%j.out
#SBATCH --error=logs/irt_prior_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# MIT Engaging cluster adaptation of predict_question_difficulty.sh
# Run from: ~/model_irt
# Submit with: sbatch predict_question_difficulty_engaging.sh

set -euo pipefail

# Project in home directory
cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

# Set HuggingFace cache (avoids filling home quota)
export HF_HOME="${PWD}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"

# Define paths
DIFFS="clean_data/swebench_verified_20251115_full/1d_1pl/question_difficulties.csv"
OUT_DIR="out/prior_qwen3vl8b"
mkdir -p "${OUT_DIR}"

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Input Files ==="
echo "Difficulties: ${DIFFS}"
wc -l "${DIFFS}"

echo ""
echo "=== Starting Pipeline ==="
python predict_question_difficulty.py \
  --difficulties "${DIFFS}" \
  --dataset_name "princeton-nlp/SWE-bench_Verified" \
  --split "test" \
  --n_inputs 500 \
  --seed 0 \
  --backbone "Qwen/Qwen3-VL-8B-Instruct" \
  --max_length 8192 \
  --batch_size 1 \
  --device_map auto \
  --torch_dtype bfloat16 \
  --attn_implementation auto \
  --regressor ridge_cv \
  --test_fraction 0.2 \
  --cv_folds 5 \
  --out_dir "${OUT_DIR}" \
  --trust_remote_code \
  --embedding_layer -1

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Results: ${OUT_DIR}/predictions.csv"
echo "Metrics: ${OUT_DIR}/metrics.json"
ls -la "${OUT_DIR}/"
