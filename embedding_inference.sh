#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship

VENV_PY="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/python"

which python
python -V
python -c "import sys; print(sys.executable)"

echo "Using venv python: ${VENV_PY}"
"${VENV_PY}" -V
"${VENV_PY}" -c "import sys; print(sys.executable)"

# Keep HF caches on scratch.
export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

"${VENV_PY}" embedding_inference.py \
             --weights_dir out/swebench_pro \
             --items_csv out/swebench_verified/predictions.csv \
             --items_splits test \
             --dataset_name princeton-nlp/SWE-bench_Verified \
             --hf_split test \
             --out_dir out/swebench_pro_to_verified