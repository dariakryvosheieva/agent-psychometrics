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

DIFFS="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_model_scaffold_shared/1d_1pl/items.csv"
OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/swebench_model_scaffold_shared"

"${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty.py \
  --difficulties "${DIFFS}" \
  --dataset_name "princeton-nlp/SWE-bench_Verified,scaleAI/SWE-bench_Pro" \
  --dataset_path "/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/terminal_bench_tasks.jsonl" \
  --split "test" \
  --n_inputs 2000 \
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
  --embedding_layer -1 \
  --overwrite \
  --agent_results /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl,/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_pro.jsonl,/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/terminal_bench_2.0.jsonl


