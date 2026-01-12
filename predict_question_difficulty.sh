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

DIFFS="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/irt_verified_1pl_e500_seed0/question_difficulties.csv"
OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/verified_qs_sol_instr_qwen25coder14b_lr"
mkdir -p "${OUT_DIR}"

"${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty_qs_solution_instruction.py \
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


