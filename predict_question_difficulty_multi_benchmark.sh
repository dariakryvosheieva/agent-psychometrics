#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=node4100

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

python predict_question_difficulty_multi_benchmark.py \
  --trust_remote_code \
  --embeddings_cache out/multi_benchmark_id/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_multibench__verified_princeton-nlp_SWE-bench_Verified_test_pro_ScaleAI_SWE-bench_Pro_test_terminal_jsonl_ter__maxlen8192.npz \
  --out_dir out/multi_benchmark_id \
  --eval_mode id \
  --include_judge