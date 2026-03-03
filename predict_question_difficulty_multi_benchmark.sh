#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=node4100,node3401

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

python -u predict_question_difficulty_multi_benchmark.py \
  --trust_remote_code \
  --train_benchmarks verified,pro,terminal_bench,gso \
  --out_dir out/all_benchmarks \
  --method combined \
  --split_by task \
  --include_zero_success \
  --embeddings_cache out/all_benchmarks/embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__1112142d9a86__maxlen8192.npz