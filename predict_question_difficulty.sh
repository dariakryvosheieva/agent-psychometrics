#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal
#SBATCH --exclude=node1602

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

python predict_question_difficulty.py \
  --trust_remote_code \
  --benchmark gso \
  --out_dir "out/llm_judge_ablation/gso/results/claude_opus_4_5" \
  --judge_features_dir "out/llm_judge_ablation/gso/features/claude_opus_4_5" \
  --include_zero_success \
  --method judge