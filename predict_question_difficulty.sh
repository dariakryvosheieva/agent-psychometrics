#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

python -u predict_question_difficulty.py \
  --trust_remote_code \
  --benchmark terminalbench \
  --agent_results data/terminalbench/responses_terminus2.jsonl \
  --seed 1 \
  --method judge \
  --overwrite