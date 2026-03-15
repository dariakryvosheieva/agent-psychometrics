#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

python -u -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
  --split_by benchmark \
  --train_benchmarks verified,terminalbench,pro \
  --ood_benchmark gso \
  --out_dir data/held_out_benchmark \
  --method combined
