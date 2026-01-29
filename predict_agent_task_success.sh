#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=node4100,node3401

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

python -u predict_agent_task_success.py \
  --train_benchmarks verified \
  --verified_agent_results_jsonl out/chris_irt/swebench_verified_20251115_full.jsonl \
  --out_dir out/agent_task_success_verified \
  --split_by task \
  --embeddings_cache out/agent_task_success_verified/task_embeddings__qs_sol_instr_b7008f2d.npz \
  --pca