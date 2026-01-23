#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1

set -euo pipefail
cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship
source .venv/bin/activate

export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

python predict_question_difficulty_combined_features.py \
       --trust_remote_code \
       --overwrite \
       --dataset_name "princeton-nlp/SWE-bench_Verified" \
       --dataset_path "" \
       --agent_results out/chris_irt/swebench_verified_20251115_full.jsonl \
       --judge_features_dir llm_judge/features/verified \
       --out_dir out/swebench_verified_combined_features \
       --debug