#!/bin/bash
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mem=100G
#SBATCH --partition=mit_normal

set -euo pipefail

cd /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship

source /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/activate

which python
python -V
python -c "import sys; print(sys.executable)"

export PYTHONHASHSEED=0

# -------- Config --------

# Verified: regenerate JSONL from experiments (mirrors job_irt_model_scaffold.sh)
EXPERIMENTS_DIR_VERIFIED="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/experiments/evaluation/verified"
VERIFIED_JSONL="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl"
AGENTS_MD="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/verified_agents.md"

# Pro: regenerate JSONL from the agent-runs CSV (model_name x instance_id -> resolved)
PRO_CSV="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/agent-runs-032fb63d-4992-4bfc-911d-3b7dafcb931f.csv"
PRO_JSONL="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_pro.jsonl"

# Terminal-Bench 2.0: prepared JSONL (subject_id, responses)
TERMINAL_BENCH_JSONL="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/terminal_bench_2.0.jsonl"

# Train options
IRT_MODEL="1pl"     # keep it Rasch-style (no discrimination)
EPOCHS=5000
SEED=0
LR=0.01

# Outputs
OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_model_scaffold_shared"
SPLIT_CSV="$OUT_DIR/agent_model_scaffold.csv"
UNSPLITTABLE_TXT="$OUT_DIR/agent_unsplittable.txt"

# -------- Run --------

python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/prep_swebench.py \
  --experiments_dir "$EXPERIMENTS_DIR_VERIFIED" \
  --output_path "$VERIFIED_JSONL"

python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/prep_swebench_pro.py \
  --csv_path "$PRO_CSV" \
  --output_path "$PRO_JSONL"

python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/split_agents_model_scaffold.py \
  --results_jsonl "$VERIFIED_JSONL" \
  --pro_results_jsonl "$PRO_JSONL" \
  --terminal_bench_results_jsonl "$TERMINAL_BENCH_JSONL" \
  --agents_md "$AGENTS_MD" \
  --output_csv "$SPLIT_CSV" \
  --unsplittable_txt "$UNSPLITTABLE_TXT"

python /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/train_model_scaffold_shared.py \
  --verified_path "$VERIFIED_JSONL" \
  --pro_path "$PRO_JSONL" \
  --terminal_bench_path "$TERMINAL_BENCH_JSONL" \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --model "$IRT_MODEL" \
  --seed "$SEED" \
  --lr "$LR"

