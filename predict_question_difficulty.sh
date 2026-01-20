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

# Usage:
#   sbatch predict_question_difficulty.sh
#   sbatch predict_question_difficulty.sh train_irt_model_scaffold_shared
#
# Optional (when predicting):
#   ENABLE_IRT_MODEL_SCAFFOLD_SHARED_TRAINING=1 sbatch predict_question_difficulty.sh
#
MODE="${1:-predict}"

# Keep HF caches on scratch.
export HF_HOME="/orcd/scratch/orcd/001/daria_k/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

SEED=0

# ---------------------------------------------------------------------------
# Optional: run the model/scaffold-shared IRT training pipeline (CPU-friendly,
# but safe to run here as well).
# Mirrors `job_irt_model_scaffold_shared.sh`.
# ---------------------------------------------------------------------------

run_irt_model_scaffold_shared() {
  export PYTHONHASHSEED="$SEED"

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
  LR=0.01

  # Filter: keep only scaffolds that occur with >= K distinct base models (per JSONL).
  MIN_DISTINCT_MODELS_PER_SCAFFOLD=2

  OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_model_scaffold_shared__min${MIN_DISTINCT_MODELS_PER_SCAFFOLD}models_per_scaffold"

  # Mapping outputs (both full + filtered, for convenience)
  SPLIT_CSV_ALL="$OUT_DIR/agent_model_scaffold__all.csv"
  UNSPLITTABLE_TXT_ALL="$OUT_DIR/agent_unsplittable__all.txt"
  SPLIT_CSV_FILTERED="$OUT_DIR/agent_model_scaffold.csv"
  UNSPLITTABLE_TXT_FILTERED="$OUT_DIR/agent_unsplittable__filtered_min_models${MIN_DISTINCT_MODELS_PER_SCAFFOLD}.txt"

  # Filtered JSONLs used for training (and downstream prediction job)
  VERIFIED_JSONL_FILTERED="$OUT_DIR/swebench_verified_20251115_full__filtered_min_models${MIN_DISTINCT_MODELS_PER_SCAFFOLD}.jsonl"
  TERMINAL_BENCH_JSONL_FILTERED="$OUT_DIR/terminal_bench_2.0__filtered_min_models${MIN_DISTINCT_MODELS_PER_SCAFFOLD}.jsonl"

  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/prep_swebench.py \
    --experiments_dir "$EXPERIMENTS_DIR_VERIFIED" \
    --output_path "$VERIFIED_JSONL"

  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/prep_swebench_pro.py \
    --csv_path "$PRO_CSV" \
    --output_path "$PRO_JSONL"

  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/split_agents_model_scaffold.py \
    --results_jsonl "$VERIFIED_JSONL" \
    --pro_results_jsonl "$PRO_JSONL" \
    --terminal_bench_results_jsonl "$TERMINAL_BENCH_JSONL" \
    --agents_md "$AGENTS_MD" \
    --output_csv "$SPLIT_CSV_ALL" \
    --unsplittable_txt "$UNSPLITTABLE_TXT_ALL"

  # Filter subject-responses JSONLs to remove scaffolds that only occur with one base model (per file).
  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/filter_subjects_by_scaffold_count.py \
    --input_jsonl "$VERIFIED_JSONL" \
    --output_jsonl "$VERIFIED_JSONL_FILTERED" \
    --min_distinct_models_per_scaffold "$MIN_DISTINCT_MODELS_PER_SCAFFOLD"

  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/filter_subjects_by_scaffold_count.py \
    --input_jsonl "$TERMINAL_BENCH_JSONL" \
    --output_jsonl "$TERMINAL_BENCH_JSONL_FILTERED" \
    --min_distinct_models_per_scaffold "$MIN_DISTINCT_MODELS_PER_SCAFFOLD"

  # Write mapping CSV for the filtered subject sets (to mirror the training inputs).
  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/split_agents_model_scaffold.py \
    --results_jsonl "$VERIFIED_JSONL_FILTERED" \
    --pro_results_jsonl "$PRO_JSONL" \
    --terminal_bench_results_jsonl "$TERMINAL_BENCH_JSONL_FILTERED" \
    --agents_md "$AGENTS_MD" \
    --output_csv "$SPLIT_CSV_FILTERED" \
    --unsplittable_txt "$UNSPLITTABLE_TXT_FILTERED"

  "${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/swebench_irt/train_model_scaffold_shared.py \
    --verified_path "$VERIFIED_JSONL_FILTERED" \
    --pro_path "$PRO_JSONL" \
    --terminal_bench_path "$TERMINAL_BENCH_JSONL_FILTERED" \
    --output_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --model "$IRT_MODEL" \
    --seed "$SEED" \
    --lr "$LR"

  # Export for callers (best-effort; bash scoping is per-process)
  echo "MODEL_SCAFFOLD_SHARED_OUT_DIR=$OUT_DIR"
  echo "MODEL_SCAFFOLD_SHARED_VERIFIED_JSONL_FILTERED=$VERIFIED_JSONL_FILTERED"
}

if [[ "$MODE" == "train_irt_model_scaffold_shared" ]]; then
  run_irt_model_scaffold_shared
  exit 0
fi

AGENT_RESULTS_OVERRIDE=()
if [[ "${ENABLE_IRT_MODEL_SCAFFOLD_SHARED_TRAINING:-0}" == "1" ]]; then
  # Run the same training pipeline before prediction, and point prediction at the filtered Verified JSONL.
  # (This keeps the IRT supervision in `predict_question_difficulty_irt.py` aligned with the shared-scaffold filtering.)
  MIN_DISTINCT_MODELS_PER_SCAFFOLD=2
  OUT_DIR="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_model_scaffold_shared__min${MIN_DISTINCT_MODELS_PER_SCAFFOLD}models_per_scaffold"
  VERIFIED_JSONL_FILTERED="$OUT_DIR/swebench_verified_20251115_full__filtered_min_models${MIN_DISTINCT_MODELS_PER_SCAFFOLD}.jsonl"

  run_irt_model_scaffold_shared
  AGENT_RESULTS_OVERRIDE=(--agent_results "$VERIFIED_JSONL_FILTERED")

  # Feed precomputed IRT scores into difficulty prediction (skip per-fold IRT).
  # NOTE: `train_model_scaffold_shared.py` writes outputs to historical subdir names.
  # Keep this mapping in sync with `swebench_irt/train_model_scaffold_shared.py`.
  case "${IRT_MODEL}" in
    1pl)    IRT_SUBDIR="1d_1pl" ;;
    2pl)    IRT_SUBDIR="1d_2pl" ;;
    2d_1pl) IRT_SUBDIR="2d_1pl" ;;
    *)
      echo "Unknown IRT_MODEL=${IRT_MODEL}. Expected one of: 1pl, 2pl, 2d_1pl" >&2
      exit 2
      ;;
  esac
  IRT_ITEMS_CSV="$OUT_DIR/${IRT_SUBDIR}/items_verified.csv"
  IRT_AGENT_MAP_CSV="$OUT_DIR/agent_model_scaffold.csv"
  IRT_MODEL_THETAS_CSV="$OUT_DIR/${IRT_SUBDIR}/model_abilities.csv"
  IRT_SCAFFOLD_THETAS_CSV="$OUT_DIR/${IRT_SUBDIR}/scaffold_abilities.csv"

  PRECOMPUTED_IRT_ARGS=(
    --irt_items_csv "$IRT_ITEMS_CSV"
    --irt_agent_map_csv "$IRT_AGENT_MAP_CSV"
    --irt_model_thetas_csv "$IRT_MODEL_THETAS_CSV"
    --irt_scaffold_thetas_csv "$IRT_SCAFFOLD_THETAS_CSV"
  )
fi

"${VENV_PY}" /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty_irt.py \
  --shuffle \
  --exclude_zero_success \
  --trust_remote_code \
  --overwrite \
  --seed "$SEED" \
  --backbone "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  "${AGENT_RESULTS_OVERRIDE[@]}" \
  ${PRECOMPUTED_IRT_ARGS:+${PRECOMPUTED_IRT_ARGS[@]}}


