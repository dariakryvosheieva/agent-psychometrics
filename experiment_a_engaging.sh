#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH --job-name=exp_a_auc
#SBATCH --output=logs/exp_a_auc_%j.out
#SBATCH --error=logs/exp_a_auc_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Experiment A: Prior Validation (IRT AUC)
# Evaluates how well Daria's embeddings predict agent success
# Run from: ~/model_irt
# Submit with: sbatch experiment_a_engaging.sh

set -euo pipefail

cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

# Define paths
EMBEDDINGS_PATH="out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
OUTPUT_DIR="chris_output/experiment_a"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V

echo ""
echo "=== Input Files ==="
echo "Embeddings: ${EMBEDDINGS_PATH}"
ls -lh "${EMBEDDINGS_PATH}"

echo ""
echo "=== Running Experiment A ==="
python -m experiment_a.train_evaluate \
  --embeddings_path "${EMBEDDINGS_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --test_fraction 0.2 \
  --split_seed 0 \
  --ridge_alpha 10000.0

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Results: ${OUTPUT_DIR}/experiment_a_results.json"
cat "${OUTPUT_DIR}/experiment_a_results.json" | python -m json.tool | head -50
