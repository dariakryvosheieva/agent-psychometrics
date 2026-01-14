#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH --job-name=eval_embed
#SBATCH --output=logs/eval_embed_%j.out
#SBATCH --error=logs/eval_embed_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluate trajectory embeddings for Experiment B posterior difficulty prediction
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/evaluate_embeddings.sh

set -euo pipefail

cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

EMBEDDINGS_DIR="chris_output/experiment_b/trajectory_embeddings/full_difficulty"

mkdir -p logs

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Python: $(which python)"

echo ""
echo "=== Embeddings ==="
echo "Dir: ${EMBEDDINGS_DIR}"
total=$(find "${EMBEDDINGS_DIR}" -name "*.npz" | wc -l)
echo "Total embeddings: ${total}"

echo ""
echo "=== Running Evaluation ==="
python -m experiment_b.train_evaluate_embeddings \
    --embeddings_dir "${EMBEDDINGS_DIR}"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
