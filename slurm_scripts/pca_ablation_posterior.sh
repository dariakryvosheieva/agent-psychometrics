#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH --job-name=pca_ablation
#SBATCH --output=logs/pca_ablation_%j.out
#SBATCH --error=logs/pca_ablation_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# PCA Ablation for Embedding Posterior Model
# Tests different PCA dimensions and ridge alphas on trajectory embeddings
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/pca_ablation_posterior.sh

set -euo pipefail

cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

EMBEDDINGS_DIR="chris_output/experiment_b/trajectory_embeddings/full_difficulty"
OUTPUT_DIR="chris_output/experiment_b/pca_ablation"

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Python: $(which python)"

echo ""
echo "=== Embeddings ==="
echo "Dir: ${EMBEDDINGS_DIR}"
total=$(find "${EMBEDDINGS_DIR}" -name "*.npz" 2>/dev/null | wc -l)
echo "Total embeddings: ${total}"

echo ""
echo "=== Running PCA Ablation ==="
python -m experiment_b.pca_ablation \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Results: ${OUTPUT_DIR}/pca_ablation_results.json"
