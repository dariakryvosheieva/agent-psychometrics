#!/bin/bash
#SBATCH --job-name=eval_embed
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_embed_%j.out
#SBATCH --error=logs/eval_embed_%j.err

# Evaluate pre-computed trajectory embeddings
# This script runs the training/evaluation pipeline (CPU-only, fast)
#
# Usage:
#   # Single experiment
#   sbatch --export=EMBEDDINGS_DIR="chris_output/experiment_b/trajectory_embeddings/full_difficulty" \
#       evaluate_embeddings_engaging.sh
#
#   # Run all ablations
#   sbatch --export=RUN_ABLATIONS=1 evaluate_embeddings_engaging.sh

set -e

module load miniforge
conda activate irt

# Configuration
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-chris_output/experiment_b/trajectory_embeddings/full_difficulty}"
AGGREGATION="${AGGREGATION:-mean_std}"
ALPHA="${ALPHA:-cv}"
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"

OUTPUT_DIR="chris_output/experiment_b/embedding_results"

echo "=============================================="
echo "Embedding Evaluation"
echo "=============================================="

if [[ "$RUN_ABLATIONS" == "1" ]]; then
    echo "Running full ablation study..."
    python -m experiment_b.train_evaluate_embeddings \
        --ablations \
        --embeddings_base_dir "chris_output/experiment_b/trajectory_embeddings" \
        --output_dir "$OUTPUT_DIR"
else
    echo "Embeddings dir: $EMBEDDINGS_DIR"
    echo "Aggregation: $AGGREGATION"
    echo "Alpha: $ALPHA"

    python -m experiment_b.train_evaluate_embeddings \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --aggregation "$AGGREGATION" \
        --alpha "$ALPHA" \
        --output_dir "$OUTPUT_DIR"
fi

echo "Done!"
