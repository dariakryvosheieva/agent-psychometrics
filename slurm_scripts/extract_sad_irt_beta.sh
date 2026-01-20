#!/bin/bash
#SBATCH --job-name=extract_beta
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/extract_beta_%j.out
#SBATCH --error=logs/extract_beta_%j.err

# Extract beta values from SAD-IRT checkpoints
# This script extracts learned difficulty parameters from checkpoints
# and saves them as CSV files for easy transfer and evaluation.
#
# Output: chris_output/sad_irt_beta_values/*.csv
#
# Usage:
#   sbatch slurm_scripts/extract_sad_irt_beta.sh
#   sbatch slurm_scripts/extract_sad_irt_beta.sh --checkpoint_dir chris_output/sad_irt_long

set -e

# Create log directory
mkdir -p logs

echo "=========================================="
echo "SAD-IRT Beta Extraction"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load conda
module load miniforge
conda activate irt

# Set HuggingFace cache to scratch (avoid home quota)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Default checkpoint directory
CHECKPOINT_DIR="${1:-chris_output/sad_irt_long}"
OUTPUT_DIR="chris_output/sad_irt_beta_values"

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run extraction (using standalone script to avoid peft/transformers imports)
python scripts/extract_sad_irt_beta.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To copy results locally, run:"
echo "  scp -r $(whoami)@engaging-submit.mit.edu:~/model_irt/$OUTPUT_DIR chris_output/"
