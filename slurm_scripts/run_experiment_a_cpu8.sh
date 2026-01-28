#!/bin/bash
#SBATCH --job-name=exp_a_cpu8
#SBATCH --output=logs/exp_a_cpu8_%j.out
#SBATCH --error=logs/exp_a_cpu8_%j.err
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Experiment A: Run on all datasets with dataset-level parallelization only
#
# Simple strategy: 4 datasets run in parallel via ProcessPoolExecutor.
# Methods and folds run sequentially within each dataset.
# This avoids parallelization overhead which was making things slower.

set -e

echo "Starting Experiment A on $(hostname)"
echo "Date: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Load conda environment
module load miniforge
conda activate irt

# Set HuggingFace cache to scratch (avoid home quota limits)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Set OMP threads to avoid over-subscription with joblib
export OMP_NUM_THREADS=1

# Change to project directory
cd ~/model_irt

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="chris_output/experiment_a/results_${TIMESTAMP}.csv"
OUTPUT_DIR="/tmp/experiment_a_${SLURM_JOB_ID}"

echo "Output CSV: $OUTPUT_CSV"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run all datasets in parallel (4 datasets via ProcessPoolExecutor)
# Methods and folds run sequentially within each dataset
python -m experiment_a.run_all_datasets \
    --output "$OUTPUT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers=4

echo ""
echo "Experiment A completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"