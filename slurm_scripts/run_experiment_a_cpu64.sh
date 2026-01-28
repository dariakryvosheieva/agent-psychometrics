#!/bin/bash
#SBATCH --job-name=exp_a_cpu64
#SBATCH --output=logs/exp_a_cpu64_%j.out
#SBATCH --error=logs/exp_a_cpu64_%j.err
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Experiment A: Run on all datasets with two-level parallelization
#
# Parallelization strategy (64 cores):
# - 4 datasets in parallel via ProcessPoolExecutor (separate processes)
# - 16 methods in parallel per dataset via joblib (within each process)
# - Folds run sequentially (avoids nested joblib issues)
#
# ProcessPoolExecutor creates independent processes, so each dataset's
# joblib instance doesn't conflict with others. This gives us:
# 4 datasets × ~15 methods = ~60 cores at peak utilization.

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

# Run all datasets with two-level parallelization:
# - max_workers=4: 4 datasets in parallel (ProcessPoolExecutor - separate processes)
# - n_jobs_methods=16: 16 methods in parallel per dataset (joblib within each process)
# - n_jobs_folds=1: folds sequential (avoids nested joblib throttling)
python -m experiment_a.run_all_datasets \
    --output "$OUTPUT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers=4 \
    --n_jobs_methods=16 \
    --n_jobs_folds=1

echo ""
echo "Experiment A completed at $(date)"
echo "Results saved to: $OUTPUT_CSV"