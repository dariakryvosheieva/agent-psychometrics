#!/bin/bash
#SBATCH --job-name=baseline_var
#SBATCH --output=logs/baseline_variance_%j.out
#SBATCH --error=logs/baseline_variance_%j.err
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluate baseline IRT variance across multiple random seeds
# This helps establish confidence intervals for the Spearman rho metric
#
# Note: IRT training uses Pyro SVI on CPU - no GPU needed

# Configuration
NUM_SEEDS=50
START_SEED=0
EPOCHS=2000
OUTPUT_DIR="chris_output/baseline_variance"

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_seeds) NUM_SEEDS="$2"; shift 2 ;;
        --start_seed) START_SEED="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "=============================================="
echo "Baseline IRT Variance Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  NUM_SEEDS: $NUM_SEEDS"
echo "  START_SEED: $START_SEED"
echo "  EPOCHS: $EPOCHS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "=============================================="

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Set HF cache to scratch to avoid home quota issues
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Run the variance evaluation
python -m experiment_sad_irt.baseline_variance \
    --num_seeds $NUM_SEEDS \
    --start_seed $START_SEED \
    --epochs $EPOCHS \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=============================================="
