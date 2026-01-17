#!/bin/bash
#SBATCH --job-name=tb_eval
#SBATCH --output=logs/terminalbench_evaluate_%j.out
#SBATCH --error=logs/terminalbench_evaluate_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# TerminalBench Experiment A Evaluation
# Runs full evaluation with embeddings (no GPU needed for evaluation)

set -e

echo "=== TerminalBench Experiment A Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
cd ~/model_irt
source .venv/bin/activate

# Create logs directory if needed
mkdir -p logs

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo ""

# Run evaluation with embeddings
python -m experiment_a_terminalbench.train_evaluate \
    --embeddings_path chris_output/experiment_a_terminalbench/embeddings/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__maxlen8192.npz

echo ""
echo "=== Evaluation complete ==="
echo "End time: $(date)"
