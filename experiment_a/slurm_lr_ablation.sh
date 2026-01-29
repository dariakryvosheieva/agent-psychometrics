#!/bin/bash
#SBATCH --job-name=lr_ablation
#SBATCH --output=lr_ablation_%j.out
#SBATCH --error=lr_ablation_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu

# Learning Rate Ablation Study
# Tests different agent_lr_scale values to see if slowing agent learning
# can solve gradient competition without freezing abilities.
# Run with: sbatch experiment_a/slurm_lr_ablation.sh

set -e

echo "=========================================="
echo "Learning Rate Ablation Study"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

# Setup environment
cd ~/model_irt
source .venv/bin/activate
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run LR ablation study
echo ""
echo "Running LR ablation study on LLM Judge..."
python -m experiment_a.test_lr_ablation

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
