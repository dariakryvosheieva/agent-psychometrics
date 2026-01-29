#!/bin/bash
#SBATCH --job-name=mlp_ablation
#SBATCH --output=mlp_ablation_%j.out
#SBATCH --error=mlp_ablation_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu

# MLP Ablation Study for Experiment A
# Tests 4 conditions: baseline, frozen_irt, strong_reg, both_fixes
# Run with: sbatch experiment_a/slurm_mlp_ablation.sh

set -e

echo "=========================================="
echo "MLP Ablation Study"
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

# Run ablation study
echo ""
echo "Running ablation study on all sources (embedding, llm_judge, grouped)..."
python -m experiment_a.run_mlp_ablation --source all --k_folds 5

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
