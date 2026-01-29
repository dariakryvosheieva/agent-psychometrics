#!/bin/bash
#SBATCH --job-name=interact_sweep
#SBATCH --output=logs/interact_sweep_%A_%a.out
#SBATCH --error=logs/interact_sweep_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=1-2

# Interaction Architecture Sweep (Part 5)
# Tests new ways to combine agent and task features:
# - Part 1: Two-Tower, Bilinear, Multiplicative (baselines too)
# - Part 2: NCF, Agent Embedding
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_interaction_sweep.sh

set -e

echo "=========================================="
echo "Interaction Architecture Sweep - Part ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo ""
echo "Running interaction architecture sweep Part ${SLURM_ARRAY_TASK_ID}..."
python -m experiment_a.mlp_ablation.interaction_sweep --part ${SLURM_ARRAY_TASK_ID}

echo ""
echo "=========================================="
echo "Part ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "=========================================="
