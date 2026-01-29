#!/bin/bash
#SBATCH --job-name=arch_sweep
#SBATCH --output=logs/arch_sweep_%A_%a.out
#SBATCH --error=logs/arch_sweep_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=4

# Architecture Sweep (Parallel Execution)
# Part 1-2: DeepMLP, SwiGLU, and AUC-based early stopping (COMPLETED)
# Part 3: Small DeepMLP architectures (COMPLETED - best: 64-64)
# Part 4: Combined features (embeddings + judge) - single-path vs dual-path
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_architecture_sweep.sh

set -e

echo "=========================================="
echo "Architecture Sweep - Part ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

echo ""
echo "Running architecture sweep Part ${SLURM_ARRAY_TASK_ID}..."
python -m experiment_a.mlp_ablation.architecture_sweep --part ${SLURM_ARRAY_TASK_ID}

echo ""
echo "=========================================="
echo "Part ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "=========================================="
