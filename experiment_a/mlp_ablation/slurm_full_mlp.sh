#!/bin/bash
#SBATCH --job-name=full_mlp
#SBATCH --output=logs/full_mlp_%j_%a.out
#SBATCH --error=logs/full_mlp_%j_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=1-2

# Full MLP Ablation Study (Parallel Execution)
# Uses SLURM array to run two parts in parallel on separate GPUs:
#   Part 1: Baselines + hidden size + weight decay ablations
#   Part 2: IRT init + early stopping + best configs
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_full_mlp.sh
# Results saved to: chris_output/experiment_a/mlp_embedding/full_mlp_results_part{1,2}.json

set -e

echo "=========================================="
echo "Full MLP Ablation - Part ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

# Setup environment
cd ~/model_irt
mkdir -p logs
source .venv/bin/activate
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run the part assigned to this array task
echo ""
echo "Running Full MLP ablation Part ${SLURM_ARRAY_TASK_ID}..."
python -m experiment_a.mlp_ablation.test_full_mlp --part ${SLURM_ARRAY_TASK_ID}

echo ""
echo "=========================================="
echo "Part ${SLURM_ARRAY_TASK_ID} finished at: $(date)"
echo "=========================================="
