#!/bin/bash
#SBATCH --job-name=init_noise
#SBATCH --output=logs/init_noise_%j.out
#SBATCH --error=logs/init_noise_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=mit_normal_gpu

# Test whether adding noise to IRT initialization breaks symmetry
# and improves learning.
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_test_init_noise.sh

set -e

echo "=========================================="
echo "Init Noise Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo ""
echo "Running init noise test..."
python -m experiment_a.mlp_ablation.test_init_noise

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
