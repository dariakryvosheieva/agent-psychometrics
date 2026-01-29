#!/bin/bash
#SBATCH --job-name=emb_mlp
#SBATCH --output=emb_mlp_%j.out
#SBATCH --error=emb_mlp_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=mit_normal_gpu

# MLP on Embeddings Ablation Study
# Tests various strategies to prevent embedding overfitting:
# - Strong feature regularization
# - IRT initialization + regularization
# - PCA dimensionality reduction
# - Dropout
# - Early stopping
#
# Run with: sbatch experiment_a/mlp_ablation/slurm_embedding_mlp.sh

set -e

echo "=========================================="
echo "MLP on Embeddings Ablation Study"
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

# Run full ablation study (5-fold CV, all configurations)
echo ""
echo "Running MLP embedding ablation (5-fold CV)..."
python -m experiment_a.mlp_ablation.test_embedding_mlp

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
