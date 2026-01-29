#!/bin/bash
#SBATCH --job-name=mlp_optimal
#SBATCH --output=chris_output/experiment_a/mlp_ablation/optimal_hyperparams_%j.out
#SBATCH --error=chris_output/experiment_a/mlp_ablation/optimal_hyperparams_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Setup environment
cd /home/chMDge/model_irt
source .venv/bin/activate

# HuggingFace cache on scratch to avoid quota
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

echo "Starting optimal hyperparameter search at $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m experiment_a.mlp_ablation.test_optimal_hyperparams

echo "Finished at $(date)"
