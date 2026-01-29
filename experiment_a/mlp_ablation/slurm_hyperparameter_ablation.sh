#!/bin/bash
#SBATCH --job-name=mlp_ablation
#SBATCH --output=logs/mlp_ablation_%j.out
#SBATCH --error=logs/mlp_ablation_%j.err
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=64G
#SBATCH --partition=mit_normal_gpu

# Hyperparameter ablation for AgentEmbeddingPredictor
# Tests combinations of LR, weight decay, dropout, hidden dims, and init strategy
# Splits work across 2 H200 GPUs to maximize throughput
#
# Run with: mkdir -p logs && sbatch experiment_a/mlp_ablation/slurm_hyperparameter_ablation.sh

set -e

echo "=========================================="
echo "MLP Hyperparameter Ablation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "=========================================="

cd /home/cge7/model_irt
source .venv/bin/activate

export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export PYTHONUNBUFFERED=1

# Check available GPUs
echo ""
echo "Available GPUs:"
nvidia-smi --list-gpus

echo ""
echo "Running ablation on 2 H200 GPUs in parallel..."
echo ""

# Run both GPU jobs in parallel
python -m experiment_a.mlp_ablation.run_hyperparameter_ablation --gpu 0 --total_gpus 2 &
PID0=$!

python -m experiment_a.mlp_ablation.run_hyperparameter_ablation --gpu 1 --total_gpus 2 &
PID1=$!

# Wait for both to complete
echo "GPU 0 PID: $PID0"
echo "GPU 1 PID: $PID1"
echo ""

wait $PID0
STATUS0=$?
echo "GPU 0 finished with status: $STATUS0"

wait $PID1
STATUS1=$?
echo "GPU 1 finished with status: $STATUS1"

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "GPU 0 exit status: $STATUS0"
echo "GPU 1 exit status: $STATUS1"
echo "=========================================="

# Exit with error if either failed
if [ $STATUS0 -ne 0 ] || [ $STATUS1 -ne 0 ]; then
    exit 1
fi
