#!/bin/bash
#SBATCH --job-name=exp_a_mlp
#SBATCH --output=logs/exp_a_mlp_%j.out
#SBATCH --error=logs/exp_a_mlp_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

# Experiment A with MLP predictors
# Runs all datasets with MLP methods enabled (parallel by default)

set -e

echo "============================================"
echo "Experiment A with MLP Predictors"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================"

# Activate virtual environment
source ~/model_irt/.venv/bin/activate

# Set HuggingFace cache to scratch (avoid home quota issues)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"

# Navigate to project
cd ~/model_irt

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check Python and PyTorch
echo ""
echo "Environment:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Create output directories
mkdir -p chris_output/experiment_a
mkdir -p logs

# Run all datasets with MLP predictors (parallel by default)
echo ""
echo "Running Experiment A on all datasets (parallel)..."
python -m experiment_a.run_all_datasets \
    --output chris_output/experiment_a/mlp_results.csv \
    --output_dir chris_output/experiment_a/mlp_debug \
    --max_workers 4

echo ""
echo "============================================"
echo "Completed at: $(date)"
echo "============================================"
