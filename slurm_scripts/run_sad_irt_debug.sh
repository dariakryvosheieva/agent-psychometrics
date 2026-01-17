#!/bin/bash
#SBATCH --job-name=sad_irt_debug
#SBATCH --output=logs/sad_irt_debug_%j.out
#SBATCH --error=logs/sad_irt_debug_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1

# Configuration - debug run with gradient logging
OUTPUT_DIR="chris_output/sad_irt_debug"
EPOCHS=1

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "DEBUG RUN: Gradient logging enabled"

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run training with debug gradients and centering
# Using smaller logging interval (every 5 steps) for more visibility
python -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 1024 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --epochs $EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    --psi_normalization center \
    --debug_gradients \
    --logging_steps 5 \
    --eval_steps 50

echo "End time: $(date)"
echo "Exit code: $?"
