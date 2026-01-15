#!/bin/bash
#SBATCH --job-name=sad_irt
#SBATCH --output=logs/sad_irt_%j.out
#SBATCH --error=logs/sad_irt_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1

# Configuration
OUTPUT_DIR="chris_output/sad_irt"
EPOCHS=10

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load modules (adjust based on cluster setup)
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check for existing checkpoint to resume from
# Find the most recently modified checkpoint (any type)
RESUME_ARG=""
LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint to resume from: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume_from $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh"
fi

# Set up distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

# Run training with accelerate for multi-GPU
accelerate launch \
    --num_processes=2 \
    --multi_gpu \
    --mixed_precision=bf16 \
    -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs $EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_ARG

echo "End time: $(date)"
echo "Exit code: $?"
