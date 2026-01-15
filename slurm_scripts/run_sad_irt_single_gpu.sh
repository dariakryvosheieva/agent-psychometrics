#!/bin/bash
#SBATCH --job-name=sad_irt
#SBATCH --output=logs/sad_irt_%j.out
#SBATCH --error=logs/sad_irt_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
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

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check for existing checkpoint to resume from
# Find the most recently modified checkpoint (any type)
RESUME_ARG=""
echo "Available checkpoints in $OUTPUT_DIR:"
ls -lt "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -5 || echo "  (none)"

LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "Auto-selected most recent checkpoint: $LATEST_CHECKPOINT"
    echo "To use a different checkpoint, cancel and run with: --resume_from <path>"
    RESUME_ARG="--resume_from $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh"
fi

# Run training (single GPU)
python -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 8192 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs $EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_ARG

echo "End time: $(date)"
echo "Exit code: $?"
