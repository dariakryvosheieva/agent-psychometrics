#!/bin/bash
#SBATCH --job-name=sad_irt_center
#SBATCH --output=logs/sad_irt_center_%j.out
#SBATCH --error=logs/sad_irt_center_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1

# Configuration - separate output dir for centering ablation
OUTPUT_DIR="chris_output/sad_irt_center"
EPOCHS=3

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Ablation: Centering (zero-mean only, no variance scaling)"

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check for existing checkpoint to resume from
RESUME_ARG=""
echo "Available checkpoints in $OUTPUT_DIR:"
ls -lt "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -5 || echo "  (none)"

LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo ""
    echo "Auto-selected most recent checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume_from $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting fresh"
fi

# Run training with centering (zero-mean only, no variance scaling)
python -m experiment_sad_irt.train_evaluate \
    --mode full_auc \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 1024 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --epochs $EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    --psi_normalization center \
    $RESUME_ARG

echo "End time: $(date)"
echo "Exit code: $?"
