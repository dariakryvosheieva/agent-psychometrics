#!/bin/bash
#SBATCH --job-name=sad_irt_long
#SBATCH --output=logs/sad_irt_long_%j.out
#SBATCH --error=logs/sad_irt_long_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1

# Configuration - 10 epochs for longer training
BASE_OUTPUT_DIR="chris_output/sad_irt_long"
EPOCHS=10
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse command line flags
DEBUG_GRADIENTS=""
NO_RESUME=""
PSI_NORM=""
PSI_NORM_VALUE=""
FREEZE_IRT=""
FREEZE_ENCODER=""
for arg in "$@"; do
    case $arg in
        --no_resume) NO_RESUME="--no_resume" ;;
        --debug_gradients) DEBUG_GRADIENTS="--debug_gradients" ;;
        --psi_normalization=*)
            PSI_NORM_VALUE="${arg#*=}"
            PSI_NORM="--psi_normalization $PSI_NORM_VALUE"
            ;;
        --freeze_irt) FREEZE_IRT="--freeze_irt" ;;
        --freeze_encoder) FREEZE_ENCODER="--freeze_encoder" ;;
    esac
done

# Build output directory based on ablation hyperparams
OUTPUT_DIR="$BASE_OUTPUT_DIR"
if [ -n "$FREEZE_ENCODER" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/freeze_encoder_${TIMESTAMP}"
elif [ -n "$FREEZE_IRT" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/freeze_irt_${TIMESTAMP}"
else
    OUTPUT_DIR="${OUTPUT_DIR}/full_${TIMESTAMP}"
fi
if [ -n "$PSI_NORM_VALUE" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_psi_${PSI_NORM_VALUE}"
fi

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Epochs: $EPOCHS"
echo "Freeze IRT: ${FREEZE_IRT:-no}"
echo "Freeze Encoder: ${FREEZE_ENCODER:-no}"

# Load modules
module load miniforge 2>/dev/null || true

# Activate environment
source .venv/bin/activate

# Enable fast HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check for existing checkpoint to resume from
RESUME_ARG=""

if [ "$NO_RESUME" = "--no_resume" ]; then
    echo "Skipping checkpoint loading (--no_resume flag set)"
else
    echo "Available checkpoints in $OUTPUT_DIR:"
    ls -lt "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -5 || echo "  (none)"

    LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo ""
        echo "Auto-selected most recent checkpoint: $LATEST_CHECKPOINT"
        echo "To skip checkpoint loading, run with: sbatch script.sh --no_resume"
        RESUME_ARG="--resume_from $LATEST_CHECKPOINT"
    else
        echo "No checkpoint found, starting fresh"
    fi
fi

# Run training (single GPU)
# Without gradient checkpointing, use smaller batch with accumulation
# Effective batch = 32 * 2 = 64
python -m experiment_sad_irt.train_evaluate \
    --frontier_cutoff_date 20250807 \
    --model_name Qwen/Qwen3-0.6B \
    --max_length 1024 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --epochs $EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    $DEBUG_GRADIENTS \
    $PSI_NORM \
    $FREEZE_IRT \
    $FREEZE_ENCODER \
    $RESUME_ARG

echo "End time: $(date)"
echo "Exit code: $?"
