#!/bin/bash
#SBATCH --job-name=swe_embed
#SBATCH --output=logs/swebench_embeddings_%A_%a.out
#SBATCH --error=logs/swebench_embeddings_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# SWE-bench Verified Embeddings Generation
# Generates embeddings for all 500 SWE-bench Verified tasks using task parallelism.
# Tasks are split across 2 H200 GPUs (250 tasks each).
#
# Usage:
#   sbatch slurm_scripts/swebench_embeddings.sh [BACKBONE]
#
# Examples:
#   sbatch slurm_scripts/swebench_embeddings.sh  # Uses default Qwen3-VL-8B
#   sbatch slurm_scripts/swebench_embeddings.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#
# After both jobs complete, merge shards:
#   python -m experiment_a.merge_embedding_shards \
#       --shard_pattern "chris_output/experiment_a/embeddings/embeddings__${BACKBONE}__*__shard*.npz" \
#       --out_path "chris_output/experiment_a/embeddings/embeddings__${BACKBONE}__merged.npz"

set -e

# Backbone model (can be passed as argument)
BACKBONE="${1:-Qwen/Qwen3-VL-8B-Instruct}"

# Task sharding configuration
TOTAL_TASKS=500
NUM_SHARDS=2
TASKS_PER_SHARD=$((TOTAL_TASKS / NUM_SHARDS))
START_IDX=$((SLURM_ARRAY_TASK_ID * TASKS_PER_SHARD))

# Handle last shard (may have extra tasks if not evenly divisible)
if [ $SLURM_ARRAY_TASK_ID -eq $((NUM_SHARDS - 1)) ]; then
    N_INPUTS=$((TOTAL_TASKS - START_IDX))
else
    N_INPUTS=$TASKS_PER_SHARD
fi

echo "=== SWE-bench Verified Embeddings Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Backbone: $BACKBONE"
echo "  Total tasks: $TOTAL_TASKS"
echo "  Shard: $SLURM_ARRAY_TASK_ID / $((NUM_SHARDS - 1))"
echo "  Task range: $START_IDX to $((START_IDX + N_INPUTS - 1))"
echo ""

# Set up environment
cd ~/model_irt
source .venv/bin/activate

# Use scratch for HuggingFace cache (avoid home quota limits)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
mkdir -p "$HF_HOME"

# Create output directory
OUTPUT_DIR="chris_output/experiment_a/embeddings"
mkdir -p "$OUTPUT_DIR"

# Create logs directory if needed
mkdir -p logs

echo "Environment:"
echo "  Python: $(which python)"
echo "  HF_HOME: $HF_HOME"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run embedding generation for this shard
python -m experiment_a.generate_embeddings \
    --backbone "$BACKBONE" \
    --start_idx $START_IDX \
    --n_inputs $N_INPUTS \
    --max_length 8192 \
    --batch_size 1 \
    --device_map auto \
    --torch_dtype bfloat16 \
    --trust_remote_code \
    --out_dir "$OUTPUT_DIR"

echo ""
echo "=== Shard $SLURM_ARRAY_TASK_ID complete ==="
echo "End time: $(date)"
echo ""
echo "After all shards complete, merge with:"
echo "  python -m experiment_a.merge_embedding_shards \\"
echo "      --shard_pattern '$OUTPUT_DIR/embeddings__*__shard*.npz' \\"
echo "      --out_path '$OUTPUT_DIR/embeddings__merged.npz'"
