#!/bin/bash
#SBATCH --job-name=traj_embed
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/traj_embed_%j.out
#SBATCH --error=logs/traj_embed_%j.err

# Compute trajectory embeddings for Experiment B posterior difficulty prediction
#
# Usage:
#   # Basic usage (default Qwen3-VL-8B-Instruct)
#   sbatch compute_trajectory_embeddings_engaging.sh
#
#   # With custom settings
#   sbatch --export=BACKBONE="Qwen/Qwen3-32B",CONTENT_TYPE="full",INSTRUCTION_TYPE="difficulty" \
#       compute_trajectory_embeddings_engaging.sh
#
#   # Parallel sharding (run 2 jobs processing different agents)
#   sbatch --export=SHARD_ID=0,NUM_SHARDS=2 compute_trajectory_embeddings_engaging.sh
#   sbatch --export=SHARD_ID=1,NUM_SHARDS=2 compute_trajectory_embeddings_engaging.sh

set -e

# Load environment
module load miniforge
conda activate irt

# Enable fast HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Configuration with defaults
BACKBONE="${BACKBONE:-Qwen/Qwen3-VL-8B-Instruct}"
CONTENT_TYPE="${CONTENT_TYPE:-full}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-difficulty}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# Paths
TRAJECTORIES_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/experiment_b/trajectory_embeddings"

echo "=============================================="
echo "Trajectory Embedding Computation"
echo "=============================================="
echo "Backbone: $BACKBONE"
echo "Content type: $CONTENT_TYPE"
echo "Instruction type: $INSTRUCTION_TYPE"
echo "Max length: $MAX_LENGTH"
echo "Shard: $SHARD_ID / $NUM_SHARDS"
echo "=============================================="

# Run embedding computation
python -m experiment_b.compute_trajectory_embeddings \
    --trajectories_dir "$TRAJECTORIES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --embedding_layer -1 \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --content_type "$CONTENT_TYPE" \
    --instruction_type "$INSTRUCTION_TYPE" \
    --shard_id "$SHARD_ID" \
    --num_shards "$NUM_SHARDS"

echo "Done!"
