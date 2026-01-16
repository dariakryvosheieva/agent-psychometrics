#!/bin/bash
#SBATCH --job-name=traj_summarize
#SBATCH --output=logs/traj_summarize_%A_%a.out
#SBATCH --error=logs/traj_summarize_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# Trajectory Summarization Pipeline
# Uses vLLM with Qwen2.5-14B-Instruct (dense model, no MoE kernels needed)
# Runs as job array: each job gets 1 GPU and processes half the data
#
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/run_trajectory_summarization.sh

set -euo pipefail

# Project directory
cd ~/model_irt

# Activate virtual environment
source .venv/bin/activate

# Set HuggingFace cache to scratch (home quota is limited)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"

# Create directories
mkdir -p logs
mkdir -p chris_output/trajectory_summaries

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Configuration
TRAJECTORY_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/trajectory_summaries"
# Using dense model to avoid MoE kernel issues (vLLM _moe_C not compiled)
MODEL="Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE=16
NUM_SHARDS=2
SHARD_ID=${SLURM_ARRAY_TASK_ID}

echo "=== Configuration ==="
echo "Trajectories: ${TRAJECTORY_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL}"
echo "Batch size: ${BATCH_SIZE}"
echo "Shard: ${SHARD_ID}/${NUM_SHARDS}"
echo ""

# Count total trajectories
TOTAL_TRAJS=$(find ${TRAJECTORY_DIR} -name "*.json" -not -name "_*" 2>/dev/null | wc -l)
echo "Total trajectories: ${TOTAL_TRAJS}"
echo ""

# Run summarization for this shard
echo "=== Starting Shard ${SHARD_ID}/${NUM_SHARDS} ==="
python -m trajectory_summarization.run_summarization \
    --model_name "${MODEL}" \
    --trajectory_dir "${TRAJECTORY_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --shard_id ${SHARD_ID} \
    --num_shards ${NUM_SHARDS} \
    --batch_size ${BATCH_SIZE} \
    --max_num_seqs 16

echo ""
echo "=== Results ==="
echo "Date: $(date)"

# Count completed summaries
COMPLETED=$(find ${OUTPUT_DIR} -name "*.json" 2>/dev/null | wc -l)
echo "Total summaries on disk: ${COMPLETED}"

# Run aggregation only from shard 0 after completion
# (Shard 1 may finish first, but aggregation will just run twice - harmless)
if [ ${SHARD_ID} -eq 0 ]; then
    echo ""
    echo "=== Running Aggregation ==="
    python -m trajectory_summarization.aggregate_summaries \
        --input_dir "${OUTPUT_DIR}" \
        --output_file "${OUTPUT_DIR}/all_summaries.jsonl" || true
fi

echo ""
echo "Shard ${SHARD_ID} completed successfully"
