#!/bin/bash
#SBATCH --job-name=traj_summarize
#SBATCH --output=logs/traj_summarize_%j.out
#SBATCH --error=logs/traj_summarize_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16

# Trajectory Summarization Pipeline
# Uses vLLM with Qwen3-8B-Instruct to summarize agent trajectories
# Each GPU runs an independent copy of the model processing half the data
#
# Run from: ~/model_irt
# Submit with: sbatch slurm_scripts/run_trajectory_summarization.sh

set -euo pipefail

# Project directory
cd ~/model_irt

# Load modules and activate conda environment
module load miniforge
conda activate irt

# Set HuggingFace cache (avoids filling home quota)
export HF_HOME="${PWD}/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}"

# Create directories
mkdir -p logs
mkdir -p chris_output/trajectory_summaries

echo "=== Environment Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Configuration
TRAJECTORY_DIR="trajectory_data/unified_trajs"
OUTPUT_DIR="chris_output/trajectory_summaries"
MODEL="Qwen/Qwen3-8B-Instruct"
BATCH_SIZE=16

echo "=== Configuration ==="
echo "Trajectories: ${TRAJECTORY_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL}"
echo "Batch size: ${BATCH_SIZE}"
echo "Using 2 GPUs with data parallelism"
echo ""

# Count total trajectories
TOTAL_TRAJS=$(find ${TRAJECTORY_DIR} -name "*.json" -not -name "_*" 2>/dev/null | wc -l)
echo "Total trajectories: ${TOTAL_TRAJS}"
echo ""

# Run GPU 0 (shard 0/2)
echo "=== Starting GPU 0 (shard 0/2) ==="
CUDA_VISIBLE_DEVICES=0 python -m trajectory_summarization.run_summarization \
    --model_name "${MODEL}" \
    --trajectory_dir "${TRAJECTORY_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --shard_id 0 \
    --num_shards 2 \
    --batch_size ${BATCH_SIZE} \
    > logs/traj_summarize_shard0_${SLURM_JOB_ID}.log 2>&1 &
PID0=$!

# Run GPU 1 (shard 1/2)
echo "=== Starting GPU 1 (shard 1/2) ==="
CUDA_VISIBLE_DEVICES=1 python -m trajectory_summarization.run_summarization \
    --model_name "${MODEL}" \
    --trajectory_dir "${TRAJECTORY_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --shard_id 1 \
    --num_shards 2 \
    --batch_size ${BATCH_SIZE} \
    > logs/traj_summarize_shard1_${SLURM_JOB_ID}.log 2>&1 &
PID1=$!

echo "Waiting for both shards to complete..."
echo "Shard 0 PID: ${PID0}"
echo "Shard 1 PID: ${PID1}"
echo ""

# Monitor progress periodically
while kill -0 ${PID0} 2>/dev/null || kill -0 ${PID1} 2>/dev/null; do
    sleep 300  # Check every 5 minutes
    COMPLETED=$(find ${OUTPUT_DIR} -name "*.json" 2>/dev/null | wc -l)
    echo "$(date): ${COMPLETED} summaries completed"
done

# Wait and capture status
wait ${PID0}
STATUS0=$?
echo "Shard 0 completed with status: ${STATUS0}"

wait ${PID1}
STATUS1=$?
echo "Shard 1 completed with status: ${STATUS1}"

echo ""
echo "=== Results ==="
echo "Date: $(date)"

# Count completed summaries
COMPLETED=$(find ${OUTPUT_DIR} -name "*.json" 2>/dev/null | wc -l)
echo "Completed summaries: ${COMPLETED} / ${TOTAL_TRAJS}"

# Show output directory
ls -la "${OUTPUT_DIR}/" 2>/dev/null | head -20

# Report final status
if [ ${STATUS0} -eq 0 ] && [ ${STATUS1} -eq 0 ]; then
    echo ""
    echo "SUCCESS: Both shards completed successfully"

    # Run aggregation
    echo ""
    echo "=== Running Aggregation ==="
    python -m trajectory_summarization.aggregate_summaries \
        --input_dir "${OUTPUT_DIR}" \
        --output_file "${OUTPUT_DIR}/all_summaries.jsonl"

    exit 0
else
    echo ""
    echo "ERROR: One or more shards failed"
    echo "Check logs/traj_summarize_shard*_${SLURM_JOB_ID}.log for details"

    # Still try to aggregate what we have
    echo ""
    echo "=== Aggregating partial results ==="
    python -m trajectory_summarization.aggregate_summaries \
        --input_dir "${OUTPUT_DIR}" \
        --output_file "${OUTPUT_DIR}/all_summaries.jsonl" || true

    exit 1
fi
