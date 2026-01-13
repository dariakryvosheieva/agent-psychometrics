#!/bin/bash
#SBATCH --job-name=test_embed
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_embed_%j.out
#SBATCH --error=logs/test_embed_%j.err

# Quick test job to verify embedding pipeline works
# Processes only 2 agents to check for errors before running full ablations
#
# Usage:
#   sbatch scripts/embedding/test_embedding_job.sh

set -e

# Load environment
module load miniforge
conda activate irt

# Enable fast HuggingFace downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=============================================="
echo "TEST: Trajectory Embedding Pipeline"
echo "=============================================="
echo "This is a quick test with only 2 agents"
echo "=============================================="

# Test with smallest model, limited agents
python -m experiment_b.compute_trajectory_embeddings \
    --trajectories_dir "trajectory_data/unified_trajs" \
    --output_dir "chris_output/experiment_b/trajectory_embeddings_test" \
    --backbone "Qwen/Qwen3-VL-8B-Instruct" \
    --embedding_layer -1 \
    --max_length 8192 \
    --batch_size 1 \
    --content_type "full" \
    --instruction_type "difficulty" \
    --limit_agents 2

echo ""
echo "=============================================="
echo "TEST COMPLETED SUCCESSFULLY"
echo "=============================================="
echo "Check output in: chris_output/experiment_b/trajectory_embeddings_test/full_difficulty/"
echo ""
echo "If this worked, you can now run the full ablations with:"
echo "  ./scripts/embedding/launch_embedding_ablations.sh"
