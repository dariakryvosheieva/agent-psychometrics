#!/bin/bash
#SBATCH --job-name=tb_embed
#SBATCH --output=logs/terminalbench_embeddings_%j.out
#SBATCH --error=logs/terminalbench_embeddings_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8

# TerminalBench Embeddings Generation
# Generates embeddings for all 89 TerminalBench tasks using configurable backbone.
#
# Usage:
#   sbatch slurm_scripts/terminalbench_embeddings.sh [BACKBONE]
#
# Examples:
#   sbatch slurm_scripts/terminalbench_embeddings.sh  # Uses default Qwen3-VL-8B
#   sbatch slurm_scripts/terminalbench_embeddings.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

set -e

# Backbone model (can be passed as argument)
BACKBONE="${1:-Qwen/Qwen3-VL-8B-Instruct}"

echo "=== TerminalBench Embeddings Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Backbone: $BACKBONE"
echo ""

# Set up environment
cd ~/model_irt
source .venv/bin/activate

# Use scratch for HuggingFace cache (avoid home quota limits)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
mkdir -p "$HF_HOME"

# Create output directory
OUTPUT_DIR="chris_output/experiment_a_terminalbench/embeddings"
mkdir -p "$OUTPUT_DIR"

# Create logs directory if needed
mkdir -p logs

echo "Environment:"
echo "  Python: $(which python)"
echo "  HF_HOME: $HF_HOME"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run embedding generation
python -m experiment_a_terminalbench.generate_embeddings \
    --items_path chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv \
    --repo_path terminal-bench \
    --out_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --max_length 8192 \
    --batch_size 1 \
    --device_map auto \
    --torch_dtype bfloat16 \
    --trust_remote_code

echo ""
echo "=== Embedding generation complete ==="
echo "End time: $(date)"
