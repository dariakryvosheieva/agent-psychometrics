#!/bin/bash
#SBATCH --job-name=tb_embed
#SBATCH --output=slurm_logs/terminalbench_embeddings_%j.out
#SBATCH --error=slurm_logs/terminalbench_embeddings_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# TerminalBench Embeddings Generation
# Generates VLM embeddings for all 89 TerminalBench tasks using Qwen3-VL-8B-Instruct

set -e

echo "=== TerminalBench Embeddings Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set up environment
cd /home/chrisge/orcd/model_irt
source .venv/bin/activate

# Use scratch for HuggingFace cache (avoid home quota limits)
export HF_HOME="$HOME/orcd/scratch/.cache/huggingface"
mkdir -p "$HF_HOME"

# Create output directory
OUTPUT_DIR="chris_output/experiment_a_terminalbench/embeddings"
mkdir -p "$OUTPUT_DIR"

# Create slurm logs directory if needed
mkdir -p slurm_logs

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  HF_HOME: $HF_HOME"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run embedding generation
python -m experiment_a_terminalbench.generate_embeddings \
    --items_path chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv \
    --repo_path terminal-bench \
    --output_path "$OUTPUT_DIR/embeddings.npz" \
    --backbone Qwen/Qwen3-VL-8B-Instruct \
    --max_length 8192 \
    --batch_size 4

echo ""
echo "=== Embedding generation complete ==="
echo "End time: $(date)"
echo "Output saved to: $OUTPUT_DIR/embeddings.npz"
