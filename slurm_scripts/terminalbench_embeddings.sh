#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G

# TerminalBench Embeddings Generation
# Generates embeddings for all 89 TerminalBench tasks using configurable backbone.
#
# Usage:
#   sbatch slurm_scripts/terminalbench_embeddings.sh [BACKBONE]
#
# Examples:
#   sbatch slurm_scripts/terminalbench_embeddings.sh  # Uses default Qwen3-VL-8B
#   sbatch slurm_scripts/terminalbench_embeddings.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

set -euo pipefail

# Backbone model (can be passed as argument)
BACKBONE="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

echo "=== TerminalBench Embeddings Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Backbone: $BACKBONE"
echo ""

# Scratch (writable on compute nodes)
SCRATCH_DIR="${SCRATCH:-/orcd/scratch/orcd/001/${USER}}"
mkdir -p "$SCRATCH_DIR"

# Always run from the submit directory (SLURM copies scripts to /var/spool/...)
REPO_DIR="${SLURM_SUBMIT_DIR:-/orcd/scratch/orcd/001/${USER}/fulcrum/fellowship}"
cd "$REPO_DIR"

# Use scratch for HuggingFace + pip caches (avoid home quota limits)
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_DIR}/.cache/pip}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

if [ -f ".venv/bin/activate" ]; then
    # Project-local virtualenv
    source .venv/bin/activate
else
    # Bootstrap a scratch virtualenv on the fly (cluster-friendly).
    # This avoids relying on a pre-existing named conda env like "irt", and avoids
    # writing into /var/spool on compute nodes.
    module load miniforge 2>/dev/null || true

    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: Python not found. Load an environment that provides python." >&2
        exit 1
    fi

    VENV_DIR="${SCRATCH_DIR}/venvs/fellowship_terminalbench"
    if [ ! -f "${VENV_DIR}/bin/activate" ]; then
        echo "No usable venv found; creating one at ${VENV_DIR}"
        python -m venv "${VENV_DIR}"
        source "${VENV_DIR}/bin/activate"

        # Install dependencies needed for embedding generation.
        # `requirements.txt` is for the broader project; we additionally install
        # runtime deps used by `predict_question_difficulty.py` and TerminalBench loaders.
        python -m pip install --upgrade pip wheel

        if ! python -c "import torch" >/dev/null 2>&1; then
            python -m pip install torch
        fi

        python -m pip install -r requirements.txt
        python -m pip install transformers datasets huggingface_hub tqdm accelerate pyyaml
        touch "${VENV_DIR}/.built_ok"
    else
        source "${VENV_DIR}/bin/activate"
    fi
fi

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
python -m experiment_a.terminalbench.generate_embeddings \
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
