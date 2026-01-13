#!/bin/bash
# Launch embedding ablation jobs on MIT Engaging
#
# This script submits multiple jobs for different backbone/content/instruction combinations.
# Due to the 2-job limit, it waits for job slots before submitting new ones.
#
# Usage:
#   ./launch_embedding_ablations.sh              # Run all ablations
#   ./launch_embedding_ablations.sh --dry_run   # Show what would be submitted

set -e

DRY_RUN=false
if [[ "$1" == "--dry_run" ]]; then
    DRY_RUN=true
    echo "DRY RUN MODE - showing commands without submitting"
fi

# Priority models (ordered by expected importance)
BACKBONES=(
    "Qwen/Qwen3-VL-8B-Instruct"      # Daria's baseline (fastest, start here)
    "Qwen/Qwen3-VL-32B-Instruct"     # Larger VL model
    "Qwen/Qwen3-30B-A3B"             # MoE efficient scaling
    "Qwen/Qwen3-32B"                 # Dense large model
    "Qwen/Qwen3-Embedding-8B"        # Purpose-built embedding model
)

# Content types (most to least information)
CONTENTS=(
    "full"              # Complete trajectory + task + solution
    "condensed"         # Summarized trajectory
    "failure_focused"   # Only errors/failures
    "no_solution"       # Generalization test (no gold solution)
)

# Instruction suffixes
INSTRUCTIONS=(
    "difficulty"    # Same as prior (baseline)
    "residual"      # Predict residual directly
    "progress"      # How close was agent to solving
    "closeness"     # Similar framing to progress
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="$SCRIPT_DIR/compute_trajectory_embeddings_engaging.sh"

# Tracking
SUBMITTED=0
SKIPPED=0

wait_for_job_slot() {
    # Wait until we have < 2 jobs running
    while true; do
        RUNNING=$(squeue -u $USER -h 2>/dev/null | wc -l || echo "0")
        if [[ "$RUNNING" -lt 2 ]]; then
            return
        fi
        echo "  Waiting for job slot (currently $RUNNING running)..."
        sleep 60
    done
}

submit_job() {
    local backbone="$1"
    local content="$2"
    local instruction="$3"

    # Create unique job name
    local safe_backbone=$(echo "$backbone" | tr '/' '_' | tr '-' '_')
    local job_name="emb_${content}_${instruction}_${safe_backbone}"

    echo "Submitting: backbone=$backbone content=$content instruction=$instruction"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] sbatch --job-name=$job_name --export=BACKBONE=$backbone,CONTENT_TYPE=$content,INSTRUCTION_TYPE=$instruction $SUBMIT_SCRIPT"
        return
    fi

    wait_for_job_slot

    sbatch \
        --job-name="$job_name" \
        --export="BACKBONE=$backbone,CONTENT_TYPE=$content,INSTRUCTION_TYPE=$instruction" \
        "$SUBMIT_SCRIPT"

    ((SUBMITTED++)) || true
    sleep 2  # Brief pause between submissions
}

echo "=============================================="
echo "Embedding Ablation Launcher"
echo "=============================================="
echo "Backbones: ${#BACKBONES[@]}"
echo "Content types: ${#CONTENTS[@]}"
echo "Instructions: ${#INSTRUCTIONS[@]}"
echo "Total combinations: $((${#BACKBONES[@]} * ${#CONTENTS[@]} * ${#INSTRUCTIONS[@]}))"
echo "=============================================="

# Run ablations in priority order
# Start with baseline backbone, sweep content and instruction
for backbone in "${BACKBONES[@]}"; do
    echo ""
    echo "=== Backbone: $backbone ==="

    for content in "${CONTENTS[@]}"; do
        for instruction in "${INSTRUCTIONS[@]}"; do
            submit_job "$backbone" "$content" "$instruction"
        done
    done
done

echo ""
echo "=============================================="
echo "Done! Submitted: $SUBMITTED jobs"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "(Dry run - no jobs actually submitted)"
fi
echo "=============================================="
