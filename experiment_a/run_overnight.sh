#!/bin/bash
# Overnight script to create sandboxes and grade them for Experiment A
#
# This script:
# 1. Runs sandbox creation and grading in parallel
# 2. Grading loop retries every 5 minutes to pick up newly created sandboxes
# 3. Uses caffeinate to prevent Mac from sleeping
#
# Usage:
#   ./experiment_a/run_overnight.sh
#
# To run with laptop closed, you need to:
#   System Settings > Battery > Options > Enable "Prevent automatic sleeping when the display is off"

set -e
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

echo "============================================================"
echo "EXPERIMENT A: Overnight Sandbox Creation + Grading"
echo "Started at: $(date)"
echo "============================================================"

# Create log directory
LOG_DIR="chris_output/experiment_a/logs"
mkdir -p "$LOG_DIR"

SANDBOX_LOG="$LOG_DIR/sandbox_creation_$(date +%Y%m%d_%H%M%S).log"
GRADING_LOG="$LOG_DIR/grading_$(date +%Y%m%d_%H%M%S).log"

echo "Sandbox log: $SANDBOX_LOG"
echo "Grading log: $GRADING_LOG"
echo ""

# Start sandbox creation in background with caffeinate
echo "Starting sandbox creation (batch_size=1, resume mode)..."
caffeinate -i python -m experiment_a.run_dummy_sandbox --resume --batch_size 1 > "$SANDBOX_LOG" 2>&1 &
SANDBOX_PID=$!

# Start grading loop in background
echo "Starting grading loop..."
(
    cd /Users/chrisge/Downloads/model_irt
    source .venv/bin/activate

    while true; do
        echo "[$(date)] Starting grading pass..."

        # Run grading with skip_existing
        python -m experiment_a.grade_sandbox_runs --skip_existing 2>&1 || true

        # Check if all 500 tasks are graded
        GRADED_COUNT=$(ls -1 chris_output/experiment_a/sandbox_features/*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "[$(date)] Graded so far: $GRADED_COUNT/500"

        if [ "$GRADED_COUNT" -ge 500 ]; then
            echo "[$(date)] All 500 tasks graded! Exiting grading loop."
            break
        fi

        # Wait 5 minutes before retrying (to let more sandboxes be created)
        echo "[$(date)] Waiting 5 minutes before next grading pass..."
        sleep 300
    done
) > "$GRADING_LOG" 2>&1 &
GRADING_PID=$!

echo ""
echo "Background processes started:"
echo "  Sandbox creation PID: $SANDBOX_PID"
echo "  Grading loop PID: $GRADING_PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $SANDBOX_LOG"
echo "  tail -f $GRADING_LOG"
echo ""
echo "To check graded count:"
echo "  ls -1 chris_output/experiment_a/sandbox_features/*.json | wc -l"
echo ""
echo "Waiting for both processes to complete..."
echo "(You can close this terminal - processes will continue in background)"
echo ""

# Use caffeinate to prevent sleep while waiting
caffeinate -i -w $SANDBOX_PID &
caffeinate -i -w $GRADING_PID &

# Wait for both to complete
wait $SANDBOX_PID 2>/dev/null || true
echo "Sandbox creation completed!"

wait $GRADING_PID 2>/dev/null || true
echo "Grading completed!"

echo ""
echo "============================================================"
echo "OVERNIGHT RUN COMPLETE"
echo "Finished at: $(date)"
echo "============================================================"

# Show final counts
GRADED_COUNT=$(ls -1 chris_output/experiment_a/sandbox_features/*.json 2>/dev/null | wc -l | tr -d ' ')

echo "Tasks graded: $GRADED_COUNT/500"
echo ""
echo "Next step - run evaluation:"
echo "  python -m experiment_a.train_evaluate"
