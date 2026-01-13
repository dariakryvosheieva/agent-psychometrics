#!/bin/bash
# Overnight Lunette feature extraction script
#
# Usage:
#   ./experiment_a/run_overnight.sh              # Run all 500 tasks
#   ./experiment_a/run_overnight.sh --test_only  # Run only 100 test tasks
#
# Run with nohup for overnight operation:
#   nohup ./experiment_a/run_overnight.sh &> overnight.log &
#   tail -f overnight.log  # Monitor progress

set -e
cd "$(dirname "$0")/.."  # Navigate to repo root

echo "======================================================================"
echo "Starting Overnight Lunette Extraction"
echo "Started at: $(date)"
echo "======================================================================"

# Activate virtual environment
source .venv/bin/activate

# Run the extraction script
# Use --test_only to focus on test tasks first for stable AUC
# Use --resume if restarting after interruption
python -m experiment_a.overnight_lunette_extraction \
    --concurrency 5 \
    "$@"

echo ""
echo "======================================================================"
echo "Extraction Complete!"
echo "Finished at: $(date)"
echo "======================================================================"

# Run evaluation with v2 features
echo ""
echo "Running evaluation with v2 features..."
python -m experiment_a.run_evaluation_v2 --with_embeddings

echo ""
echo "Done!"
