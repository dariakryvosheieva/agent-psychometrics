#!/bin/bash
# Fix unified trajectories with list content on cluster
#
# Usage:
#   # Dry run first
#   bash scripts/fix_unified_trajectories.sh --dry_run
#
#   # Apply fixes
#   bash scripts/fix_unified_trajectories.sh

set -e

# Activate virtual environment
source .venv/bin/activate

# Run the fix script
python scripts/fix_unified_trajectories.py "$@"

echo ""
echo "Done! You can now resume compute_trajectory_embeddings.py"
