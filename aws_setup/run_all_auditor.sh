#!/bin/bash
# Run V4 auditor agent on all datasets sequentially.
#
# Each dataset resumes independently via incremental CSV — if interrupted,
# just re-run this script and it picks up where it left off.
#
# Usage:
#   cd model_irt && source .venv/bin/activate
#   bash aws_setup/run_all_auditor.sh
#
# To run a single dataset:
#   python -m experiment_a.auditor_agent.run_auditor --dataset gso --batch_size 50 --max_connections 30

set -euo pipefail

echo "=== Running V4 auditor on all datasets ==="
echo "Start time: $(date)"
echo ""

# GSO — 102 tasks, images share base layers, can do all at once
echo "=== [1/4] GSO (102 tasks) ==="
python -m experiment_a.auditor_agent.run_auditor \
    --dataset gso \
    --batch_size 50 \
    --max_connections 30
echo "GSO complete: $(date)"
echo ""

# Terminal Bench — 88 tasks, different repos per task
echo "=== [2/4] Terminal Bench (88 tasks) ==="
python -m experiment_a.auditor_agent.run_auditor \
    --dataset terminalbench \
    --batch_size 44 \
    --max_connections 30
echo "Terminal Bench complete: $(date)"
echo ""

# SWE-bench Pro — 731 tasks, largest dataset
echo "=== [3/4] SWE-bench Pro (731 tasks) ==="
python -m experiment_a.auditor_agent.run_auditor \
    --dataset swebench_pro \
    --batch_size 50 \
    --max_connections 30
echo "SWE-bench Pro complete: $(date)"
echo ""

# SWE-bench Verified — 500 tasks, re-extract with V4
echo "=== [4/4] SWE-bench Verified (500 tasks) ==="
python -m experiment_a.auditor_agent.run_auditor \
    --dataset swebench \
    --batch_size 50 \
    --max_connections 30
echo "SWE-bench Verified complete: $(date)"
echo ""

echo "=== All datasets complete ==="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  chris_output/auditor_features/gso_v4/auditor_features_incremental.csv"
echo "  chris_output/auditor_features/terminalbench_v4/auditor_features_incremental.csv"
echo "  chris_output/auditor_features/swebench_pro_v4/auditor_features_incremental.csv"
echo "  chris_output/auditor_features/swebench_verified_v4/auditor_features_incremental.csv"
echo ""
echo "scp them back with:"
echo "  scp -i ~/.ssh/auditor-key.pem ec2-user@\$(curl -s ifconfig.me):~/model_irt/chris_output/auditor_features/*/auditor_features_incremental.csv ."
