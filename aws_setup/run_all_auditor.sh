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
#   python -m llm_judge_feature_extraction.auditor_agent.run_auditor --dataset gso --batch_size 50 --max_connections 30

set -euo pipefail

S3_BUCKET="fulcrum-auditor-agent-results-2026"

echo "=== Running V4 auditor on all datasets ==="
echo "Start time: $(date)"
echo ""

# Download any existing results from S3 (for resuming after a previous run)
echo "=== Downloading existing results from S3 ==="
mkdir -p chris_output/auditor_features
aws s3 sync "s3://$S3_BUCKET/auditor_features/" chris_output/auditor_features/ || true
echo ""

# GSO — 102 tasks, images share base layers, can do all at once
echo "=== [1/4] GSO (102 tasks) ==="
python -m llm_judge_feature_extraction.auditor_agent.run_auditor \
    --dataset gso \
    --batch_size 50 \
    --max_connections 30
echo "GSO complete: $(date)"
docker system prune -af
echo ""

# Terminal Bench — 89 tasks, different repos per task
echo "=== [2/4] Terminal Bench (89 tasks) ==="
python -m llm_judge_feature_extraction.auditor_agent.run_auditor \
    --dataset terminalbench \
    --batch_size 44 \
    --max_connections 30
echo "Terminal Bench complete: $(date)"
docker system prune -af
echo ""

# SWE-bench Pro — 731 tasks, largest dataset
echo "=== [3/4] SWE-bench Pro (731 tasks) ==="
python -m llm_judge_feature_extraction.auditor_agent.run_auditor \
    --dataset swebench_pro \
    --batch_size 50 \
    --max_connections 30
echo "SWE-bench Pro complete: $(date)"
docker system prune -af
echo ""

# SWE-bench Verified — 500 tasks, re-extract with V4
echo "=== [4/4] SWE-bench Verified (500 tasks) ==="
python -m llm_judge_feature_extraction.auditor_agent.run_auditor \
    --dataset swebench_verified \
    --batch_size 50 \
    --max_connections 30
echo "SWE-bench Verified complete: $(date)"
echo ""

echo "=== All datasets complete ==="
echo "End time: $(date)"

# Upload results to S3 and self-terminate
echo ""
echo "=== Uploading results to S3 ==="
aws s3 sync chris_output/auditor_features/ "s3://$S3_BUCKET/auditor_features/"
echo "Upload complete. Results in s3://$S3_BUCKET/auditor_features/"
echo ""
echo "=== Shutting down instance ==="
sudo shutdown -h now
