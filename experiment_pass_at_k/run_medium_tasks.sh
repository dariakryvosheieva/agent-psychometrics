#!/bin/bash
# Run pass@k experiment on medium difficulty tasks (around 0)
# Both o1 and o3, sequentially, with per-task parallelization

set -e

cd /Users/chrisge/Downloads/model_irt
source .venv/bin/activate

TASK_IDS="django__django-11211,sphinx-doc__sphinx-9281,django__django-15930,scikit-learn__scikit-learn-10908,django__django-11333"

echo "=============================================="
echo "Running pass@k on medium difficulty tasks"
echo "Tasks: $TASK_IDS"
echo "=============================================="
echo ""

echo "Running o1-2024-12-17..."
python -m experiment_pass_at_k.run_pass_k \
    --model openai/o1-2024-12-17 \
    --task_ids "$TASK_IDS" \
    --parallel-tasks 1

echo ""
echo "Running o3-2025-04-16..."
python -m experiment_pass_at_k.run_pass_k \
    --model openai/o3-2025-04-16 \
    --task_ids "$TASK_IDS" \
    --parallel-tasks 1

echo ""
echo "Cleaning up Docker state..."
docker stop $(docker ps -q) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true
docker system prune -f 2>/dev/null || true

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "=============================================="
