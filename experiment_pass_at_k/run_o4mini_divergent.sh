#!/bin/bash
# Run o4-mini with k=10 on divergent tasks (where gpt-5.2 passes but o4-mini fails in pass@1)

set -e

cd /Users/chrisge/Downloads/model_irt
source .venv/bin/activate

# 6 divergent tasks from screening
TASKS="matplotlib__matplotlib-23412,pydata__xarray-6461,django__django-12304,django__django-14122,pydata__xarray-2905,django__django-12125"

echo "=============================================="
echo "o4-mini Pass@10 on Divergent Tasks"
echo "=============================================="
echo ""
echo "Tasks (gpt-5.2 passes, o4-mini fails in pass@1):"
echo "  $TASKS"
echo "=============================================="
echo ""

python -m experiment_pass_at_k.run_pass_k \
    --model openai/o4-mini \
    --task_ids "$TASKS" \
    --k 10 \
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
