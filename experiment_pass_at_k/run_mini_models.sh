#!/bin/bash
# Run pass@k experiment comparing gpt-5-mini and o4-mini
# Tasks selected from difficulty range [-1.5, 1.0] where at least one model passes

set -e

cd /Users/chrisge/Downloads/model_irt
source .venv/bin/activate

# 10 tasks evenly spread across difficulty range -1.5 to 1.0
# At least one of gpt-5 or o4-mini historically passes each task
TASKS="django__django-11551,sphinx-doc__sphinx-9711,scikit-learn__scikit-learn-11310,sphinx-doc__sphinx-8475,django__django-14559,django__django-13401,django__django-13315,sphinx-doc__sphinx-9673,django__django-16877,django__django-14311"

echo "=============================================="
echo "GPT-5-mini vs o4-mini Pass@10 Experiment"
echo "=============================================="
echo ""
echo "Tasks (difficulty -1.5 to 1.0):"
echo "  $TASKS"
echo "=============================================="
echo ""

echo "Running gpt-5-mini..."
python -m experiment_pass_at_k.run_pass_k \
    --model openai/gpt-5-mini \
    --task_ids "$TASKS" \
    --parallel-tasks 1

echo ""
echo "Running o4-mini..."
python -m experiment_pass_at_k.run_pass_k \
    --model openai/o4-mini \
    --task_ids "$TASKS" \
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
