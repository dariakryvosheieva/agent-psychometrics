"""
Experiment C: Lunette grading cost measurement.

This script:
1. Runs a dummy solver on N SWE-bench tasks
2. Gets the run_id from Lunette for each
3. Runs Lunette grading (via investigate API with GradingPlan)
4. Reports timing and run_ids for server-side cost lookup

Usage:
    # Single task test
    python llm_judge/experiment_c_single_task.py --n_tasks 1

    # Full 10-task experiment
    python llm_judge/experiment_c_single_task.py --n_tasks 10
"""

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd

# Output directory
OUTPUT_DIR = Path("chris_output/experiment_c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IRT difficulty data for stratified sampling
ITEMS_PATH = Path("clean_data/swebench_verified_20250930_full/1d/items.csv")


def get_lunette_api_key() -> str:
    """Get Lunette API key from config file."""
    config_path = Path.home() / ".lunette" / "config.json"
    with open(config_path) as f:
        return json.load(f)["api_key"]


def get_recent_runs(api_key: str, limit: int = 10) -> list[dict]:
    """Get recent runs from Lunette API."""
    with httpx.Client(
        base_url="https://lunette.dev/api", headers={"X-API-Key": api_key}
    ) as client:
        r = client.get("/runs/")
        return r.json()[:limit]


def select_tasks(n_tasks: int, seed: int = 42) -> list[str]:
    """Select tasks stratified by IRT difficulty."""
    import numpy as np

    np.random.seed(seed)

    items = pd.read_csv(ITEMS_PATH, index_col=0)

    # Stratify by difficulty quintiles
    items["quintile"] = pd.qcut(items["b"], q=5, labels=False)

    selected = []
    tasks_per_quintile = n_tasks // 5
    remainder = n_tasks % 5

    for q in range(5):
        quintile_tasks = items[items["quintile"] == q].index.tolist()
        n_select = tasks_per_quintile + (1 if q < remainder else 0)
        n_select = min(n_select, len(quintile_tasks))
        selected.extend(np.random.choice(quintile_tasks, n_select, replace=False))

    return selected[:n_tasks]


def run_dummy_eval(task_id: str) -> tuple[float, str | None]:
    """
    Run dummy eval on a single task.

    Returns:
        (elapsed_time, run_id or None if failed)
    """
    print(f"\n--- Running dummy eval on {task_id} ---")

    # Get runs before to detect new one
    api_key = get_lunette_api_key()
    runs_before = {r["id"] for r in get_recent_runs(api_key, limit=50)}

    start_time = time.time()

    # Run the dummy eval
    cmd = [
        "inspect",
        "eval",
        "lunette_utils/dummy_swebench_task.py@dummy_swebench",
        "--model",
        "mockllm/model",
        "--sandbox",
        "lunette",
        "--limit",
        "1",
        "-T",
        f"instance_id={task_id}",
        "--no-score",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"  Eval failed: {result.stderr[:200]}")
        return elapsed, None

    print(f"  Eval completed in {elapsed:.1f}s")

    # Wait a bit for Lunette to register the run
    time.sleep(2)

    # Find the new run
    runs_after = get_recent_runs(api_key, limit=50)
    new_runs = [r for r in runs_after if r["id"] not in runs_before]

    if not new_runs:
        # Look for mockllm runs as fallback
        mockllm_runs = [
            r for r in runs_after if "mockllm" in r.get("model", "").lower()
        ]
        if mockllm_runs:
            run_id = mockllm_runs[0]["id"]
            print(f"  Found mockllm run: {run_id[:8]}...")
            return elapsed, run_id
        print("  Warning: No new run found")
        return elapsed, None

    run_id = new_runs[0]["id"]
    print(f"  Run ID: {run_id[:8]}...")
    return elapsed, run_id


async def run_lunette_grading(run_id: str) -> tuple[float, dict]:
    """
    Run Lunette grading on a run.

    Returns:
        (elapsed_time, results_dict)
    """
    from lunette import LunetteClient
    from lunette.analysis import GradingPlan

    print(f"--- Running Lunette grading on {run_id[:8]}... ---")

    plan = GradingPlan(
        name="task-difficulty",
        prompt="""You are analyzing a SWE-bench coding task to understand what makes it difficult.

## Your Goals
Evaluate the difficulty of this task by exploring the codebase and problem statement.
You have access to the sandbox environment where the task would be solved.

## Environment Access
You can:
- Browse the repository structure
- Read files to understand the codebase
- Examine the problem statement and any hints

## Evaluation Criteria

Consider these factors when assessing difficulty:

1. **Problem clarity**: Is the issue well-specified? Are there reproduction steps?
2. **Fix hints**: Does the problem description hint at the solution?
3. **Fix locality**: Is it a single-file fix or multi-file?
4. **Domain knowledge**: How much specialized knowledge is needed?
5. **Codebase complexity**: How large/complex is the relevant codebase?

## Instructions

1. First, explore the repository structure
2. Read the problem statement carefully
3. Try to locate relevant files and understand the scope
4. Assign a difficulty score from 0.0 (trivial) to 1.0 (extremely hard)

Respond with your assessment of the overall task difficulty.
""",
        enable_sandbox=True,
    )

    start_time = time.time()

    async with LunetteClient() as client:
        results = await client.investigate(
            run_id=run_id,
            plan=plan,
            limit=1,  # Only grade ONE trajectory
        )

    elapsed = time.time() - start_time
    print(f"  Grading completed in {elapsed:.1f}s")

    return elapsed, results


def main():
    """Run the experiment."""
    parser = argparse.ArgumentParser(description="Experiment C: Lunette grading cost")
    parser.add_argument("--n_tasks", type=int, default=1, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Experiment C: Lunette Grading Cost ({args.n_tasks} tasks)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Select tasks
    tasks = select_tasks(args.n_tasks, args.seed)
    print(f"\nSelected tasks: {tasks}")

    # Save task selection
    task_file = OUTPUT_DIR / "selected_tasks.json"
    with open(task_file, "w") as f:
        json.dump(
            {"tasks": tasks, "seed": args.seed, "timestamp": datetime.now().isoformat()},
            f,
            indent=2,
        )

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_tasks": args.n_tasks,
        "seed": args.seed,
        "tasks": [],
    }

    # Process each task sequentially
    for i, task_id in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/{len(tasks)}: {task_id}")
        print("=" * 60)

        task_result = {"task_id": task_id}

        # Phase 1: Run dummy eval
        eval_time, run_id = run_dummy_eval(task_id)
        task_result["dummy_eval_seconds"] = eval_time
        task_result["run_id"] = run_id

        if run_id:
            # Phase 2: Run Lunette grading
            grading_time, grading_results = asyncio.run(run_lunette_grading(run_id))
            task_result["grading_seconds"] = grading_time

            # Extract score if available
            try:
                result_str = str(grading_results)
                if "score" in result_str:
                    import re

                    score_match = re.search(r"'score':\s*([\d.]+)", result_str)
                    if score_match:
                        task_result["lunette_score"] = float(score_match.group(1))
            except Exception:
                pass
        else:
            task_result["grading_seconds"] = None
            task_result["error"] = "No run_id from dummy eval"

        results["tasks"].append(task_result)

        # Save intermediate results
        output_file = OUTPUT_DIR / f"lunette_grading_{args.n_tasks}tasks.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [t for t in results["tasks"] if t.get("run_id")]
    total_eval_time = sum(t["dummy_eval_seconds"] for t in results["tasks"])
    total_grading_time = sum(t.get("grading_seconds", 0) or 0 for t in successful)

    print(f"Tasks attempted: {len(results['tasks'])}")
    print(f"Tasks successful: {len(successful)}")
    print(f"Total dummy eval time: {total_eval_time:.1f}s")
    print(f"Total grading time: {total_grading_time:.1f}s")
    print(f"Total time: {total_eval_time + total_grading_time:.1f}s")
    print(f"\nAvg per task: {(total_eval_time + total_grading_time) / len(successful):.1f}s")

    print("\nRun IDs for server-side cost lookup:")
    for t in successful:
        print(f"  {t['task_id']}: {t['run_id']}")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
