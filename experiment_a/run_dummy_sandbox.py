#!/usr/bin/env python3
"""Run dummy solver through Inspect with Lunette sandbox to create proper sandboxed runs.

This script runs the dummy_swebench task through Inspect with --sandbox lunette,
which provisions Docker containers with the actual repo checkout at the correct commit.
This is required for Lunette grading judges to have sandbox access and run shell commands.

##############################################################################
# CRITICAL: DO NOT BATCH TASKS - ONE RUN PER TASK ONLY
##############################################################################
#
# When grading with Lunette, if a run contains multiple trajectories (batched),
# the grading judge targets the WRONG trajectory. Lunette's investigate() API
# does not properly support TrajectoryFilters to target a specific trajectory
# within a batched run.
#
# Therefore: ALWAYS use batch_size=1 (the default). Each task MUST have its
# own dedicated run with exactly ONE trajectory.
#
# If you accidentally create batched runs, you must:
# 1. Delete the batched runs from Lunette API
# 2. Delete the entries from tracking.json
# 3. Re-upload each task individually with batch_size=1
#
# See lunette_utils/LUNETTE.md for more details on this issue.
##############################################################################

The script:
1. Loads SWE-bench Verified tasks
2. Runs Inspect eval ONE TASK AT A TIME with --sandbox lunette
3. Parses eval logs to extract run_ids and trajectory_ids
4. Saves tracking info for subsequent grading

Usage:
    # Dry run to see execution plan
    python -m experiment_a.run_dummy_sandbox --dry_run

    # Run on small sample for validation (ONE task per run)
    python -m experiment_a.run_dummy_sandbox --limit 20

    # Run on test split only (priority for AUC)
    python -m experiment_a.run_dummy_sandbox --test_only

    # Resume from previous run
    python -m experiment_a.run_dummy_sandbox --resume

    # Full run (one task per run, this is correct)
    python -m experiment_a.run_dummy_sandbox
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.data_loader import stable_split_tasks

# Try to import Lunette for API access
try:
    from lunette import LunetteClient
    import httpx
    HAS_LUNETTE = True
except ImportError:
    HAS_LUNETTE = False
    print("Warning: lunette-sdk not installed")

# Output directory
OUTPUT_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_runs"
TRACKING_FILE = OUTPUT_DIR / "tracking.json"


def load_swebench_verified() -> List[dict]:
    """Load SWE-bench Verified dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "version": item["version"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "hints_text": item["hints_text"],
            "base_commit": item["base_commit"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        })

    print(f"Loaded {len(tasks)} tasks")
    return tasks


def get_test_task_ids() -> set:
    """Get the set of test task IDs for the default train/test split."""
    items_path = ROOT / "clean_data" / "swebench_verified_20251120_full" / "1d_1pl" / "items.csv"
    if not items_path.exists():
        print(f"Warning: items.csv not found at {items_path}")
        return set()

    items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(items.index)
    _, test_task_ids = stable_split_tasks(all_task_ids, test_fraction=0.2, seed=42)
    return set(test_task_ids)


class TrackingManager:
    """Manage tracking of sandbox runs for resume capability."""

    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.data = self._load()

    def _load(self) -> dict:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_tasks": {},
            "failed_tasks": {},
            "stats": {
                "total": 0,
                "completed": 0,
                "failed": 0,
            }
        }

    def save(self):
        """Save tracking data to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def mark_completed(self, task_id: str, run_id: str, trajectory_id: str, eval_log: str):
        """Mark a task as successfully completed."""
        self.data["completed_tasks"][task_id] = {
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "eval_log": eval_log,
            "completed_at": datetime.now().isoformat(),
        }
        self.data["stats"]["completed"] = len(self.data["completed_tasks"])
        self.save()

    def mark_failed(self, task_id: str, error: str, attempt: int = 1):
        """Mark a task as failed."""
        self.data["failed_tasks"][task_id] = {
            "error": error,
            "attempts": attempt,
            "failed_at": datetime.now().isoformat(),
        }
        self.data["stats"]["failed"] = len(self.data["failed_tasks"])
        self.save()

    def is_completed(self, task_id: str) -> bool:
        """Check if a task is already completed."""
        return task_id in self.data["completed_tasks"]

    def get_pending_tasks(self, all_task_ids: List[str], require_unbatched: bool = True) -> List[str]:
        """Get list of task IDs that need sandbox runs.

        Returns tasks that either:
        1. Haven't been processed yet
        2. Were processed but don't have a valid run_id (e.g., from batched runs)
        3. If require_unbatched=True, also returns tasks that are in batched runs
           (multiple tasks sharing the same run_id) since these need re-creation
           for proper grading.
        """
        # First, count how many tasks share each run_id
        from collections import Counter
        run_id_counts = Counter()
        for task_id in all_task_ids:
            if task_id in self.data["completed_tasks"]:
                run_id = self.data["completed_tasks"][task_id].get("run_id")
                if run_id and run_id != "unknown":
                    run_id_counts[run_id] += 1

        pending = []
        for task_id in all_task_ids:
            if task_id not in self.data["completed_tasks"]:
                pending.append(task_id)
            else:
                run_id = self.data["completed_tasks"][task_id].get("run_id")
                # Include tasks with invalid run_ids
                if not run_id or run_id == "unknown":
                    pending.append(task_id)
                # Include tasks in batched runs if require_unbatched
                elif require_unbatched and run_id_counts[run_id] > 1:
                    pending.append(task_id)
        return pending


async def find_all_task_runs_from_api(client: LunetteClient, task_ids: List[str], max_runs: int = 500) -> Dict[str, Dict]:
    """Find run_id and trajectory_id for multiple tasks from Lunette API.

    Args:
        client: Lunette client
        task_ids: List of task IDs to find
        max_runs: Maximum number of runs to search through

    Returns:
        Dict mapping task_id -> {run_id, trajectory_id}
    """
    found = {}
    task_id_set = set(task_ids)

    try:
        async with httpx.AsyncClient(
            base_url=client.base_url,
            headers={"X-API-Key": client.api_key},
            timeout=120
        ) as http:
            # Get recent runs
            r = await http.get("/runs/")
            runs = r.json()

            # Filter to dummy_swebench runs and search
            dummy_runs = [run for run in runs if run.get("task") == "dummy_swebench"]
            print(f"    Found {len(dummy_runs)} dummy_swebench runs to search")

            for run in dummy_runs[:max_runs]:
                run_id = run.get("id")

                try:
                    # Fetch run details to get trajectories
                    r2 = await http.get(f"/runs/{run_id}")
                    run_data = r2.json()

                    trajectories = run_data.get("trajectories", [])
                    for traj in trajectories:
                        sample_id = traj.get("sample")
                        if sample_id in task_id_set and sample_id not in found:
                            found[sample_id] = {
                                "run_id": run_id,
                                "trajectory_id": traj.get("id"),
                            }

                except Exception as e:
                    # Skip individual run errors
                    continue

                # Stop if we've found all tasks
                if len(found) == len(task_ids):
                    break

            print(f"    Found run info for {len(found)}/{len(task_ids)} tasks")

    except Exception as e:
        print(f"    Error querying Lunette API: {e}")

    return found


def parse_eval_log(log_path: str) -> Dict:
    """Parse Inspect eval log to extract sample info."""
    try:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(log_path)

        result = {
            "eval_id": log.eval.run_id if log.eval else None,
            "samples": {},
        }

        if log.samples:
            for sample in log.samples:
                sample_id = sample.id
                result["samples"][sample_id] = {
                    "id": sample_id,
                    "has_messages": len(sample.messages) > 0 if sample.messages else False,
                }

        return result

    except Exception as e:
        print(f"    Warning: Could not parse log {log_path}: {e}")
        return {}


def run_inspect_batch(task_ids: List[str], output_dir: Path, timeout: int = 600) -> Dict:
    """Run Inspect eval on task(s).

    ##########################################################################
    # CRITICAL: ALWAYS PASS EXACTLY ONE TASK (batch_size=1)
    ##########################################################################
    #
    # Lunette grading DOES NOT WORK with batched runs (multiple trajectories
    # per run). The investigate() API targets the wrong trajectory when a run
    # has multiple trajectories, even with TrajectoryFilters.
    #
    # This function accepts a list for API compatibility, but you MUST only
    # pass a single task_id. The default batch_size=1 ensures this.
    ##########################################################################

    Returns dict with log_path, run info, and any errors.
    """
    # SAFETY CHECK: Warn loudly if someone tries to batch
    if len(task_ids) > 1:
        print("=" * 70)
        print("WARNING: BATCHING MULTIPLE TASKS IS NOT SUPPORTED FOR GRADING!")
        print(f"You passed {len(task_ids)} tasks. Lunette grading will FAIL.")
        print("Use batch_size=1 (the default) to create one run per task.")
        print("=" * 70)
    if not task_ids:
        return {"error": "No tasks provided"}

    # Build command
    cmd = [
        "inspect", "eval",
        "lunette_utils/dummy_swebench_task.py@dummy_swebench",
        "--model", "mockllm/model",
        "--sandbox", "lunette",
        "--no-score",
        "--sample-id", ",".join(task_ids),  # Comma-separated list
    ]

    print(f"    Running: {' '.join(cmd[:8])} ... ({len(task_ids)} tasks)")

    result = {
        "task_ids": task_ids,
        "log_path": None,
        "error": None,
        "stdout": "",
        "stderr": "",
    }

    try:
        start_time = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )

        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["elapsed_time"] = time.time() - start_time

        # Find log path in output
        for line in proc.stdout.split("\n"):
            if "Log:" in line:
                result["log_path"] = line.split("Log:")[-1].strip()
                break

        if proc.returncode != 0:
            result["error"] = f"Exit code {proc.returncode}: {proc.stderr[:500]}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Timed out after {timeout}s"
    except Exception as e:
        result["error"] = str(e)

    return result


async def run_sandbox_pipeline(
    tasks: List[dict],
    tracker: TrackingManager,
    batch_size: int = 10,
    timeout: int = 600,
) -> Dict:
    """Run the full sandbox creation pipeline.

    Returns stats dict.
    """
    stats = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }

    # Process in batches
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        batch_task_ids = [t["instance_id"] for t in batch]
        batch_end = min(batch_start + batch_size, len(tasks))

        print(f"\n{'='*60}")
        print(f"BATCH {batch_start // batch_size + 1}: Tasks {batch_start + 1}-{batch_end}")
        print("=" * 60)

        # Run Inspect eval on batch
        result = run_inspect_batch(batch_task_ids, OUTPUT_DIR, timeout=timeout)

        if result["error"]:
            print(f"  Batch error: {result['error']}")
            for task_id in batch_task_ids:
                tracker.mark_failed(task_id, result["error"])
                stats["failed"] += 1
            continue

        print(f"  Completed in {result.get('elapsed_time', 0):.1f}s")
        print(f"  Log: {result.get('log_path', 'N/A')}")

        # Parse log to verify samples
        if result["log_path"]:
            log_info = parse_eval_log(result["log_path"])
            print(f"  Samples in log: {len(log_info.get('samples', {}))}")

        # Query Lunette API to get run_ids and trajectory_ids for all tasks in batch
        print("  Querying Lunette API for run info...")
        async with LunetteClient() as client:
            # Find all tasks in batch at once (more efficient)
            all_run_info = await find_all_task_runs_from_api(client, batch_task_ids)

            for task_id in batch_task_ids:
                run_info = all_run_info.get(task_id)

                if run_info:
                    tracker.mark_completed(
                        task_id=task_id,
                        run_id=run_info["run_id"],
                        trajectory_id=run_info["trajectory_id"],
                        eval_log=result.get("log_path", ""),
                    )
                    stats["success"] += 1
                    print(f"    {task_id}: run_id={run_info['run_id'][:8]}...")
                else:
                    # Task completed but couldn't find in API - mark as completed anyway
                    tracker.mark_completed(
                        task_id=task_id,
                        run_id="unknown",
                        trajectory_id="unknown",
                        eval_log=result.get("log_path", ""),
                    )
                    stats["success"] += 1
                    print(f"    {task_id}: Completed (run_id lookup pending)")

        # Save progress after each batch
        tracker.save()
        print(f"\n  Progress: {stats['success']} completed, {stats['failed']} failed")

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Run dummy solver through Inspect with Lunette sandbox"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show execution plan without running"
    )
    # CRITICAL: batch_size MUST be 1 for Lunette grading to work correctly.
    # Batched runs (multiple trajectories per run) cause grading to target
    # the wrong trajectory. DO NOT CHANGE THIS DEFAULT.
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="MUST BE 1. Batching breaks Lunette grading. (default: 1)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit total number of tasks to process"
    )
    parser.add_argument(
        "--task_ids", type=str, default=None,
        help="Comma-separated list of specific task IDs to process"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run (skip completed tasks)"
    )
    parser.add_argument(
        "--test_only", action="store_true",
        help="Only process test split tasks (priority for AUC)"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Timeout in seconds per batch (default: 600)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for tracking files"
    )
    args = parser.parse_args()

    # CRITICAL: Enforce batch_size=1
    if args.batch_size != 1:
        print("=" * 70)
        print("ERROR: batch_size MUST be 1!")
        print("")
        print("Lunette grading DOES NOT WORK with batched runs. When a run has")
        print("multiple trajectories, the investigate() API targets the wrong one.")
        print("")
        print("Each task MUST have its own run with exactly ONE trajectory.")
        print("Remove --batch_size or set it to 1.")
        print("=" * 70)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_file = output_dir / "tracking.json"

    print("=" * 60)
    print("EXPERIMENT A: Run Dummy Sandbox for Lunette Feature Extraction")
    print("=" * 60)

    # Load tasks
    tasks = load_swebench_verified()

    # Filter to specific task IDs if provided
    if args.task_ids:
        task_id_set = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t["instance_id"] in task_id_set]
        print(f"Filtered to {len(tasks)} specified tasks")

    # Filter to test split if requested
    if args.test_only:
        test_ids = get_test_task_ids()
        if test_ids:
            tasks = [t for t in tasks if t["instance_id"] in test_ids]
            print(f"Filtered to {len(tasks)} test split tasks")
        else:
            print("Warning: Could not load test split, processing all tasks")

    # Initialize tracker
    tracker = TrackingManager(tracking_file)

    # Filter to pending tasks if resuming
    if args.resume:
        pending_ids = tracker.get_pending_tasks([t["instance_id"] for t in tasks])
        tasks = [t for t in tasks if t["instance_id"] in pending_ids]
        print(f"Resuming: {len(tasks)} tasks remaining")

    # Apply limit
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limited to {args.limit} tasks")

    # Update total count
    tracker.data["stats"]["total"] = len(tasks)
    tracker.save()

    print(f"\nTasks to process: {len(tasks)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nSample tasks (first 10):")
        for task in tasks[:10]:
            print(f"  - {task['instance_id']} ({task['repo']})")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")

        n_batches = (len(tasks) + args.batch_size - 1) // args.batch_size
        print(f"\nEstimated batches: {n_batches}")
        print(f"Estimated time: ~{n_batches * 2} minutes (assuming ~2 min/batch)")
        return

    if not HAS_LUNETTE:
        print("\nError: lunette-sdk not installed")
        print("Run: pip install lunette-sdk")
        return

    # Run pipeline
    print("\nStarting sandbox creation pipeline...")
    stats = await run_sandbox_pipeline(
        tasks=tasks,
        tracker=tracker,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"\nTracking file: {tracking_file}")
    print(f"\nNext step: Run grade_sandbox_runs.py to extract features")


if __name__ == "__main__":
    asyncio.run(main())
