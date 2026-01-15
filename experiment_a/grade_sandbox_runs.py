#!/usr/bin/env python3
"""Grade sandbox runs using Lunette with structured output to extract features.

This script grades the sandbox runs created by run_dummy_sandbox.py using
Lunette's investigate() API with structured output schemas. The grading judge
has full sandbox access and can run shell commands to extract accurate
environment-based features.

IMPORTANT: Batch grading does NOT work. When using limit > 1 in investigate(),
the Lunette API returns 504 Gateway Timeout errors. Each investigation takes
~55 seconds, and the API cannot handle multiple simultaneous investigations.
Therefore, this script grades tasks one at a time.

Usage:
    # Dry run to see execution plan
    python -m experiment_a.grade_sandbox_runs --dry_run

    # Grade specific tasks
    python -m experiment_a.grade_sandbox_runs --limit 20

    # Skip already graded tasks
    python -m experiment_a.grade_sandbox_runs --skip_existing

    # Full grading
    python -m experiment_a.grade_sandbox_runs
"""

import argparse
import asyncio
import json
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

# Try to import Lunette
try:
    from lunette import LunetteClient

    # Import our structured output support
    from experiment_a.lunette_structured_output import (
        FeatureExtractionPlan,
        SemanticFeatureExtractionPlan,
        TaskDifficultyFeatures,
        SemanticOnlyFeatures,
        format_feature_extraction_prompt,
    )

    HAS_LUNETTE = True
except ImportError as e:
    HAS_LUNETTE = False
    print(f"Warning: lunette-sdk not installed or import error: {e}")

# Directories
SANDBOX_RUNS_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_runs"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_features"
TRACKING_FILE = SANDBOX_RUNS_DIR / "tracking.json"


def load_swebench_metadata() -> Dict[str, dict]:
    """Load SWE-bench Verified metadata for all tasks."""
    from datasets import load_dataset

    print("Loading SWE-bench Verified metadata...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    metadata = {}
    for item in ds:
        metadata[item["instance_id"]] = {
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
        }

    print(f"Loaded metadata for {len(metadata)} tasks")
    return metadata


def load_tracking_info(tracking_file: Path) -> Dict[str, dict]:
    """Load tracking info from run_dummy_sandbox.py output."""
    if not tracking_file.exists():
        print(f"Error: Tracking file not found: {tracking_file}")
        print("Run run_dummy_sandbox.py first to create sandbox runs.")
        return {}

    with open(tracking_file) as f:
        data = json.load(f)

    return data.get("completed_tasks", {})


async def grade_single_task(
    client: LunetteClient,
    task_id: str,
    run_id: str,
    task_metadata: dict,
    output_dir: Path,
    semantic_only: bool = False,
    max_retries: int = 3,
    retry_delay: int = 60,
) -> tuple[Optional[Dict], dict]:
    """Grade a single task using Lunette with structured output.

    Note: Batch grading (limit > 1 in investigate()) causes 504 Gateway Timeout
    errors, so we must grade each task individually. Each investigation takes
    approximately 55 seconds.

    Includes retry logic for 504 Gateway Timeout errors.

    Args:
        client: Lunette client
        task_id: Task instance ID
        run_id: Lunette run ID from sandbox creation
        task_metadata: SWE-bench task metadata
        output_dir: Directory to save features
        semantic_only: Use SemanticOnlyFeatures schema (faster)
        max_retries: Maximum number of retries on 504 timeout
        retry_delay: Seconds to wait between retries

    Returns:
        Tuple of (feature dict or None, timing dict)
    """
    timing = {"task_id": task_id, "attempts": 0, "retry_wait_s": 0.0, "api_call_s": 0.0}
    task_start = time.perf_counter()
    output_file = output_dir / f"{task_id}.json"

    # Format grading prompt with task info
    grading_prompt = format_feature_extraction_prompt(
        instance_id=task_id,
        repo=task_metadata["repo"],
        version=task_metadata.get("version", "unknown"),
        problem_statement=task_metadata["problem_statement"],
        patch=task_metadata["patch"],
        fail_to_pass=task_metadata.get("FAIL_TO_PASS", "[]"),
        pass_to_pass=task_metadata.get("PASS_TO_PASS", "[]"),
        hints_text=task_metadata.get("hints_text", ""),
    )

    # Create plan with structured output schema
    # Use trajectory_filters to target the specific task within a batched run
    from lunette.analysis import TrajectoryFilters

    trajectory_filter = TrajectoryFilters(sample=task_id)

    if semantic_only:
        plan = SemanticFeatureExtractionPlan(
            name="task-difficulty-features-semantic",
            prompt=grading_prompt,
            trajectory_filters=trajectory_filter,
        )
    else:
        plan = FeatureExtractionPlan(
            name="task-difficulty-features",
            prompt=grading_prompt,
            trajectory_filters=trajectory_filter,
        )

    # Retry loop for 504 timeouts
    last_error = None
    timeout_log_file = output_dir / "504_timeout_log.jsonl"

    for attempt in range(max_retries + 1):
        timing["attempts"] = attempt + 1
        try:
            # Run investigation
            api_start = time.perf_counter()
            results = await client.investigate(
                run_id=run_id,
                plan=plan,
                limit=1,
            )
            timing["api_call_s"] += time.perf_counter() - api_start
            # Success - break out of retry loop
            break
        except Exception as e:
            timing["api_call_s"] += time.perf_counter() - api_start
            last_error = e
            error_str = str(e)
            # Check if it's a 504 timeout
            if "504" in error_str or "Gateway Time-out" in error_str:
                # Log the 504 timeout for troubleshooting
                timeout_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "task_id": task_id,
                    "run_id": run_id,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": error_str,
                    "plan_name": plan.name,
                    "semantic_only": semantic_only,
                }
                with open(timeout_log_file, "a") as log_f:
                    log_f.write(json.dumps(timeout_entry) + "\n")

                if attempt < max_retries:
                    print(f"    504 timeout at {datetime.now().strftime('%H:%M:%S')}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                    timing["retry_wait_s"] += retry_delay
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"    504 timeout after {max_retries} retries, giving up")
            elif "404" in error_str or "Not Found" in error_str:
                # Run doesn't exist in Lunette - skip this task
                print(f"    404 Not Found - run doesn't exist in Lunette, skipping")
                error_file = output_dir / f"{task_id}_error.json"
                with open(error_file, "w") as f:
                    json.dump({
                        "error": "404 Not Found - run doesn't exist",
                        "task_id": task_id,
                        "run_id": run_id,
                        "graded_at": datetime.now().isoformat(),
                    }, f, indent=2)
                timing["total_s"] = time.perf_counter() - task_start
                return None, timing
            else:
                # Non-timeout, non-404 error, don't retry
                raise
    else:
        # All retries exhausted
        if last_error:
            # Save error and return None
            print(f"    Error after retries: {last_error}")
            error_file = output_dir / f"{task_id}_error.json"
            with open(error_file, "w") as f:
                json.dump({
                    "error": str(last_error),
                    "task_id": task_id,
                    "run_id": run_id,
                    "graded_at": datetime.now().isoformat(),
                }, f, indent=2)
            timing["total_s"] = time.perf_counter() - task_start
            return None, timing

    # Process results
    try:
        if not results.results:
            print(f"    No results returned from Lunette")
            timing["total_s"] = time.perf_counter() - task_start
            return None, timing

        # Get structured response
        result_data = results.results[0].data

        if isinstance(result_data, dict):
            features = result_data
        else:
            print(f"    Unexpected result type: {type(result_data)}")
            timing["total_s"] = time.perf_counter() - task_start
            return None, timing

        # Add metadata
        features["_instance_id"] = task_id
        features["_run_id"] = run_id
        features["_graded_at"] = datetime.now().isoformat()
        features["_schema"] = "SemanticOnlyFeatures" if semantic_only else "TaskDifficultyFeatures"

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(features, f, indent=2)

        timing["total_s"] = time.perf_counter() - task_start
        return features, timing

    except Exception as e:
        print(f"    Error processing results: {e}")
        import traceback
        traceback.print_exc()

        # Save error
        error_file = output_dir / f"{task_id}_error.json"
        with open(error_file, "w") as f:
            json.dump({
                "error": str(e),
                "task_id": task_id,
                "run_id": run_id,
                "graded_at": datetime.now().isoformat(),
            }, f, indent=2)

        timing["total_s"] = time.perf_counter() - task_start
        return None, timing


def aggregate_to_csv(output_dir: Path) -> Optional[Path]:
    """Aggregate all feature JSON files to a single CSV."""
    feature_files = list(output_dir.glob("*.json"))
    feature_files = [
        f for f in feature_files
        if not f.name.endswith("_error.json")
        and f.name != "grading_stats.json"
    ]

    if not feature_files:
        print("No feature files to aggregate")
        return None

    rows = []
    for f in feature_files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            task_id = data.get("_instance_id", f.stem)
            row = {"task_id": task_id}
            row.update(data)
            rows.append(row)
        except Exception as e:
            print(f"  Warning: Could not load {f.name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / "lunette_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nAggregated {len(rows)} features to {csv_path}")
        return csv_path

    return None


async def main():
    parser = argparse.ArgumentParser(
        description="Grade sandbox runs using Lunette with structured output"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show execution plan without running"
    )
    parser.add_argument(
        "--tracking_file", type=str, default=str(TRACKING_FILE),
        help="Path to tracking JSON from run_dummy_sandbox.py"
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for features"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tasks to grade"
    )
    parser.add_argument(
        "--task_ids", type=str, default=None,
        help="Comma-separated list of specific task IDs to grade"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip tasks with existing feature files"
    )
    parser.add_argument(
        "--semantic_only", action="store_true",
        help="Use SemanticOnlyFeatures schema (faster, no shell exploration)"
    )
    args = parser.parse_args()

    tracking_file = Path(args.tracking_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT A: Grade Sandbox Runs for Feature Extraction")
    print("=" * 60)

    # Load tracking info from sandbox runs
    print(f"\nLoading tracking info from: {tracking_file}")
    tracking_info = load_tracking_info(tracking_file)

    if not tracking_info:
        print("No completed sandbox runs found. Run run_dummy_sandbox.py first.")
        return

    print(f"Found {len(tracking_info)} completed sandbox runs")

    # Load SWE-bench metadata
    swebench_metadata = load_swebench_metadata()

    # Build list of tasks to grade
    tasks_to_grade = []
    skipped_no_run_id = 0
    skipped_no_metadata = 0
    for task_id, run_info in tracking_info.items():
        run_id = run_info.get("run_id")
        trajectory_id = run_info.get("trajectory_id", "unknown")
        if not run_id or run_id == "unknown":
            skipped_no_run_id += 1
            continue

        if task_id not in swebench_metadata:
            skipped_no_metadata += 1
            continue

        tasks_to_grade.append({
            "task_id": task_id,
            "run_id": run_id,
            "trajectory_id": trajectory_id,
            "metadata": swebench_metadata[task_id],
        })

    if skipped_no_run_id > 0:
        print(f"  Skipped {skipped_no_run_id} tasks with no run_id")
    if skipped_no_metadata > 0:
        print(f"  Skipped {skipped_no_metadata} tasks not in SWE-bench metadata")

    # Filter to specific task IDs if provided
    if args.task_ids:
        task_id_set = set(args.task_ids.split(","))
        tasks_to_grade = [t for t in tasks_to_grade if t["task_id"] in task_id_set]
        print(f"Filtered to {len(tasks_to_grade)} specified tasks")

    # Skip existing if requested
    if args.skip_existing:
        original_count = len(tasks_to_grade)
        tasks_to_grade = [
            t for t in tasks_to_grade
            if not (output_dir / f"{t['task_id']}.json").exists()
        ]
        skipped = original_count - len(tasks_to_grade)
        if skipped > 0:
            print(f"Skipping {skipped} tasks with existing features")

    # Apply limit
    if args.limit:
        tasks_to_grade = tasks_to_grade[:args.limit]
        print(f"Limited to {args.limit} tasks")

    print(f"\nTasks to grade: {len(tasks_to_grade)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nOutput directory: {output_dir}")
        print(f"Schema: {'SemanticOnlyFeatures' if args.semantic_only else 'TaskDifficultyFeatures'}")
        print(f"\nSample tasks (first 10):")
        for task in tasks_to_grade[:10]:
            print(f"  - {task['task_id']} (run_id: {task['run_id'][:8]}...)")
        if len(tasks_to_grade) > 10:
            print(f"  ... and {len(tasks_to_grade) - 10} more")

        # Estimate cost
        cost_per_task = 0.15
        print(f"\nEstimated cost: ~${len(tasks_to_grade) * cost_per_task:.2f}")
        print(f"  ({len(tasks_to_grade)} tasks × ${cost_per_task}/task)")
        return

    if not HAS_LUNETTE:
        print("\nError: lunette-sdk not installed")
        print("Run: pip install lunette-sdk")
        return

    # Grade tasks one at a time
    # NOTE: Batch grading (limit > 1 in investigate()) causes 504 Gateway Timeout errors.
    # The Lunette API takes ~55 seconds per single investigation, and batching causes timeouts.
    # Therefore we must grade each task individually.
    stats = {
        "total": len(tasks_to_grade),
        "success": 0,
        "failed": 0,
    }

    # Grading log file for detailed tracking
    grading_log_file = output_dir / "grading_log.jsonl"
    timing_log_file = output_dir / "timing_log.jsonl"

    async with LunetteClient() as client:
        for task_idx, task in enumerate(tasks_to_grade):
            task_id = task["task_id"]
            run_id = task["run_id"]
            trajectory_id = task.get("trajectory_id", "unknown")

            print(f"\n[{task_idx + 1}/{len(tasks_to_grade)}] Grading: {task_id}")
            print(f"  Run ID: {run_id[:16]}...")
            print(f"  Trajectory ID: {trajectory_id[:16]}...")

            features, timing = await grade_single_task(
                client=client,
                task_id=task_id,
                run_id=run_id,
                task_metadata=task["metadata"],
                output_dir=output_dir,
                semantic_only=args.semantic_only,
            )

            # Log timing data
            timing["timestamp"] = datetime.now().isoformat()
            timing["success"] = features is not None
            with open(timing_log_file, "a") as tf:
                tf.write(json.dumps(timing) + "\n")

            # Print timing summary
            api_pct = (timing["api_call_s"] / timing["total_s"] * 100) if timing["total_s"] > 0 else 0
            retry_pct = (timing["retry_wait_s"] / timing["total_s"] * 100) if timing["total_s"] > 0 else 0
            print(f"  Time: {timing['total_s']:.1f}s (API: {timing['api_call_s']:.1f}s/{api_pct:.0f}%, retry wait: {timing['retry_wait_s']:.0f}s/{retry_pct:.0f}%, attempts: {timing['attempts']})")

            # Log grading result
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id,
                "run_id": run_id,
                "trajectory_id": trajectory_id,
                "success": features is not None,
            }
            with open(grading_log_file, "a") as log_f:
                log_f.write(json.dumps(log_entry) + "\n")

            if features:
                stats["success"] += 1
                print(f"  ✓ Success")

                # Update tracking file to mark as graded
                tracking_info[task_id]["graded"] = True
                tracking_info[task_id]["graded_at"] = datetime.now().isoformat()
                with open(tracking_file, "w") as f:
                    json.dump({
                        "completed_tasks": tracking_info,
                        "failed_tasks": {},
                        "stats": {
                            "total_uploaded": len(tracking_info),
                            "total_graded": stats["success"],
                        }
                    }, f, indent=2)
            else:
                stats["failed"] += 1
                print(f"  ✗ Failed")

    # Aggregate to CSV
    csv_path = aggregate_to_csv(output_dir)

    # Save stats
    stats_file = output_dir / "grading_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "stats": stats,
            "graded_at": datetime.now().isoformat(),
            "csv_path": str(csv_path) if csv_path else None,
            "semantic_only": args.semantic_only,
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nFeatures CSV: {csv_path}")
    print(f"\nNext step: Run train_evaluate.py with:")
    print(f"  python -m experiment_a.train_evaluate \\")
    print(f"      --lunette_features_path {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
