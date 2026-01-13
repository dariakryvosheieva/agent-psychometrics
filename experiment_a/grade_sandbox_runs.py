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
) -> Optional[Dict]:
    """Grade a single task using Lunette with structured output.

    Note: Batch grading (limit > 1 in investigate()) causes 504 Gateway Timeout
    errors, so we must grade each task individually. Each investigation takes
    approximately 55 seconds.

    Args:
        client: Lunette client
        task_id: Task instance ID
        run_id: Lunette run ID from sandbox creation
        task_metadata: SWE-bench task metadata
        output_dir: Directory to save features
        semantic_only: Use SemanticOnlyFeatures schema (faster)

    Returns:
        Feature dict or None if failed
    """
    output_file = output_dir / f"{task_id}.json"

    try:
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
        if semantic_only:
            plan = SemanticFeatureExtractionPlan(
                name="task-difficulty-features-semantic",
                prompt=grading_prompt,
            )
        else:
            plan = FeatureExtractionPlan(
                name="task-difficulty-features",
                prompt=grading_prompt,
            )

        # Run investigation
        results = await client.investigate(
            run_id=run_id,
            plan=plan,
            limit=1,
        )

        if not results.results:
            print(f"    No results returned from Lunette")
            return None

        # Get structured response
        result_data = results.results[0].data

        if isinstance(result_data, dict):
            features = result_data
        else:
            print(f"    Unexpected result type: {type(result_data)}")
            return None

        # Add metadata
        features["_instance_id"] = task_id
        features["_run_id"] = run_id
        features["_graded_at"] = datetime.now().isoformat()
        features["_schema"] = "SemanticOnlyFeatures" if semantic_only else "TaskDifficultyFeatures"

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(features, f, indent=2)

        return features

    except Exception as e:
        print(f"    Error: {e}")
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

        return None


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
    for task_id, run_info in tracking_info.items():
        run_id = run_info.get("run_id")
        if not run_id or run_id == "unknown":
            print(f"  Skipping {task_id}: no run_id")
            continue

        if task_id not in swebench_metadata:
            print(f"  Skipping {task_id}: not in SWE-bench metadata")
            continue

        tasks_to_grade.append({
            "task_id": task_id,
            "run_id": run_id,
            "metadata": swebench_metadata[task_id],
        })

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
        print(f"Batch size: {args.batch_size} trajectories per investigate() call")
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

    all_features = []

    async with LunetteClient() as client:
        for task_idx, task in enumerate(tasks_to_grade):
            task_id = task["task_id"]
            run_id = task["run_id"]

            print(f"\n[{task_idx + 1}/{len(tasks_to_grade)}] Grading: {task_id}")
            print(f"  Run ID: {run_id[:16]}...")

            features = await grade_single_task(
                client=client,
                task_id=task_id,
                run_id=run_id,
                task_metadata=task["metadata"],
                output_dir=output_dir,
                semantic_only=args.semantic_only,
            )

            if features:
                all_features.append(features)
                stats["success"] += 1
                print(f"  ✓ Success")
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
