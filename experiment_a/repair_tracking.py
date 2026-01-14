#!/usr/bin/env python3
"""Repair tracking file using Lunette API as source of truth.

This script rebuilds the tracking file by:
1. Querying Lunette API for all single-trajectory mockllm/model runs
2. Building a clean mapping: task_id -> {run_id, trajectory_id}
3. Preserving graded status from existing feature files
4. Writing a consistent tracking file

Usage:
    python -m experiment_a.repair_tracking
    python -m experiment_a.repair_tracking --dry_run
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRACKING_FILE = ROOT / "chris_output" / "experiment_a" / "sandbox_runs" / "tracking.json"
FEATURES_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_features"


async def main():
    parser = argparse.ArgumentParser(description="Repair tracking file from Lunette API")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without writing")
    args = parser.parse_args()

    from lunette import LunetteClient

    print("=" * 70)
    print("TRACKING FILE REPAIR")
    print("=" * 70)

    # 1. Load existing tracking file for reference
    print("\n[1] Loading existing tracking file...")
    old_tracking = {}
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE) as f:
            old_data = json.load(f)
        old_tracking = old_data.get("completed_tasks", {})
        print(f"    Existing tasks in tracking: {len(old_tracking)}")
    else:
        print("    No existing tracking file found")

    # 2. Check which tasks have been graded (have feature files)
    print("\n[2] Checking graded feature files...")
    graded_task_ids = set()
    for f in FEATURES_DIR.glob("*.json"):
        if f.stem not in ("grading_stats", "lunette_features") and not f.stem.endswith("_error"):
            graded_task_ids.add(f.stem)
    print(f"    Tasks with feature files: {len(graded_task_ids)}")

    # 3. Query Lunette API for all single-trajectory mockllm/model runs
    print("\n[3] Querying Lunette API for all runs...")
    async with LunetteClient() as client:
        response = await client._client.get("/runs/")
        all_runs = response.json()

        # Filter to single-trajectory mockllm/model runs
        mockllm_runs = [r for r in all_runs
                       if r.get("model") == "mockllm/model"
                       and r.get("trajectory_count") == 1]
        print(f"    Total runs in API: {len(all_runs)}")
        print(f"    Single-trajectory mockllm/model runs: {len(mockllm_runs)}")

        # 4. Build new tracking from API
        print("\n[4] Building new tracking from API...")
        new_tracking = {}
        duplicates = {}  # task_id -> list of run_ids (to detect duplicates)

        for i, run in enumerate(mockllm_runs):
            run_id = run["id"]
            created_at = run.get("created_at", "")

            try:
                detail = await client._client.get(f"/runs/{run_id}")
                if detail.status_code == 200:
                    data = detail.json()
                    for traj in data.get("trajectories", []):
                        task_id = traj.get("sample")
                        traj_id = traj.get("id")

                        if task_id:
                            # Track duplicates
                            if task_id not in duplicates:
                                duplicates[task_id] = []
                            duplicates[task_id].append({
                                "run_id": run_id,
                                "trajectory_id": traj_id,
                                "created_at": created_at,
                            })

                            # Keep the most recent run for each task
                            if task_id not in new_tracking or created_at > new_tracking[task_id].get("created_at", ""):
                                new_tracking[task_id] = {
                                    "run_id": run_id,
                                    "trajectory_id": traj_id,
                                    "model": "mockllm/model",
                                    "uploaded": True,
                                    "graded": task_id in graded_task_ids,
                                    "created_at": created_at,
                                }
                                if task_id in graded_task_ids:
                                    # Try to get graded_at from feature file
                                    feature_file = FEATURES_DIR / f"{task_id}.json"
                                    if feature_file.exists():
                                        try:
                                            with open(feature_file) as ff:
                                                features = json.load(ff)
                                            if "_graded_at" in features:
                                                new_tracking[task_id]["graded_at"] = features["_graded_at"]
                                        except Exception:
                                            pass
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(mockllm_runs)} runs...")

        print(f"    Tasks found in API: {len(new_tracking)}")

        # 5. Report duplicates
        dup_tasks = {k: v for k, v in duplicates.items() if len(v) > 1}
        if dup_tasks:
            print(f"\n[5] Found {len(dup_tasks)} tasks with multiple runs (using most recent):")
            for task_id in sorted(dup_tasks.keys())[:5]:
                runs = dup_tasks[task_id]
                print(f"    - {task_id}: {len(runs)} runs")
            if len(dup_tasks) > 5:
                print(f"    ... and {len(dup_tasks) - 5} more")

        # 6. Compare old vs new
        print("\n[6] Comparison with old tracking...")
        old_set = set(old_tracking.keys())
        new_set = set(new_tracking.keys())

        added = new_set - old_set
        removed = old_set - new_set
        updated = 0

        for task_id in old_set & new_set:
            old_run_id = old_tracking[task_id].get("run_id")
            new_run_id = new_tracking[task_id].get("run_id")
            if old_run_id != new_run_id:
                updated += 1

        print(f"    Tasks added: {len(added)}")
        print(f"    Tasks removed: {len(removed)}")
        print(f"    Tasks with updated run_id: {updated}")

        # 7. Build final tracking data
        tracking_data = {
            "completed_tasks": new_tracking,
            "failed_tasks": {},
            "stats": {
                "total_uploaded": len(new_tracking),
                "total_graded": len(graded_task_ids & new_set),
                "repaired_from_api": datetime.now().isoformat(),
            }
        }

        # 8. Write or show results
        if args.dry_run:
            print("\n[DRY RUN] Would write tracking file with:")
            print(f"    Total tasks: {len(new_tracking)}")
            print(f"    Graded tasks: {len(graded_task_ids & new_set)}")
            print(f"    File: {TRACKING_FILE}")
        else:
            print("\n[7] Writing repaired tracking file...")
            TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACKING_FILE, "w") as f:
                json.dump(tracking_data, f, indent=2)
            print(f"    Written: {TRACKING_FILE}")
            print(f"    Total tasks: {len(new_tracking)}")
            print(f"    Graded tasks: {len(graded_task_ids & new_set)}")

        # 9. Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Tasks in repaired tracking: {len(new_tracking)}")
        print(f"Tasks marked as graded: {len(graded_task_ids & new_set)}")
        print(f"Tasks remaining to grade: {len(new_set) - len(graded_task_ids & new_set)}")

        # Check for missing tasks
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        all_task_ids = set(item["instance_id"] for item in ds)
        missing = all_task_ids - new_set

        if missing:
            print(f"\nMissing from Lunette ({len(missing)} tasks):")
            for t in sorted(missing)[:5]:
                print(f"  - {t}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
            print("\nTo upload missing tasks:")
            print("  python -m experiment_a.run_dummy_sandbox --resume")


if __name__ == "__main__":
    asyncio.run(main())
