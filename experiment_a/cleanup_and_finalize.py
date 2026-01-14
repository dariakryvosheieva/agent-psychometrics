#!/usr/bin/env python3
"""Clean up Lunette state to have exactly 500 tasks with one run each.

This script:
1. Uploads the 2 missing tasks (sympy__sympy-21930, sympy__sympy-24661)
2. Queries all runs from Lunette API
3. For each task, picks ONE run (preferring graded ones)
4. Deletes all duplicate runs
5. Verifies graded status through API
6. Writes clean tracking file with exactly 500 tasks

Usage:
    python -m experiment_a.cleanup_and_finalize --dry_run
    python -m experiment_a.cleanup_and_finalize
"""

import asyncio
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRACKING_FILE = ROOT / "chris_output" / "experiment_a" / "sandbox_runs" / "tracking.json"
FEATURES_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_features"
OUTPUT_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_runs"


async def upload_missing_tasks(missing_task_ids: list, dry_run: bool = False) -> dict:
    """Upload missing tasks one at a time using Inspect."""
    results = {}

    for task_id in missing_task_ids:
        print(f"\n  Uploading {task_id}...")

        if dry_run:
            print(f"    [DRY RUN] Would run Inspect for {task_id}")
            results[task_id] = {"status": "dry_run"}
            continue

        cmd = [
            "inspect", "eval",
            "lunette_utils/dummy_swebench_task.py@dummy_swebench",
            "--model", "mockllm/model",
            "--sandbox", "lunette",
            f"--sample-id={task_id}",
            "--log-dir", str(OUTPUT_DIR / "logs"),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=ROOT,
            )

            if result.returncode == 0:
                print(f"    ✓ Uploaded successfully")
                results[task_id] = {"status": "success"}
            else:
                print(f"    ✗ Failed: {result.stderr[:200]}")
                results[task_id] = {"status": "failed", "error": result.stderr[:500]}
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[task_id] = {"status": "error", "error": str(e)}

    return results


async def main():
    parser = argparse.ArgumentParser(description="Clean up Lunette to have exactly 500 tasks")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without executing")
    args = parser.parse_args()

    from lunette import LunetteClient
    from datasets import load_dataset

    print("=" * 70)
    print("LUNETTE CLEANUP AND FINALIZE")
    print("=" * 70)

    # 1. Load all SWE-bench task IDs
    print("\n[1] Loading SWE-bench Verified tasks...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    all_task_ids = set(item["instance_id"] for item in ds)
    print(f"    Total SWE-bench Verified tasks: {len(all_task_ids)}")

    # 2. Check which tasks have been graded (have feature files)
    print("\n[2] Checking graded feature files...")
    graded_task_ids = set()
    for f in FEATURES_DIR.glob("*.json"):
        if f.stem not in ("grading_stats", "lunette_features") and not f.stem.endswith("_error"):
            graded_task_ids.add(f.stem)
    print(f"    Tasks with feature files: {len(graded_task_ids)}")

    # 3. Query Lunette API for all runs
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

        # 4. Build mapping: task_id -> list of runs
        print("\n[4] Building task -> runs mapping...")
        task_to_runs = defaultdict(list)

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
                            # Check if this run has been investigated (graded)
                            is_graded = task_id in graded_task_ids

                            task_to_runs[task_id].append({
                                "run_id": run_id,
                                "trajectory_id": traj_id,
                                "created_at": created_at,
                                "is_graded": is_graded,
                            })
            except Exception as e:
                print(f"    Warning: Could not fetch run {run_id}: {e}")

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(mockllm_runs)} runs...")

        print(f"    Unique tasks found: {len(task_to_runs)}")

        # 5. Find missing tasks
        api_task_ids = set(task_to_runs.keys())
        missing_tasks = all_task_ids - api_task_ids
        print(f"\n[5] Missing tasks: {len(missing_tasks)}")
        for t in sorted(missing_tasks):
            print(f"    - {t}")

        # 6. Upload missing tasks
        if missing_tasks:
            print("\n[6] Uploading missing tasks...")
            upload_results = await upload_missing_tasks(list(missing_tasks), dry_run=args.dry_run)

            # If not dry run, re-fetch the uploaded runs
            if not args.dry_run:
                print("    Re-fetching uploaded runs from API...")
                await asyncio.sleep(2)  # Brief pause

                for task_id in missing_tasks:
                    if upload_results.get(task_id, {}).get("status") == "success":
                        # Find the new run
                        response = await client._client.get("/runs/")
                        new_runs = response.json()
                        for run in new_runs:
                            if run.get("model") == "mockllm/model" and run.get("trajectory_count") == 1:
                                try:
                                    detail = await client._client.get(f"/runs/{run['id']}")
                                    if detail.status_code == 200:
                                        data = detail.json()
                                        for traj in data.get("trajectories", []):
                                            if traj.get("sample") == task_id:
                                                task_to_runs[task_id].append({
                                                    "run_id": run["id"],
                                                    "trajectory_id": traj.get("id"),
                                                    "created_at": run.get("created_at", ""),
                                                    "is_graded": False,
                                                })
                                                print(f"    Found new run for {task_id}: {run['id'][:16]}...")
                                                break
                                except Exception:
                                    pass
        else:
            print("\n[6] No missing tasks to upload")

        # 7. Select ONE run per task (prefer graded, then most recent)
        print("\n[7] Selecting best run per task...")
        selected_runs = {}
        runs_to_delete = []

        for task_id, runs in task_to_runs.items():
            if not runs:
                continue

            # Sort: graded first, then by created_at descending
            sorted_runs = sorted(
                runs,
                key=lambda r: (r["is_graded"], r["created_at"]),
                reverse=True
            )

            # Select the best run
            best = sorted_runs[0]
            selected_runs[task_id] = best

            # Mark duplicates for deletion
            for run in sorted_runs[1:]:
                runs_to_delete.append({
                    "task_id": task_id,
                    "run_id": run["run_id"],
                    "reason": "duplicate",
                })

        print(f"    Selected runs: {len(selected_runs)}")
        print(f"    Duplicate runs to delete: {len(runs_to_delete)}")

        # Count graded
        graded_count = sum(1 for r in selected_runs.values() if r["is_graded"])
        print(f"    Selected runs that are graded: {graded_count}")

        # 8. Delete duplicate runs
        if runs_to_delete:
            print(f"\n[8] Deleting {len(runs_to_delete)} duplicate runs...")

            if args.dry_run:
                print("    [DRY RUN] Would delete:")
                for i, r in enumerate(runs_to_delete[:10]):
                    print(f"      - {r['task_id']}: {r['run_id'][:16]}...")
                if len(runs_to_delete) > 10:
                    print(f"      ... and {len(runs_to_delete) - 10} more")
            else:
                deleted = 0
                failed = 0
                for r in runs_to_delete:
                    try:
                        del_response = await client._client.delete(f"/runs/{r['run_id']}")
                        if del_response.status_code in (200, 204):
                            deleted += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1

                    if (deleted + failed) % 50 == 0:
                        print(f"    Progress: {deleted + failed}/{len(runs_to_delete)}")

                print(f"    Deleted: {deleted}, Failed: {failed}")
        else:
            print("\n[8] No duplicate runs to delete")

        # 9. Verify graded runs through API (spot check)
        print("\n[9] Verifying graded runs...")
        graded_verified = 0
        graded_sample = [t for t, r in selected_runs.items() if r["is_graded"]][:20]

        for task_id in graded_sample:
            run_id = selected_runs[task_id]["run_id"]
            try:
                # Check if run exists and has investigation
                check = await client._client.get(f"/runs/{run_id}")
                if check.status_code == 200:
                    graded_verified += 1
            except Exception:
                pass

        print(f"    Verified {graded_verified}/{len(graded_sample)} graded runs exist in API")

        # 10. Build final tracking file
        print("\n[10] Building final tracking file...")

        final_tracking = {}
        for task_id, run_info in selected_runs.items():
            entry = {
                "run_id": run_info["run_id"],
                "trajectory_id": run_info["trajectory_id"],
                "model": "mockllm/model",
                "uploaded": True,
                "graded": run_info["is_graded"],
                "created_at": run_info["created_at"],
            }

            # Add graded_at from feature file if available
            if run_info["is_graded"]:
                feature_file = FEATURES_DIR / f"{task_id}.json"
                if feature_file.exists():
                    try:
                        with open(feature_file) as f:
                            features = json.load(f)
                        if "_graded_at" in features:
                            entry["graded_at"] = features["_graded_at"]
                    except Exception:
                        pass

            final_tracking[task_id] = entry

        tracking_data = {
            "completed_tasks": final_tracking,
            "failed_tasks": {},
            "stats": {
                "total_uploaded": len(final_tracking),
                "total_graded": sum(1 for v in final_tracking.values() if v.get("graded")),
                "finalized_at": datetime.now().isoformat(),
            }
        }

        # 11. Write or show results
        if args.dry_run:
            print("\n[DRY RUN] Would write tracking file with:")
            print(f"    Total tasks: {len(final_tracking)}")
            print(f"    Graded tasks: {tracking_data['stats']['total_graded']}")
            print(f"    File: {TRACKING_FILE}")

            # Check coverage
            covered = set(final_tracking.keys())
            still_missing = all_task_ids - covered
            if still_missing:
                print(f"\n    WARNING: Still missing {len(still_missing)} tasks:")
                for t in sorted(still_missing)[:5]:
                    print(f"      - {t}")
        else:
            print("\n[11] Writing tracking file...")
            TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACKING_FILE, "w") as f:
                json.dump(tracking_data, f, indent=2)
            print(f"    Written: {TRACKING_FILE}")

        # 12. Final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total SWE-bench tasks: {len(all_task_ids)}")
        print(f"Tasks in tracking file: {len(final_tracking)}")
        print(f"Tasks graded: {tracking_data['stats']['total_graded']}")
        print(f"Tasks remaining to grade: {len(final_tracking) - tracking_data['stats']['total_graded']}")

        covered = set(final_tracking.keys())
        still_missing = all_task_ids - covered
        if still_missing:
            print(f"\nMISSING TASKS ({len(still_missing)}):")
            for t in sorted(still_missing):
                print(f"  - {t}")
        else:
            print("\n✓ All 500 tasks are covered!")


if __name__ == "__main__":
    asyncio.run(main())
