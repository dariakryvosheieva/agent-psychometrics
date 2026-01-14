#!/usr/bin/env python3
"""Verify consistency between local tracking, Lunette API, and graded features.

This script checks:
1. All 500 SWE-bench tasks - which are uploaded to Lunette?
2. Tracking file vs Lunette API - are they consistent?
3. Graded features - do we have JSON files for tasks marked as graded?
4. Lunette investigations - have the graded runs actually been investigated?

Usage:
    python -m experiment_a.verify_lunette_state
"""

import asyncio
import json
from pathlib import Path
from collections import defaultdict

# Paths
ROOT = Path(__file__).resolve().parents[1]
TRACKING_FILE = ROOT / "chris_output" / "experiment_a" / "sandbox_runs" / "tracking.json"
FEATURES_DIR = ROOT / "chris_output" / "experiment_a" / "sandbox_features"


async def main():
    from lunette import LunetteClient
    from datasets import load_dataset

    print("=" * 70)
    print("LUNETTE STATE VERIFICATION")
    print("=" * 70)

    # 1. Load all SWE-bench task IDs
    print("\n[1] Loading SWE-bench Verified tasks...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    all_task_ids = set(item["instance_id"] for item in ds)
    print(f"    Total SWE-bench Verified tasks: {len(all_task_ids)}")

    # 2. Load tracking file
    print("\n[2] Loading tracking file...")
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE) as f:
            tracking_data = json.load(f)
        tracked_tasks = tracking_data.get("completed_tasks", {})
        print(f"    Tasks in tracking file: {len(tracked_tasks)}")

        # Count tasks with valid run_ids
        valid_tracked = {k: v for k, v in tracked_tasks.items()
                        if v.get("run_id") and v["run_id"] != "unknown"}
        print(f"    Tasks with valid run_id: {len(valid_tracked)}")
    else:
        print("    ERROR: Tracking file not found!")
        tracked_tasks = {}
        valid_tracked = {}

    # 3. Query Lunette API for all mockllm/model runs
    print("\n[3] Querying Lunette API...")
    async with LunetteClient() as client:
        response = await client._client.get("/runs/")
        all_runs = response.json()

        # Filter to single-trajectory mockllm/model runs
        mockllm_runs = [r for r in all_runs
                       if r.get("model") == "mockllm/model"
                       and r.get("trajectory_count") == 1]
        print(f"    Total runs in API: {len(all_runs)}")
        print(f"    Single-trajectory mockllm/model runs: {len(mockllm_runs)}")

        # Build mapping: task_id -> run_id from API
        print("\n[4] Fetching task IDs from Lunette runs...")
        api_task_to_run = {}
        api_run_to_task = {}

        for i, run in enumerate(mockllm_runs):
            run_id = run["id"]
            try:
                detail = await client._client.get(f"/runs/{run_id}")
                if detail.status_code == 200:
                    data = detail.json()
                    for traj in data.get("trajectories", []):
                        task_id = traj.get("sample")
                        if task_id:
                            api_task_to_run[task_id] = {
                                "run_id": run_id,
                                "trajectory_id": traj.get("id"),
                            }
                            api_run_to_task[run_id] = task_id
            except Exception:
                pass

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(mockllm_runs)} runs...")

        print(f"    Tasks found in Lunette API: {len(api_task_to_run)}")

        # 5. Compare tracking vs API
        print("\n[5] Comparing tracking file vs Lunette API...")

        tracked_set = set(valid_tracked.keys())
        api_set = set(api_task_to_run.keys())

        in_tracking_not_api = tracked_set - api_set
        in_api_not_tracking = api_set - tracked_set
        in_both = tracked_set & api_set

        print(f"    In both tracking and API: {len(in_both)}")
        print(f"    In tracking but NOT in API: {len(in_tracking_not_api)}")
        print(f"    In API but NOT in tracking: {len(in_api_not_tracking)}")

        if in_tracking_not_api:
            print(f"\n    Tasks in tracking but missing from API (first 5):")
            for t in sorted(in_tracking_not_api)[:5]:
                print(f"      - {t} (run_id: {valid_tracked[t].get('run_id', 'N/A')[:16]}...)")

        if in_api_not_tracking:
            print(f"\n    Tasks in API but missing from tracking (first 5):")
            for t in sorted(in_api_not_tracking)[:5]:
                print(f"      - {t}")

        # Check for run_id mismatches
        mismatches = []
        for task_id in in_both:
            tracked_run_id = valid_tracked[task_id].get("run_id")
            api_run_id = api_task_to_run[task_id]["run_id"]
            if tracked_run_id != api_run_id:
                mismatches.append((task_id, tracked_run_id, api_run_id))

        if mismatches:
            print(f"\n    WARNING: {len(mismatches)} tasks have run_id MISMATCHES:")
            for task_id, tracked, api in mismatches[:5]:
                print(f"      - {task_id}: tracking={tracked[:16]}... vs API={api[:16]}...")
        else:
            print(f"\n    All {len(in_both)} matching tasks have consistent run_ids")

        # 6. Check graded features
        print("\n[6] Checking graded feature files...")

        feature_files = list(FEATURES_DIR.glob("*.json"))
        # Exclude non-task files
        graded_task_ids = set()
        for f in feature_files:
            if f.stem not in ("grading_stats", "lunette_features") and not f.stem.endswith("_error"):
                graded_task_ids.add(f.stem)

        print(f"    Feature JSON files found: {len(graded_task_ids)}")

        # Check tracking file for graded status
        tracked_as_graded = {k for k, v in tracked_tasks.items() if v.get("graded")}
        print(f"    Tasks marked as graded in tracking: {len(tracked_as_graded)}")

        # Compare
        graded_but_no_file = tracked_as_graded - graded_task_ids
        file_but_not_tracked = graded_task_ids - tracked_as_graded

        if graded_but_no_file:
            print(f"\n    WARNING: {len(graded_but_no_file)} tasks marked graded but NO feature file:")
            for t in sorted(graded_but_no_file)[:5]:
                print(f"      - {t}")

        if file_but_not_tracked:
            print(f"\n    Tasks with feature files but not marked graded: {len(file_but_not_tracked)}")

        # 7. Check Lunette investigations for graded runs
        print("\n[7] Checking Lunette investigations for graded runs...")

        # Sample a few graded tasks to verify they have investigations
        sample_graded = list(graded_task_ids)[:10]
        verified_investigations = 0

        for task_id in sample_graded:
            if task_id in api_task_to_run:
                run_id = api_task_to_run[task_id]["run_id"]
                try:
                    # Check if run has investigations
                    inv_response = await client._client.get(f"/runs/{run_id}")
                    if inv_response.status_code == 200:
                        run_data = inv_response.json()
                        # Check for investigation results in trajectories
                        for traj in run_data.get("trajectories", []):
                            if traj.get("sample") == task_id:
                                # The presence of scored investigations would be in a different endpoint
                                # For now, just verify the run exists
                                verified_investigations += 1
                                break
                except Exception:
                    pass

        print(f"    Verified {verified_investigations}/{len(sample_graded)} sampled graded runs exist in API")

        # 8. Summary of missing tasks
        print("\n[8] Coverage Summary...")
        missing_from_lunette = all_task_ids - api_set
        print(f"    Total SWE-bench tasks: {len(all_task_ids)}")
        print(f"    Uploaded to Lunette: {len(api_set)}")
        print(f"    Missing from Lunette: {len(missing_from_lunette)}")
        print(f"    Graded (have features): {len(graded_task_ids)}")
        print(f"    Remaining to grade: {len(api_set) - len(graded_task_ids)}")

        if missing_from_lunette:
            print(f"\n    Missing tasks (first 10):")
            for t in sorted(missing_from_lunette)[:10]:
                print(f"      - {t}")
            if len(missing_from_lunette) > 10:
                print(f"      ... and {len(missing_from_lunette) - 10} more")

        # 9. Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        if missing_from_lunette:
            print(f"\n1. Upload {len(missing_from_lunette)} missing tasks:")
            print("   python -m experiment_a.run_dummy_sandbox --resume")

        if in_api_not_tracking:
            print(f"\n2. Update tracking file with {len(in_api_not_tracking)} tasks from API:")
            print("   (Run the rebuild tracking script)")

        remaining_to_grade = api_set - graded_task_ids
        if remaining_to_grade:
            print(f"\n3. Grade {len(remaining_to_grade)} remaining tasks:")
            print("   python -m experiment_a.grade_sandbox_runs --skip_existing")


if __name__ == "__main__":
    asyncio.run(main())
