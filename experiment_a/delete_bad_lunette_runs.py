#!/usr/bin/env python3
"""Delete bad Lunette runs that need cleanup.

This script identifies and deletes:
1. Runs from dummy_agent and dummy_agent_v2 (wrong approach - used save_run without sandbox)
2. Batched runs with multiple trajectories (causes grading issues)
3. Any runs that weren't created with the proper 1-run-1-trajectory sandbox method

Usage:
    # Dry run - see what would be deleted
    python -m experiment_a.delete_bad_lunette_runs --dry_run

    # Actually delete
    python -m experiment_a.delete_bad_lunette_runs

    # Delete only dummy_agent runs
    python -m experiment_a.delete_bad_lunette_runs --agents dummy_agent,dummy_agent_v2
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path


async def get_all_runs(client):
    """Fetch all runs from Lunette."""
    response = await client._client.get("/runs/")
    return response.json()


async def get_run_details(client, run_id: str):
    """Get details for a specific run including trajectory count."""
    try:
        response = await client._client.get(f"/runs/{run_id}")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"  Error fetching {run_id}: {e}")
    return None


async def delete_run(client, run_id: str, dry_run: bool = True) -> bool:
    """Delete a run from Lunette."""
    if dry_run:
        print(f"  [DRY RUN] Would delete: {run_id}")
        return True

    try:
        response = await client._client.delete(f"/runs/{run_id}")
        if response.status_code == 200:
            print(f"  Deleted: {run_id}")
            return True
        else:
            print(f"  Failed to delete {run_id}: {response.status_code}")
            return False
    except Exception as e:
        print(f"  Error deleting {run_id}: {e}")
        return False


async def main(
    dry_run: bool = True,
    delete_dummy_agents: bool = True,
    delete_batched: bool = True,
    agents_to_delete: list[str] | None = None,
    keep_run_ids: set[str] | None = None,
):
    """Main function to identify and delete bad runs."""
    from lunette import LunetteClient

    # Load tracking file to identify good runs to keep
    tracking_path = Path("chris_output/experiment_a/sandbox_runs/tracking.json")
    good_run_ids = set()

    if tracking_path.exists():
        with open(tracking_path) as f:
            tracking = json.load(f)

        # Identify run_ids that have only 1 task (unbatched)
        run_id_counts = Counter()
        for task_id, task_data in tracking.get("completed_tasks", {}).items():
            run_id = task_data.get("run_id")
            if run_id and run_id != "unknown":
                run_id_counts[run_id] += 1

        # Only keep run_ids that have exactly 1 task
        for run_id, count in run_id_counts.items():
            if count == 1:
                good_run_ids.add(run_id)

        print(f"Found {len(good_run_ids)} good unbatched run_ids to keep")

    if keep_run_ids:
        good_run_ids.update(keep_run_ids)

    # Default agents to delete if none specified
    if agents_to_delete is None:
        agents_to_delete = ["dummy_agent", "dummy_agent_v2"]

    async with LunetteClient() as client:
        print("Fetching all runs from Lunette...")
        runs = await get_all_runs(client)
        print(f"Found {len(runs)} total runs")

        # Categorize runs
        runs_to_delete = []
        runs_to_keep = []

        for run in runs:
            run_id = run.get("id")
            model = run.get("model", "unknown")
            task = run.get("task", "unknown")
            traj_count = run.get("trajectory_count", 0)

            # Check if this is a good run to keep
            if run_id in good_run_ids:
                runs_to_keep.append({
                    "run_id": run_id,
                    "model": model,
                    "reason": "unbatched sandbox run"
                })
                continue

            # Check if this is a dummy agent run to delete
            if delete_dummy_agents and model in agents_to_delete:
                runs_to_delete.append({
                    "run_id": run_id,
                    "model": model,
                    "task": task,
                    "traj_count": traj_count,
                    "reason": f"dummy agent ({model})"
                })
                continue

            # Check if this is a batched run (multiple trajectories) from mockllm
            if delete_batched and model == "mockllm/model" and traj_count > 1:
                runs_to_delete.append({
                    "run_id": run_id,
                    "model": model,
                    "task": task,
                    "traj_count": traj_count,
                    "reason": f"batched run ({traj_count} trajectories)"
                })
                continue

            # Keep everything else (real agent trajectories, etc.)
            runs_to_keep.append({
                "run_id": run_id,
                "model": model,
                "reason": "real agent or other"
            })

        print(f"\n=== Summary ===")
        print(f"Runs to DELETE: {len(runs_to_delete)}")
        print(f"Runs to KEEP: {len(runs_to_keep)}")

        if runs_to_delete:
            print(f"\n=== Runs to Delete ===")

            # Group by reason
            by_reason = {}
            for run in runs_to_delete:
                reason = run["reason"]
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(run)

            for reason, runs_list in sorted(by_reason.items()):
                print(f"\n{reason}: {len(runs_list)} runs")
                for run in runs_list[:5]:  # Show first 5
                    print(f"  - {run['run_id'][:8]}... ({run.get('traj_count', '?')} trajs)")
                if len(runs_list) > 5:
                    print(f"  ... and {len(runs_list) - 5} more")

            if dry_run:
                print(f"\n[DRY RUN] Would delete {len(runs_to_delete)} runs")
                print("Run without --dry_run to actually delete")
            else:
                print(f"\nDeleting {len(runs_to_delete)} runs...")
                deleted = 0
                failed = 0
                for run in runs_to_delete:
                    success = await delete_run(client, run["run_id"], dry_run=False)
                    if success:
                        deleted += 1
                    else:
                        failed += 1

                print(f"\nDeleted: {deleted}")
                print(f"Failed: {failed}")
        else:
            print("\nNo runs to delete!")

        # Show what we're keeping
        print(f"\n=== Runs Being Kept ===")
        keep_by_model = {}
        for run in runs_to_keep:
            model = run["model"]
            if model not in keep_by_model:
                keep_by_model[model] = 0
            keep_by_model[model] += 1

        for model, count in sorted(keep_by_model.items(), key=lambda x: -x[1]):
            print(f"  {model}: {count} runs")


def delete_bad_graded_files(dry_run: bool = True) -> int:
    """Delete graded JSON files that came from batched runs.

    Returns the number of files deleted.
    """
    from collections import Counter

    # Load tracking to find batched run_ids
    tracking_path = Path("chris_output/experiment_a/sandbox_runs/tracking.json")
    if not tracking_path.exists():
        print("No tracking file found")
        return 0

    with open(tracking_path) as f:
        tracking = json.load(f)

    # Count tasks per run_id to find batched runs
    run_id_counts = Counter()
    for task_id, task_data in tracking.get("completed_tasks", {}).items():
        run_id = task_data.get("run_id")
        if run_id and run_id != "unknown":
            run_id_counts[run_id] += 1

    # Find batched run_ids (more than 1 task)
    batched_run_ids = {rid for rid, count in run_id_counts.items() if count > 1}
    print(f"Found {len(batched_run_ids)} batched run_ids")

    # Check which graded files came from batched runs
    features_dir = Path("chris_output/experiment_a/sandbox_features")
    files_to_delete = []

    for json_file in features_dir.glob("*.json"):
        if "_error" in json_file.name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
            run_id = data.get("_run_id")

            if run_id in batched_run_ids:
                files_to_delete.append(json_file)
        except:
            pass

    print(f"Found {len(files_to_delete)} graded files from batched runs to delete")

    if files_to_delete:
        for f in files_to_delete[:10]:
            print(f"  {f.name}")
        if len(files_to_delete) > 10:
            print(f"  ... and {len(files_to_delete) - 10} more")

        if dry_run:
            print(f"\n[DRY RUN] Would delete {len(files_to_delete)} graded files")
        else:
            for f in files_to_delete:
                f.unlink()
                print(f"  Deleted: {f.name}")
            print(f"Deleted {len(files_to_delete)} graded files")

    return len(files_to_delete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete bad Lunette runs")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--agents",
        type=str,
        help="Comma-separated list of agent names to delete (default: dummy_agent,dummy_agent_v2)"
    )
    parser.add_argument(
        "--no_dummy_agents",
        action="store_true",
        help="Don't delete dummy agent runs"
    )
    parser.add_argument(
        "--no_batched",
        action="store_true",
        help="Don't delete batched runs"
    )
    parser.add_argument(
        "--cleanup_graded",
        action="store_true",
        help="Also delete graded JSON files that came from batched runs"
    )

    args = parser.parse_args()

    agents = None
    if args.agents:
        agents = [a.strip() for a in args.agents.split(",")]

    # Delete bad graded files first
    if args.cleanup_graded:
        print("=== Cleaning up bad graded files ===\n")
        delete_bad_graded_files(dry_run=args.dry_run)
        print()

    # Then delete bad Lunette runs
    asyncio.run(main(
        dry_run=args.dry_run,
        delete_dummy_agents=not args.no_dummy_agents,
        delete_batched=not args.no_batched,
        agents_to_delete=agents,
    ))
