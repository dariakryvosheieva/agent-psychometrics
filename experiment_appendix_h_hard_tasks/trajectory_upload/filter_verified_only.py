#!/usr/bin/env python3
"""Filter unified_trajs to only include SWE-bench Verified tasks.

This script:
1. Loads the list of 500 SWE-bench Verified task IDs
2. Removes any trajectory files that are NOT in the Verified set
3. Reports what was removed

Usage:
    # Dry run (preview what would be deleted)
    python trajectory_upload/filter_verified_only.py --dry_run

    # Actually delete non-Verified trajectories
    python trajectory_upload/filter_verified_only.py

    # Specify a different directory
    python trajectory_upload/filter_verified_only.py --trajectory_dir path/to/unified_trajs
"""

import argparse
import json
from pathlib import Path


def load_verified_task_ids() -> set:
    """Load the set of SWE-bench Verified task IDs."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return set(ex["instance_id"] for ex in ds)


def filter_verified_only(
    trajectory_dir: Path,
    dry_run: bool = True,
) -> dict:
    """Remove non-Verified trajectories from unified_trajs.

    Args:
        trajectory_dir: Path to unified_trajs directory
        dry_run: If True, only report what would be deleted

    Returns:
        Summary dict with counts and deleted files
    """
    verified_ids = load_verified_task_ids()
    print(f"Loaded {len(verified_ids)} SWE-bench Verified task IDs")

    summary = {
        "verified_count": len(verified_ids),
        "total_files": 0,
        "kept_files": 0,
        "deleted_files": 0,
        "deleted_by_agent": {},
        "deleted_task_ids": set(),
    }

    # Iterate over all agent directories
    for agent_dir in sorted(trajectory_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        if agent_dir.name.startswith("_") or agent_dir.name.startswith("."):
            continue

        agent_name = agent_dir.name
        agent_deleted = 0

        for json_file in sorted(agent_dir.glob("*.json")):
            # Skip metadata files (starting with _)
            if json_file.name.startswith("_"):
                continue

            summary["total_files"] += 1
            task_id = json_file.stem

            if task_id in verified_ids:
                summary["kept_files"] += 1
            else:
                summary["deleted_files"] += 1
                summary["deleted_task_ids"].add(task_id)
                agent_deleted += 1

                if not dry_run:
                    json_file.unlink()

        if agent_deleted > 0:
            summary["deleted_by_agent"][agent_name] = agent_deleted

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Filter unified_trajs to only include SWE-bench Verified tasks"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        default=Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs"),
        help="Path to unified_trajs directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    if not args.trajectory_dir.exists():
        print(f"Error: Directory not found: {args.trajectory_dir}")
        return

    print(f"Filtering trajectories in: {args.trajectory_dir}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    summary = filter_verified_only(args.trajectory_dir, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total trajectory files: {summary['total_files']:,}")
    print(f"Kept (Verified): {summary['kept_files']:,}")
    print(f"{'Would delete' if args.dry_run else 'Deleted'}: {summary['deleted_files']:,}")

    if summary["deleted_by_agent"]:
        print(f"\n{'Would delete' if args.dry_run else 'Deleted'} by agent:")
        for agent, count in sorted(
            summary["deleted_by_agent"].items(), key=lambda x: -x[1]
        )[:20]:
            print(f"  {agent}: {count}")
        if len(summary["deleted_by_agent"]) > 20:
            print(f"  ... and {len(summary['deleted_by_agent']) - 20} more agents")

    # Show sample of deleted task IDs
    sample_deleted = sorted(summary["deleted_task_ids"])[:10]
    if sample_deleted:
        print(f"\nSample {'to delete' if args.dry_run else 'deleted'} task IDs:")
        for tid in sample_deleted:
            print(f"  {tid}")
        if len(summary["deleted_task_ids"]) > 10:
            print(f"  ... and {len(summary['deleted_task_ids']) - 10} more")

    if args.dry_run:
        print("\n*** DRY RUN - no files were deleted ***")
        print("Run without --dry_run to actually delete files")


if __name__ == "__main__":
    main()
