#!/usr/bin/env python3
"""Fix unified trajectories that have list content instead of string content.

Some agents (OpenHands, codesweep) had their content stored as lists of content
blocks like [{'type': 'text', 'text': '...'}] instead of plain strings due to
a bug in convert_yaml_history(). This script normalizes all content to strings.

Usage:
    # Dry run to see what would be fixed
    python scripts/fix_unified_trajectories.py --dry_run

    # Fix all trajectories
    python scripts/fix_unified_trajectories.py

    # Fix specific agent
    python scripts/fix_unified_trajectories.py --agent 20250804_codesweep_sweagent_kimi_k2_instruct
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def normalize_content(content: Any) -> str:
    """Normalize message content to string.

    Handles list content like [{'type': 'text', 'text': '...'}].
    """
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", str(item)))
            else:
                text_parts.append(str(item))
        return " ".join(text_parts)
    return str(content) if content else ""


def fix_trajectory(traj_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """Fix a single trajectory file.

    Returns dict with counts of fixed messages.
    """
    with open(traj_path) as f:
        data = json.load(f)

    messages = data.get("messages", [])
    fixed_count = 0

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = normalize_content(content)
            fixed_count += 1

    if fixed_count > 0 and not dry_run:
        with open(traj_path, "w") as f:
            json.dump(data, f, indent=2)

    return {"fixed": fixed_count, "total": len(messages)}


def main():
    parser = argparse.ArgumentParser(description="Fix unified trajectories with list content")
    parser.add_argument(
        "--trajectories_dir",
        type=str,
        default="experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs",
        help="Directory containing unified trajectories",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Fix only a specific agent (directory name)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be fixed without modifying files",
    )
    args = parser.parse_args()

    traj_dir = Path(args.trajectories_dir)
    if not traj_dir.exists():
        print(f"Error: Directory not found: {traj_dir}")
        return

    # Collect agent directories to process
    if args.agent:
        agent_dirs = [traj_dir / args.agent]
        if not agent_dirs[0].exists():
            print(f"Error: Agent directory not found: {agent_dirs[0]}")
            return
    else:
        agent_dirs = sorted([d for d in traj_dir.iterdir() if d.is_dir()])

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(agent_dirs)} agent directories...")
    print()

    total_files = 0
    total_fixed_files = 0
    total_fixed_messages = 0

    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        traj_files = list(agent_dir.glob("*.json"))
        # Skip metadata files
        traj_files = [f for f in traj_files if not f.name.startswith("_")]

        agent_fixed_files = 0
        agent_fixed_messages = 0

        for traj_file in traj_files:
            try:
                result = fix_trajectory(traj_file, dry_run=args.dry_run)
                if result["fixed"] > 0:
                    agent_fixed_files += 1
                    agent_fixed_messages += result["fixed"]
            except Exception as e:
                print(f"  Error processing {traj_file.name}: {e}")

        total_files += len(traj_files)
        total_fixed_files += agent_fixed_files
        total_fixed_messages += agent_fixed_messages

        if agent_fixed_files > 0:
            print(f"  {agent_name}: {agent_fixed_files} files, {agent_fixed_messages} messages fixed")

    print()
    print("=" * 60)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files with fixes: {total_fixed_files}")
    print(f"  Messages fixed: {total_fixed_messages}")

    if args.dry_run and total_fixed_messages > 0:
        print()
        print("Run without --dry_run to apply fixes.")


if __name__ == "__main__":
    main()
