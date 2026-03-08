"""EDA script for viewing frontier tasks and their trajectories.

This script helps with qualitative analysis of frontier tasks to inform
feature design for difficulty prediction.

Usage:
    python -m experiment_appendix_h_hard_tasks.trajectory_features.frontier_trajectory_eda --limit 5
    python -m experiment_appendix_h_hard_tasks.trajectory_features.frontier_trajectory_eda --agent 20250415_openhands
    python -m experiment_appendix_h_hard_tasks.trajectory_features.frontier_trajectory_eda --sort desc  # hardest first
"""

import argparse
from pathlib import Path

from experiment_appendix_h_hard_tasks.swebench.config import SWEBenchConfig
from experiment_appendix_h_hard_tasks.trajectory_features.config import SELECTED_AGENTS
from experiment_appendix_h_hard_tasks.trajectory_features.prompts import format_trajectory_for_prompt
from experiment_appendix_h_hard_tasks.trajectory_features.utils import (
    load_frontier_tasks_with_difficulties,
    load_trajectory,
)


def display_task_with_trajectory(
    task_id: str,
    oracle_difficulty: float,
    trajectory: dict,
    rank: int,
    total: int,
    max_chars: int = 3000,
) -> None:
    """Display task info and trajectory excerpt."""
    print("\n" + "=" * 80)
    print(f"FRONTIER TASK [{rank}/{total}]")
    print("=" * 80)

    print(f"\nTask ID: {task_id}")
    print(f"Oracle difficulty (β): {oracle_difficulty:.3f}")
    print(f"Agent: {trajectory['agent']}")
    print(f"Resolved: {trajectory['resolved']}")
    print(f"Messages: {len(trajectory.get('messages', []))}")

    # Format trajectory with truncation
    formatted = format_trajectory_for_prompt(
        trajectory,
        max_messages=20,  # Limit for EDA viewing
        max_chars_per_message=500,
    )

    # Further truncate for display
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars] + "\n\n... [truncated for display]"

    print("\n" + "-" * 40)
    print("TRAJECTORY EXCERPT:")
    print("-" * 40)
    print(formatted)


def main():
    parser = argparse.ArgumentParser(
        description="View frontier tasks sorted by oracle difficulty with trajectory excerpts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of tasks to display (default: 10)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="20250415_openhands",
        help="Agent to view trajectories from (default: 20250415_openhands)",
    )
    parser.add_argument(
        "--sort",
        choices=["asc", "desc"],
        default="asc",
        help="Sort order: 'asc' for easiest first, 'desc' for hardest first (default: asc)",
    )
    parser.add_argument(
        "--trajs-dir",
        type=Path,
        default=Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs"),
        help="Directory containing trajectory files",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="View a specific task by ID (ignores --limit and --sort)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list frontier tasks with difficulties, don't show trajectories",
    )

    args = parser.parse_args()

    # Load config and data
    config = SWEBenchConfig()
    frontier_tasks, oracle_items, pre_frontier, post_frontier = (
        load_frontier_tasks_with_difficulties(config)
    )

    print(f"Pre-frontier agents: {len(pre_frontier)}")
    print(f"Post-frontier agents: {len(post_frontier)}")
    print(f"Frontier tasks (zero_pre): {len(frontier_tasks)}")

    # Get difficulties for frontier tasks
    frontier_difficulties = []
    for task_id in frontier_tasks:
        if task_id in oracle_items.index:
            frontier_difficulties.append((task_id, oracle_items.loc[task_id, "b"]))
        else:
            print(f"Warning: Task {task_id} not found in oracle IRT items")

    # Sort by difficulty
    ascending = args.sort == "asc"
    frontier_difficulties.sort(key=lambda x: x[1], reverse=not ascending)

    print(f"\n{'Easiest' if ascending else 'Hardest'} frontier tasks first:")
    print(f"Difficulty range: {frontier_difficulties[0][1]:.3f} to {frontier_difficulties[-1][1]:.3f}")

    # Handle --task-id flag
    if args.task_id:
        task_ids_to_show = [(args.task_id, oracle_items.loc[args.task_id, "b"])]
    else:
        task_ids_to_show = frontier_difficulties[:args.limit]

    # List-only mode
    if args.list_only:
        print("\n" + "-" * 60)
        print(f"{'Rank':<6} {'Difficulty':>12} {'Task ID'}")
        print("-" * 60)
        for i, (task_id, difficulty) in enumerate(frontier_difficulties):
            print(f"{i+1:<6} {difficulty:>12.3f} {task_id}")
        return

    # Display tasks with trajectories
    total = len(task_ids_to_show)
    for rank, (task_id, difficulty) in enumerate(task_ids_to_show, 1):
        try:
            trajectory = load_trajectory(args.agent, task_id, args.trajs_dir)
            display_task_with_trajectory(
                task_id, difficulty, trajectory, rank, total
            )
        except FileNotFoundError as e:
            print(f"\nSkipping {task_id}: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Displayed {total} of {len(frontier_difficulties)} frontier tasks")
    print(f"Agent: {args.agent}")
    print(f"\nAvailable pre-frontier agents for trajectory viewing:")
    for agent_info in SELECTED_AGENTS:
        print(f"  - {agent_info.name} (θ={agent_info.theta:.2f})")


if __name__ == "__main__":
    main()
