"""Extract trajectory features for tasks.

This script extracts features from agent trajectories using the
LLMFeatureExtractor infrastructure.

Usage:
    # Dry run on frontier tasks (cost estimate)
    python -m experiment_b.trajectory_features.extract_frontier_features --version v2 --dry-run

    # Extract from frontier tasks (default)
    python -m experiment_b.trajectory_features.extract_frontier_features --version v2

    # Extract from custom task list (e.g., all-fail non-frontier tasks)
    python -m experiment_b.trajectory_features.extract_frontier_features --version v2 \\
        --task-list chris_output/trajectory_features/all_fail_nonfrontier_tasks.json \\
        --output-dir chris_output/trajectory_features/nonfrontier_v2_openhands

    # Limit number of tasks (useful for testing)
    python -m experiment_b.trajectory_features.extract_frontier_features --version v2 --limit 5
"""

import argparse
import json
from pathlib import Path

from experiment_b.trajectory_features.simple_extractor import SimpleFeatureExtractor
from experiment_b.swebench.config import SWEBenchConfig
from experiment_b.trajectory_features.utils import (
    build_task_dicts,
    load_frontier_tasks_with_difficulties,
)


def get_prompt_config(version: str):
    """Get the prompt configuration for a version."""
    if version == "v1":
        from experiment_b.trajectory_features.prompts_frontier_v1 import (
            get_frontier_v1_config,
        )
        return get_frontier_v1_config()
    elif version == "v2":
        from experiment_b.trajectory_features.prompts_frontier_v2 import (
            get_frontier_v2_config,
        )
        return get_frontier_v2_config()
    elif version == "v3":
        from experiment_b.trajectory_features.prompts_frontier_v3 import (
            get_frontier_v3_config,
        )
        return get_frontier_v3_config()
    else:
        raise ValueError(f"Unknown version: {version}. Available: v1, v2, v3")


def main():
    parser = argparse.ArgumentParser(
        description="Extract trajectory features for frontier tasks"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Prompt version to use (default: v1)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="20250415_openhands",
        help="Agent to extract trajectories from (default: 20250415_openhands)",
    )
    parser.add_argument(
        "--trajs-dir",
        type=Path,
        default=Path("trajectory_data/unified_trajs"),
        help="Directory containing trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: chris_output/trajectory_features/frontier_{version})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan and cost estimate without running",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider's default)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run extraction in parallel (faster but higher API load)",
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help="JSON file containing list of task IDs to extract (default: frontier tasks)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(f"chris_output/trajectory_features/frontier_{args.version}")

    # Load configuration
    config = SWEBenchConfig()
    prompt_config = get_prompt_config(args.version)

    print(f"Prompt version: {args.version}")
    print(f"Agent: {args.agent}")
    print(f"Output directory: {args.output_dir}")

    # Load task list
    if args.task_list:
        # Use custom task list
        print(f"\nLoading task list from {args.task_list}...")
        with open(args.task_list) as f:
            target_tasks = json.load(f)
        print(f"  Loaded {len(target_tasks)} tasks from custom list")
    else:
        # Default: frontier tasks
        print("\nLoading frontier tasks...")
        target_tasks, oracle_items, pre_frontier, post_frontier = (
            load_frontier_tasks_with_difficulties(config)
        )
        print(f"  Pre-frontier agents: {len(pre_frontier)}")
        print(f"  Post-frontier agents: {len(post_frontier)}")
        print(f"  Frontier tasks: {len(target_tasks)}")

    # Build task dictionaries
    print(f"\nLoading trajectories from {args.agent}...")
    task_dicts, missing = build_task_dicts(
        target_tasks, args.agent, args.trajs_dir
    )
    print(f"  Built {len(task_dicts)} task dictionaries")
    if missing:
        print(f"  Warning: {len(missing)} trajectories not found")

    # Create extractor
    extractor = SimpleFeatureExtractor(
        prompt_config=prompt_config,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
    )

    # Run extraction
    if args.dry_run:
        extractor.dry_run(task_dicts, limit=args.limit)
    elif args.parallel:
        csv_path = extractor.run_parallel(task_dicts, limit=args.limit)
        if csv_path:
            print(f"\nFeatures saved to: {csv_path}")
    else:
        csv_path = extractor.run(task_dicts, limit=args.limit)
        if csv_path:
            print(f"\nFeatures saved to: {csv_path}")


if __name__ == "__main__":
    main()
