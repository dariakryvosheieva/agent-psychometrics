"""Extract missing trajectory features to reach target coverage.

This script fills in missing trajectories to reach a target number of agents per task,
reusing existing extracted data.

Usage:
    # Fill to 6 agents per task for 500 tasks, reusing existing data
    python -m experiment_appendix_h_hard_tasks.trajectory_features.extract_missing \
        --existing_path chris_output/trajectory_features/raw_features_100tasks.csv \
        --output_path chris_output/trajectory_features/raw_features_500tasks_6agents.csv \
        --n_tasks 500 \
        --agents_per_task 6 \
        --parallel 20
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from .config import AGENT_NAMES, SELECTED_AGENTS
from .extract_features import TrajectoryFeatureExtractor


def get_target_agents_for_task(
    task_id: str,
    agents_per_task: int,
    trajectory_dir: Path,
    agent_abilities: dict,
) -> List[str]:
    """Get the target agents for a task, evenly spaced by ability."""
    # Find all available agents for this task
    available_agents = []
    for agent in AGENT_NAMES:
        if (trajectory_dir / agent / f"{task_id}.json").exists():
            available_agents.append(agent)

    if len(available_agents) < agents_per_task:
        return available_agents  # Return all if not enough

    # Sort by ability and pick evenly spaced
    available_with_ability = [(a, agent_abilities.get(a, 0)) for a in available_agents]
    available_with_ability.sort(key=lambda x: x[1])

    step = len(available_with_ability) / agents_per_task
    indices = [int(i * step) for i in range(agents_per_task)]
    return [available_with_ability[i][0] for i in indices]


def compute_missing_pairs(
    existing_df: pd.DataFrame,
    n_tasks: int,
    agents_per_task: int,
    trajectory_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[str], int]:
    """Compute which (agent, task_id) pairs are missing.

    Returns:
        Tuple of (pairs_to_extract, selected_tasks, n_reused)
    """
    agent_abilities = {a.name: a.theta for a in SELECTED_AGENTS}

    # Get existing pairs
    if len(existing_df) > 0:
        existing_pairs = set(zip(existing_df['agent'], existing_df['task_id']))
    else:
        existing_pairs = set()

    # Build task -> available agents mapping
    task_agents: Dict[str, List[str]] = {}
    for agent in AGENT_NAMES:
        agent_dir = trajectory_dir / agent
        if not agent_dir.exists():
            continue
        for task_file in agent_dir.glob("*.json"):
            task_id = task_file.stem
            if task_id not in task_agents:
                task_agents[task_id] = []
            task_agents[task_id].append(agent)

    # Filter to tasks with enough agents
    valid_tasks = [t for t, a in task_agents.items() if len(a) >= agents_per_task]
    print(f"Found {len(valid_tasks)} tasks with >= {agents_per_task} agents")

    # Prioritize tasks we already have partial data for
    existing_tasks = set(existing_df['task_id'].unique()) if len(existing_df) > 0 else set()
    tasks_with_data = [t for t in valid_tasks if t in existing_tasks]
    tasks_without_data = [t for t in valid_tasks if t not in existing_tasks]

    print(f"  Tasks with existing data: {len(tasks_with_data)}")
    print(f"  Tasks without existing data: {len(tasks_without_data)}")

    # Select tasks: prioritize ones with existing data
    import random
    random.seed(42)

    selected_tasks = tasks_with_data.copy()
    if len(selected_tasks) < n_tasks:
        remaining_needed = n_tasks - len(selected_tasks)
        random.shuffle(tasks_without_data)
        selected_tasks.extend(tasks_without_data[:remaining_needed])
    else:
        selected_tasks = selected_tasks[:n_tasks]

    selected_tasks = sorted(selected_tasks)
    print(f"Selected {len(selected_tasks)} tasks")

    # Compute missing pairs for each task
    task_missing_pairs: Dict[str, List[Tuple[str, str]]] = {}
    n_reused = 0
    for task_id in selected_tasks:
        target_agents = get_target_agents_for_task(
            task_id, agents_per_task, trajectory_dir, agent_abilities
        )
        missing = []
        for agent in target_agents:
            if (agent, task_id) in existing_pairs:
                n_reused += 1
            else:
                missing.append((agent, task_id))
        task_missing_pairs[task_id] = missing

    # Sort tasks by how many are missing (fewest first = closest to complete)
    # This prioritizes filling tasks that are almost at agents_per_task
    sorted_tasks = sorted(task_missing_pairs.keys(),
                         key=lambda t: len(task_missing_pairs[t]))

    # Build pairs list in priority order
    pairs_to_extract = []
    for task_id in sorted_tasks:
        pairs_to_extract.extend(task_missing_pairs[task_id])

    # Show priority breakdown
    missing_counts = [len(task_missing_pairs[t]) for t in selected_tasks]
    from collections import Counter
    count_dist = Counter(missing_counts)
    print(f"Tasks by missing count: {dict(sorted(count_dist.items()))}")

    return pairs_to_extract, selected_tasks, n_reused


async def extract_missing(
    existing_path: Path,
    output_path: Path,
    n_tasks: int,
    agents_per_task: int,
    parallel: int,
    model: str,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Extract missing trajectories and merge with existing data."""
    trajectory_dir = Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs")

    # Load existing data
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        print(f"Loaded {len(existing_df)} existing trajectories from {existing_path}")
    else:
        existing_df = pd.DataFrame()
        print("No existing data found, starting fresh")

    # Compute what's missing
    pairs_to_extract, selected_tasks, n_reused = compute_missing_pairs(
        existing_df, n_tasks, agents_per_task, trajectory_dir
    )

    print(f"\n=== EXTRACTION PLAN ===")
    print(f"Target: {n_tasks} tasks × {agents_per_task} agents = {n_tasks * agents_per_task} trajectories")
    print(f"Reusing: {n_reused} existing trajectories")
    print(f"Need to extract: {len(pairs_to_extract)} new trajectories")

    # Estimate cost
    cost_per_traj = 0.044
    estimated_cost = len(pairs_to_extract) * cost_per_traj
    print(f"Estimated cost: ${estimated_cost:.2f}")

    if dry_run:
        print("\n[DRY RUN] Would extract the above trajectories.")
        return existing_df

    if len(pairs_to_extract) == 0:
        print("\nNothing to extract - all trajectories already exist!")
        # Filter existing to selected tasks and save
        result_df = existing_df[existing_df['task_id'].isin(selected_tasks)]
        result_df.to_csv(output_path, index=False)
        print(f"Saved {len(result_df)} rows to {output_path}")
        return result_df

    # Extract missing trajectories
    extractor = TrajectoryFeatureExtractor(model=model)
    print(f"\nUsing model: {model}")
    print(f"Parallel calls: {parallel}")
    print(f"Starting extraction...\n")

    # Save incrementally to a temp file (for crash recovery)
    temp_output = output_path.parent / f".{output_path.stem}_incremental.csv"

    new_results = await extractor.extract_batch_async(
        pairs_to_extract,
        output_path=temp_output,  # Save incrementally for crash recovery
        resume=True,  # Resume from temp file if it exists
        parallel=parallel,
    )

    print(f"\nExtracted {len(new_results)} new trajectories")
    print(f"Usage: {extractor.usage.summary()}")

    # Merge with existing (only keep rows for selected tasks)
    existing_for_selected = existing_df[existing_df['task_id'].isin(selected_tasks)]

    if len(new_results) > 0:
        merged_df = pd.concat([existing_for_selected, new_results], ignore_index=True)
    else:
        merged_df = existing_for_selected

    # Remove any duplicates (shouldn't happen, but safety check)
    merged_df = merged_df.drop_duplicates(subset=['agent', 'task_id'], keep='last')

    # Verify coverage
    task_counts = merged_df.groupby('task_id').size()
    print(f"\n=== FINAL COVERAGE ===")
    print(f"Total trajectories: {len(merged_df)}")
    print(f"Tasks: {len(task_counts)}")
    print(f"Agents per task: min={task_counts.min()}, max={task_counts.max()}, mean={task_counts.mean():.1f}")

    # Save
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(merged_df)} rows to {output_path}")

    # Clean up temp file after successful save
    if temp_output.exists():
        temp_output.unlink()
        print(f"Cleaned up temp file: {temp_output}")

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract missing trajectory features to reach target coverage"
    )
    parser.add_argument(
        "--existing_path",
        type=str,
        default="chris_output/trajectory_features/raw_features_100tasks.csv",
        help="Path to existing extracted features",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="chris_output/trajectory_features/raw_features_500tasks_6agents.csv",
        help="Output path for merged results",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=500,
        help="Target number of tasks",
    )
    parser.add_argument(
        "--agents_per_task",
        type=int,
        default=6,
        help="Target number of agents per task",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=20,
        help="Number of parallel API calls",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be extracted, don't actually run",
    )
    args = parser.parse_args()

    asyncio.run(extract_missing(
        existing_path=Path(args.existing_path),
        output_path=Path(args.output_path),
        n_tasks=args.n_tasks,
        agents_per_task=args.agents_per_task,
        parallel=args.parallel,
        model=args.model,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
