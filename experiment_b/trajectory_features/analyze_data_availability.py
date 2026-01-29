"""Analyze data availability for trajectory feature extraction.

This script performs Phase 1 analysis to understand:
1. How many non-frontier tasks have all 13 agents failing
2. Partial-fail distribution (how many tasks have N agents failing)
3. Whether features differ between success/failure on same task

Usage:
    python -m experiment_b.trajectory_features.analyze_data_availability
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from experiment_b.swebench.config import SWEBenchConfig
from experiment_b.shared.data_preparation import (
    identify_frontier_tasks_zero_pre,
    split_agents_by_dates,
)
from experiment_b.trajectory_features.build_frontier_model import FULL_COVERAGE_AGENTS


def get_task_agent_outcomes(
    config: SWEBenchConfig,
    agents: List[str],
) -> Dict[str, Dict[str, int]]:
    """Get outcomes for each (task, agent) pair.

    Args:
        config: Dataset configuration
        agents: List of agent names to check

    Returns:
        Dict[task_id, Dict[agent, 0|1]] mapping tasks to agent outcomes
    """
    responses = config.responses
    all_tasks = config.all_task_ids

    task_outcomes = {}
    for task_id in all_tasks:
        task_outcomes[task_id] = {}
        for agent in agents:
            if agent in responses and task_id in responses[agent]:
                task_outcomes[task_id][agent] = responses[agent][task_id]

    return task_outcomes


def analyze_fail_distribution(
    task_outcomes: Dict[str, Dict[str, int]],
    agents: List[str],
    frontier_tasks: Set[str],
) -> Tuple[Dict[int, List[str]], pd.DataFrame]:
    """Analyze how many agents fail on each non-frontier task.

    Args:
        task_outcomes: Dict[task_id, Dict[agent, 0|1]]
        agents: List of agent names
        frontier_tasks: Set of frontier task IDs to exclude

    Returns:
        Tuple of:
        - tasks_by_fail_count: Dict mapping fail_count -> list of task_ids
        - summary_df: DataFrame with summary statistics
    """
    n_agents = len(agents)
    tasks_by_fail_count: Dict[int, List[str]] = {i: [] for i in range(n_agents + 1)}

    for task_id, outcomes in task_outcomes.items():
        if task_id in frontier_tasks:
            continue

        # Count how many agents fail on this task
        fail_count = sum(
            1 for agent in agents
            if agent in outcomes and outcomes[agent] == 0
        )
        tasks_by_fail_count[fail_count].append(task_id)

    # Build summary
    summary_data = []
    for fail_count in range(n_agents + 1):
        tasks = tasks_by_fail_count[fail_count]
        summary_data.append({
            "fail_count": fail_count,
            "n_tasks": len(tasks),
            "percentage": len(tasks) / sum(len(v) for v in tasks_by_fail_count.values()) * 100,
        })

    summary_df = pd.DataFrame(summary_data)
    return tasks_by_fail_count, summary_df


def analyze_success_failure_features(
    raw_features_path: Path,
) -> pd.DataFrame:
    """Analyze feature differences between success and failure trajectories.

    Uses the existing behavioral features from raw_features_500tasks_6agents.csv
    which has 'resolved' column indicating success/failure.

    Args:
        raw_features_path: Path to raw features CSV

    Returns:
        DataFrame with feature comparison statistics
    """
    df = pd.read_csv(raw_features_path)

    # Behavioral features (numeric columns, excluding metadata)
    feature_cols = [
        "loop_detection",
        "localization_quality",
        "debugging_cycles",
        "error_recovery",
        "exploration_breadth",
        "focus_drift",
        "solution_completeness",
        "edge_case_handling",
        "test_verification",
        "trajectory_length",
    ]

    # Only keep features that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    results = []
    for col in feature_cols:
        success = df[df["resolved"] == True][col].dropna()
        failure = df[df["resolved"] == False][col].dropna()

        if len(success) < 5 or len(failure) < 5:
            continue

        # Compute statistics
        t_stat, p_value = stats.ttest_ind(success, failure)
        effect_size = (success.mean() - failure.mean()) / np.sqrt(
            (success.std()**2 + failure.std()**2) / 2
        )

        results.append({
            "feature": col,
            "success_mean": success.mean(),
            "success_std": success.std(),
            "failure_mean": failure.mean(),
            "failure_std": failure.std(),
            "diff": success.mean() - failure.mean(),
            "effect_size": effect_size,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        })

    return pd.DataFrame(results)


def main():
    config = SWEBenchConfig()

    print("=" * 60)
    print("Phase 1: Data Availability Analysis")
    print("=" * 60)

    # Get full-coverage agent names (the actual agent IDs)
    full_coverage_agents = list(FULL_COVERAGE_AGENTS.values())
    print(f"\n13 Full-coverage agents:")
    for short_name, full_name in FULL_COVERAGE_AGENTS.items():
        print(f"  - {short_name}: {full_name}")

    # Load frontier tasks
    print("\nIdentifying frontier tasks (zero_pre definition)...")
    all_agents = config.all_agents
    agent_dates = config.get_agent_dates(all_agents)
    pre_frontier, post_frontier = split_agents_by_dates(
        all_agents, agent_dates, config.cutoff_date
    )
    frontier_tasks = set(identify_frontier_tasks_zero_pre(
        config.responses_path, pre_frontier, post_frontier
    ))
    print(f"  Pre-frontier agents: {len(pre_frontier)}")
    print(f"  Post-frontier agents: {len(post_frontier)}")
    print(f"  Frontier tasks (zero_pre): {len(frontier_tasks)}")

    # Get outcomes for full-coverage agents
    print("\nAnalyzing task outcomes for 13 full-coverage agents...")
    task_outcomes = get_task_agent_outcomes(config, full_coverage_agents)

    # Analyze fail distribution
    print("\n" + "-" * 60)
    print("Section 1: Fail Distribution for Non-Frontier Tasks")
    print("-" * 60)
    tasks_by_fail_count, summary_df = analyze_fail_distribution(
        task_outcomes, full_coverage_agents, frontier_tasks
    )

    print("\nHow many agents fail on each non-frontier task?")
    print(summary_df.to_string(index=False))

    # Key statistics
    all_fail_tasks = tasks_by_fail_count[13]
    print(f"\n*** Tasks where ALL 13 agents fail: {len(all_fail_tasks)} ***")
    print(f"    This is the maximum training set size with complete feature vectors")

    # At least N agents fail
    for threshold in [10, 11, 12]:
        tasks_at_threshold = sum(
            len(tasks_by_fail_count[n]) for n in range(threshold, 14)
        )
        print(f"    Tasks where >= {threshold} agents fail: {tasks_at_threshold}")

    # Save all-fail task list
    output_dir = Path("chris_output/trajectory_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fail_path = output_dir / "all_fail_nonfrontier_tasks.json"
    with open(all_fail_path, "w") as f:
        json.dump(all_fail_tasks, f, indent=2)
    print(f"\nAll-fail task list saved to: {all_fail_path}")

    # Analyze success/failure feature differences
    print("\n" + "-" * 60)
    print("Section 2: Success vs Failure Feature Differences")
    print("-" * 60)

    raw_features_path = Path("chris_output/trajectory_features/raw_features_500tasks_6agents.csv")
    if raw_features_path.exists():
        feature_comparison = analyze_success_failure_features(raw_features_path)
        print("\nFeature differences between successful and failed trajectories:")
        print(feature_comparison.to_string(index=False))

        significant_features = feature_comparison[feature_comparison["significant"]]
        print(f"\n*** {len(significant_features)} features show significant differences ***")
        if len(significant_features) > 0:
            print("\nThis validates training on failures only - features carry outcome signal")
    else:
        print(f"\nWarning: Raw features file not found: {raw_features_path}")
        print("Skipping success/failure analysis")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  - Non-frontier tasks with all 13 agents failing: {len(all_fail_tasks)}")
    print(f"  - Frontier tasks (zero_pre): {len(frontier_tasks)}")
    print(f"  - Total tasks with complete feature vectors: {len(all_fail_tasks) + len(frontier_tasks)}")

    if len(all_fail_tasks) >= 50:
        print("\n*** Recommendation: Sufficient all-fail tasks for training ***")
        print("    Proceed with extracting features from all-fail non-frontier tasks")
    else:
        print("\n*** Warning: Few all-fail tasks ***")
        print("    Consider allowing partial failures with NaN handling")


if __name__ == "__main__":
    main()
