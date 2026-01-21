"""Data splitting utilities for frontier task difficulty evaluation.

This module provides functions for:
- Splitting agents by date (generic, works with any date source)
- Computing pass rates per task for different agent groups
- Identifying frontier tasks (hard for pre-frontier, easier for post-frontier)
- Identifying nontrivial anchor tasks for scale alignment
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def split_agents_by_dates(
    agents: List[str],
    agent_dates: Dict[str, str],
    cutoff_date: str,
) -> Tuple[List[str], List[str]]:
    """Split agents into pre-frontier and post-frontier by date cutoff.

    This is a generic function that works with any source of agent dates
    (e.g., from agent name prefix for SWE-bench, or from metadata for TerminalBench).

    Args:
        agents: List of agent names
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD format)
        cutoff_date: Date string in YYYYMMDD format. Agents with date >= cutoff
                     are post-frontier, agents with date < cutoff are pre-frontier.

    Returns:
        Tuple of (pre_frontier_agents, post_frontier_agents)

    Raises:
        ValueError: If any agent is missing a date in agent_dates
    """
    pre_frontier = []
    post_frontier = []
    missing_dates = []

    for agent in agents:
        date = agent_dates.get(agent)
        if not date:
            missing_dates.append(agent)
            continue

        if date >= cutoff_date:
            post_frontier.append(agent)
        else:
            pre_frontier.append(agent)

    if missing_dates:
        raise ValueError(
            f"{len(missing_dates)} agents missing dates. "
            f"First 5: {missing_dates[:5]}"
        )

    return pre_frontier, post_frontier


def compute_pass_rates(
    responses_path: Path,
    agents: List[str],
) -> Dict[str, float]:
    """Compute empirical pass rate for each task among specified agents.

    Args:
        responses_path: Path to JSONL response matrix
        agents: List of agent names to include

    Returns:
        Dict mapping task_id -> pass_rate (0-1)
    """
    agent_set = set(agents)
    task_successes: Dict[str, List[int]] = defaultdict(list)

    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            if data["subject_id"] not in agent_set:
                continue
            for task_id, response in data["responses"].items():
                task_successes[task_id].append(response)

    return {
        task_id: sum(successes) / len(successes) if successes else 0.0
        for task_id, successes in task_successes.items()
    }


def identify_frontier_tasks(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
    pre_threshold: float = 0.1,
    post_threshold: float = 0.1,
) -> List[str]:
    """Identify frontier tasks: hard for pre-frontier, easier for post-frontier.

    Frontier tasks are those where:
    - Pass rate among pre-frontier agents <= pre_threshold (e.g., 10%)
    - Pass rate among post-frontier agents > post_threshold (e.g., 10%)

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names
        pre_threshold: Maximum pass rate for pre-frontier (default 0.1 = 10%)
        post_threshold: Minimum pass rate for post-frontier (default 0.1 = 10%)

    Returns:
        List of task_ids that are frontier tasks
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    frontier_tasks = []

    # Check for tasks in pre but not in post (and vice versa)
    pre_only = set(pre_pass_rates.keys()) - set(post_pass_rates.keys())
    post_only = set(post_pass_rates.keys()) - set(pre_pass_rates.keys())

    if pre_only:
        raise ValueError(
            f"{len(pre_only)} tasks have pre-frontier data but no post-frontier data. "
            f"First 5: {list(pre_only)[:5]}"
        )
    if post_only:
        raise ValueError(
            f"{len(post_only)} tasks have post-frontier data but no pre-frontier data. "
            f"First 5: {list(post_only)[:5]}"
        )

    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates[task_id]
        post_rate = post_pass_rates[task_id]

        if pre_rate <= pre_threshold and post_rate > post_threshold:
            frontier_tasks.append(task_id)

    return frontier_tasks


def identify_frontier_tasks_irt(
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    agent_dates: Dict[str, str],
    cutoff_date: str,
    solve_probability: float = 0.3,
) -> List[str]:
    """Identify frontier tasks using IRT probability threshold.

    A task is frontier if NO agent before the cutoff date has theta >= beta + logit(p)
    (i.e., no pre-frontier agent can solve it with at least p probability).

    This differs from identify_frontier_tasks() which uses empirical pass rates.
    The IRT-based definition is more principled since P(success) = sigmoid(theta - beta).

    Args:
        oracle_items: DataFrame with 'b' column (oracle task difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle agent abilities)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)
        cutoff_date: Cutoff date string (YYYYMMDD). Tasks where the first capable
                     agent appears ON or AFTER this date are frontier tasks.
        solve_probability: Probability threshold for considering an agent "capable"
            of solving a task (default 0.3, i.e., 30% solve rate)

    Returns:
        List of task_ids that are frontier tasks (excludes tasks with no capable agent)
    """
    from experiment_b.shared.date_forecasting import (
        compute_first_capable_dates,
        split_tasks_by_first_capable_date,
        parse_date,
    )

    # Compute first capable date for each task
    result = compute_first_capable_dates(
        oracle_items, oracle_abilities, agent_dates, solve_probability
    )

    # Split by cutoff: tasks where first capable agent is on/after cutoff are "frontier"
    cutoff_datetime = parse_date(cutoff_date)
    pre_cutoff_tasks, post_cutoff_tasks = split_tasks_by_first_capable_date(
        result.first_capable_dates, cutoff_datetime
    )

    # Exclude tasks with no capable agent (no ground truth for evaluation)
    # These are logged for visibility
    if result.tasks_without_capable_agent:
        logger.info(
            f"Excluding {len(result.tasks_without_capable_agent)} tasks with no capable agent "
            f"(no agent has >={solve_probability:.0%} solve probability)"
        )

    return post_cutoff_tasks


def identify_frontier_tasks_zero_pre(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
) -> List[str]:
    """Identify frontier tasks: zero pre-frontier solves, nonzero post-frontier solves.

    Frontier tasks are those where:
    - Pass rate among pre-frontier agents == 0 (no pre-frontier agent solves it)
    - Pass rate among post-frontier agents > 0 (at least one post-frontier agent solves it)

    This is a stricter criterion than identify_frontier_tasks() which allows up to
    10% pre-frontier pass rate.

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names

    Returns:
        List of task_ids that are frontier tasks
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    frontier_tasks = []
    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates[task_id]
        post_rate = post_pass_rates[task_id]

        if pre_rate == 0.0 and post_rate > 0.0:
            frontier_tasks.append(task_id)

    return frontier_tasks


def identify_nontrivial_tasks(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """Identify tasks with non-trivial pass rates in BOTH agent groups.

    Non-trivial tasks have meaningful variation - neither too easy nor too hard
    for both pre-frontier and post-frontier agents. These are useful as anchor
    tasks for aligning IRT scales.

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names
        min_pass_rate: Minimum pass rate threshold (default 0.10 = 10%)
        max_pass_rate: Maximum pass rate threshold (default 0.90 = 90%)

    Returns:
        Tuple of (nontrivial_task_ids, pre_pass_rates, post_pass_rates)
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    # Check for tasks in pre but not in post (and vice versa)
    pre_only = set(pre_pass_rates.keys()) - set(post_pass_rates.keys())
    post_only = set(post_pass_rates.keys()) - set(pre_pass_rates.keys())

    if pre_only:
        raise ValueError(
            f"{len(pre_only)} tasks have pre-frontier data but no post-frontier data. "
            f"First 5: {list(pre_only)[:5]}"
        )
    if post_only:
        raise ValueError(
            f"{len(post_only)} tasks have post-frontier data but no pre-frontier data. "
            f"First 5: {list(post_only)[:5]}"
        )

    nontrivial_tasks = []
    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates[task_id]
        post_rate = post_pass_rates[task_id]

        # Both groups must have meaningful variation
        pre_nontrivial = min_pass_rate <= pre_rate <= max_pass_rate
        post_nontrivial = min_pass_rate <= post_rate <= max_pass_rate

        if pre_nontrivial and post_nontrivial:
            nontrivial_tasks.append(task_id)

    return nontrivial_tasks, pre_pass_rates, post_pass_rates


def get_all_agents_from_responses(responses_path: Path) -> List[str]:
    """Get list of all agent IDs from response matrix.

    Args:
        responses_path: Path to JSONL response matrix

    Returns:
        List of agent IDs (subject_id values)
    """
    agents = []
    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            agents.append(data["subject_id"])
    return agents


def get_agents_with_trajectories(trajectories_dir: Path) -> Set[str]:
    """Get set of agents that have trajectory data.

    Args:
        trajectories_dir: Path to trajectory directory

    Returns:
        Set of agent names that have trajectory subdirectories
    """
    agents = set()
    for path in trajectories_dir.iterdir():
        if path.is_dir() and not path.name.startswith("_"):
            agents.add(path.name)
    return agents


def get_pre_frontier_agents(
    responses_path: Path,
    trajectories_dir: Path,
    cutoff_date: str = "20250807",
) -> Tuple[List[str], List[str]]:
    """Get pre-frontier and post-frontier agent lists for training/inference.

    This is the canonical function for determining agent lists. It ensures
    consistent ordering between training and inference by:
    1. Reading agents from response matrix (preserves JSONL line order)
    2. Filtering to agents with trajectories
    3. Splitting by cutoff date

    NOTE: This function is specific to SWE-bench agents that have date prefixes
    in their names. For other datasets, use split_agents_by_dates() directly
    with agent dates from the dataset config.

    Args:
        responses_path: Path to JSONL response matrix
        trajectories_dir: Path to trajectory directory
        cutoff_date: Date cutoff for pre/post frontier (YYYYMMDD format)

    Returns:
        Tuple of (pre_frontier_agents, post_frontier_agents)

    Example:
        >>> pre_frontier, post_frontier = get_pre_frontier_agents(
        ...     Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"),
        ...     Path("chris_output/trajectory_summaries_api"),
        ... )
        >>> print(f"Pre-frontier: {len(pre_frontier)} agents")
    """
    # Import here to avoid circular imports
    from experiment_b.datasets.swebench import extract_date_prefix

    # Get all agents in response matrix order (this order is preserved!)
    all_agents = get_all_agents_from_responses(responses_path)

    # Get agents with trajectories
    traj_agents = get_agents_with_trajectories(trajectories_dir)

    # Filter to agents with both (preserving response matrix order)
    agents_with_both = [a for a in all_agents if a in traj_agents]

    # Build agent_dates dict from name prefixes
    agent_dates = {
        agent: extract_date_prefix(agent)
        for agent in agents_with_both
    }

    # Split by cutoff date
    pre_frontier, post_frontier = split_agents_by_dates(
        agents_with_both, agent_dates, cutoff_date=cutoff_date
    )

    return pre_frontier, post_frontier


if __name__ == "__main__":
    # Import here since this is just for testing
    from experiment_b.datasets.swebench import extract_date_prefix

    # Test the splitting logic
    responses_path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")

    # Get all agents
    all_agents = get_all_agents_from_responses(responses_path)
    print(f"Total agents in response matrix: {len(all_agents)}")

    # Build agent_dates dict from name prefixes
    agent_dates = {agent: extract_date_prefix(agent) for agent in all_agents}

    # Split by cutoff date
    pre_frontier, post_frontier = split_agents_by_dates(
        all_agents, agent_dates, cutoff_date="20250807"
    )
    print(f"\nPre-frontier (< 20250807): {len(pre_frontier)} agents")
    print(f"Post-frontier (>= 20250807): {len(post_frontier)} agents")

    # Identify frontier tasks
    frontier_tasks = identify_frontier_tasks(
        responses_path,
        pre_frontier,
        post_frontier,
        pre_threshold=0.1,
        post_threshold=0.1,
    )
    print(f"\nFrontier tasks (<=10% pre, >10% post): {len(frontier_tasks)}")

    # Identify nontrivial anchor tasks
    nontrivial_tasks, pre_rates, post_rates = identify_nontrivial_tasks(
        responses_path,
        pre_frontier,
        post_frontier,
    )
    print(f"Nontrivial anchor tasks (10-90% in both): {len(nontrivial_tasks)}")

    # Show some examples
    if frontier_tasks:
        print("\nExample frontier tasks:")
        for task_id in frontier_tasks[:5]:
            pre_rate = pre_rates[task_id]
            post_rate = post_rates[task_id]
            print(f"  {task_id}: pre={pre_rate:.1%}, post={post_rate:.1%}")
