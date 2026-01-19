"""Data splitting utilities for SAD-IRT frontier difficulty evaluation."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def extract_date_prefix(agent_name: str) -> str:
    """Extract YYYYMMDD date prefix from agent name.

    Args:
        agent_name: Agent name like '20240620_sweagent_claude3.5sonnet'

    Returns:
        Date string like '20240620' or empty string if no valid prefix
    """
    match = re.match(r"^(\d{8})_", agent_name)
    if match:
        return match.group(1)
    return ""


def split_agents_by_cutoff(
    agents: List[str],
    cutoff_date: str = "20250807",
) -> Tuple[List[str], List[str]]:
    """Split agents into pre-frontier and post-frontier by date cutoff.

    Args:
        agents: List of agent names
        cutoff_date: Date string in YYYYMMDD format. Agents with date >= cutoff
                     are post-frontier, agents with date < cutoff are pre-frontier.

    Returns:
        Tuple of (pre_frontier_agents, post_frontier_agents)
    """
    pre_frontier = []
    post_frontier = []

    for agent in agents:
        date_prefix = extract_date_prefix(agent)
        if not date_prefix:
            # No valid date prefix, skip this agent
            continue

        if date_prefix >= cutoff_date:
            post_frontier.append(agent)
        else:
            pre_frontier.append(agent)

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
    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates.get(task_id, 0.0)
        post_rate = post_pass_rates.get(task_id, 0.0)

        if pre_rate <= pre_threshold and post_rate > post_threshold:
            frontier_tasks.append(task_id)

    return frontier_tasks


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

    The order of agents in the returned lists matches what the SAD-IRT model
    expects for theta embeddings.

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
    # Get all agents in response matrix order (this order is preserved!)
    all_agents = get_all_agents_from_responses(responses_path)

    # Get agents with trajectories
    traj_agents = get_agents_with_trajectories(trajectories_dir)

    # Filter to agents with both (preserving response matrix order)
    agents_with_both = [a for a in all_agents if a in traj_agents]

    # Split by cutoff date
    pre_frontier, post_frontier = split_agents_by_cutoff(
        agents_with_both, cutoff_date=cutoff_date
    )

    return pre_frontier, post_frontier


if __name__ == "__main__":
    # Test the splitting logic
    responses_path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    trajectories_dir = Path("chris_output/trajectory_summaries_api")

    # Get all agents
    all_agents = get_all_agents_from_responses(responses_path)
    print(f"Total agents in response matrix: {len(all_agents)}")

    # Get agents with trajectories
    traj_agents = get_agents_with_trajectories(trajectories_dir)
    print(f"Agents with trajectories: {len(traj_agents)}")

    # Filter to agents with both responses and trajectories
    agents_with_both = [a for a in all_agents if a in traj_agents]
    print(f"Agents with both: {len(agents_with_both)}")

    # Split by cutoff date
    pre_frontier, post_frontier = split_agents_by_cutoff(agents_with_both, cutoff_date="20250807")
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

    # Show some examples
    if frontier_tasks:
        pre_rates = compute_pass_rates(responses_path, pre_frontier)
        post_rates = compute_pass_rates(responses_path, post_frontier)
        print("\nExample frontier tasks:")
        for task_id in frontier_tasks[:5]:
            print(f"  {task_id}: pre={pre_rates[task_id]:.1%}, post={post_rates[task_id]:.1%}")
