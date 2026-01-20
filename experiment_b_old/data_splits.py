"""Split agents and tasks for Experiment B training/validation."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ExperimentSplit:
    """Container for experiment data splits."""

    m1_agents: List[str]  # Oldest 40%
    m2_agents: List[str]  # Middle 40%
    m3_agents: List[str]  # Newest 20%
    d_train_tasks: List[str]  # Tasks for training
    d_valid_tasks: List[str]  # Tasks for validation (disjoint from d_train)
    # Pass rates for debugging
    m1_pass_rates: Dict[str, float]
    m2_pass_rates: Dict[str, float]
    m3_pass_rates: Dict[str, float]


def extract_submission_date(agent_name: str) -> Optional[datetime]:
    """Extract submission date from agent name prefix.

    Args:
        agent_name: Agent name like '20240620_sweagent_claude3.5sonnet'

    Returns:
        datetime object or None if no valid date prefix found
    """
    # Match YYYYMMDD at start of string
    match = re.match(r"^(\d{8})_", agent_name)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            return None
    return None


def split_agents_by_date(
    agents: List[str],
    m1_fraction: float = 0.4,
    m2_fraction: float = 0.4,
) -> Tuple[List[str], List[str], List[str], datetime, datetime]:
    """Split agents into M1, M2, M3 groups by submission date.

    Args:
        agents: List of agent names
        m1_fraction: Fraction of oldest agents for M1 (default 40%)
        m2_fraction: Fraction of middle agents for M2 (default 40%)

    Returns:
        Tuple of (m1_agents, m2_agents, m3_agents, t1_cutoff, t2_cutoff)
    """
    # Extract dates and sort
    agent_dates = []
    for agent in agents:
        date = extract_submission_date(agent)
        if date:
            agent_dates.append((agent, date))

    # Sort by date (oldest first)
    agent_dates.sort(key=lambda x: x[1])

    n = len(agent_dates)
    n_m1 = int(n * m1_fraction)
    n_m2 = int(n * m2_fraction)

    m1 = [a[0] for a in agent_dates[:n_m1]]
    m2 = [a[0] for a in agent_dates[n_m1 : n_m1 + n_m2]]
    m3 = [a[0] for a in agent_dates[n_m1 + n_m2 :]]

    # Get cutoff dates
    t1_cutoff = agent_dates[n_m1 - 1][1] if n_m1 > 0 else None
    t2_cutoff = agent_dates[n_m1 + n_m2 - 1][1] if n_m1 + n_m2 > 0 else None

    return m1, m2, m3, t1_cutoff, t2_cutoff


def compute_empirical_pass_rates(
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


def select_tasks_by_pass_rate(
    pass_rates_weak: Dict[str, float],
    pass_rates_strong: Dict[str, float],
    weak_threshold: float = 0.2,
    strong_min_improvement: float = 0.1,
) -> List[str]:
    """Select tasks where weak models struggle but strong models improve.

    Args:
        pass_rates_weak: Pass rates for weak model group
        pass_rates_strong: Pass rates for strong model group
        weak_threshold: Max pass rate for weak group (default 20%)
        strong_min_improvement: Min improvement for strong group

    Returns:
        List of task_ids meeting criteria
    """
    selected = []
    for task_id in pass_rates_weak:
        weak_rate = pass_rates_weak.get(task_id, 0)
        strong_rate = pass_rates_strong.get(task_id, 0)

        if weak_rate <= weak_threshold and strong_rate > weak_rate + strong_min_improvement:
            selected.append(task_id)

    return selected


def get_agents_with_trajectories(trajectories_dir: Path) -> Set[str]:
    """Get set of agents that have trajectory data."""
    agents = set()
    for path in trajectories_dir.iterdir():
        if path.is_dir() and not path.name.startswith("_"):
            agents.add(path.name)
    return agents


def create_experiment_split(
    responses_path: Path,
    trajectories_dir: Path,
    weak_threshold: float = 0.2,
    strong_min_improvement: float = 0.1,
    m1_fraction: float = 0.4,
    m2_fraction: float = 0.4,
) -> ExperimentSplit:
    """Create the full experiment split.

    D_train: Tasks hard for M1 but easier for M2
    D_valid: Tasks hard for M2 but easier for M3 (disjoint from D_train)
    """
    # Load all agents from responses
    response_agents = set()
    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            response_agents.add(data["subject_id"])

    # Get agents with trajectories
    traj_agents = get_agents_with_trajectories(trajectories_dir)

    # Use only agents in both
    agents = list(response_agents & traj_agents)
    print(f"Agents in response matrix: {len(response_agents)}")
    print(f"Agents with trajectories: {len(traj_agents)}")
    print(f"Agents in both: {len(agents)}")

    # Split agents by date
    m1, m2, m3, t1, t2 = split_agents_by_date(agents, m1_fraction, m2_fraction)
    print(f"M1 (oldest): {len(m1)} agents, cutoff: {t1}")
    print(f"M2 (middle): {len(m2)} agents, cutoff: {t2}")
    print(f"M3 (newest): {len(m3)} agents")

    # Compute pass rates
    pr_m1 = compute_empirical_pass_rates(responses_path, m1)
    pr_m2 = compute_empirical_pass_rates(responses_path, m2)
    pr_m3 = compute_empirical_pass_rates(responses_path, m3)

    # Select training tasks (hard for M1, easier for M2)
    d_train = select_tasks_by_pass_rate(pr_m1, pr_m2, weak_threshold, strong_min_improvement)
    print(f"D_train candidates (hard for M1, easier for M2): {len(d_train)}")

    # Select validation tasks (hard for M2, easier for M3)
    d_valid_candidates = select_tasks_by_pass_rate(pr_m2, pr_m3, weak_threshold, strong_min_improvement)
    print(f"D_valid candidates (hard for M2, easier for M3): {len(d_valid_candidates)}")

    # Ensure disjoint
    d_train_set = set(d_train)
    d_valid = [t for t in d_valid_candidates if t not in d_train_set]
    print(f"D_valid (after removing overlap): {len(d_valid)}")

    return ExperimentSplit(
        m1_agents=m1,
        m2_agents=m2,
        m3_agents=m3,
        d_train_tasks=d_train,
        d_valid_tasks=d_valid,
        m1_pass_rates=pr_m1,
        m2_pass_rates=pr_m2,
        m3_pass_rates=pr_m3,
    )


if __name__ == "__main__":
    # Test the splitting logic
    from experiment_b.config import ExperimentConfig

    config = ExperimentConfig()
    split = create_experiment_split(
        responses_path=config.responses_path,
        trajectories_dir=config.trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )
    print(f"\nFinal split:")
    print(f"  M1: {len(split.m1_agents)} agents")
    print(f"  M2: {len(split.m2_agents)} agents")
    print(f"  M3: {len(split.m3_agents)} agents")
    print(f"  D_train: {len(split.d_train_tasks)} tasks")
    print(f"  D_valid: {len(split.d_valid_tasks)} tasks")
