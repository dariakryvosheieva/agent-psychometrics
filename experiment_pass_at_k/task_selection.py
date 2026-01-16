"""Task selection for Pass@K experiment.

Selects tasks where:
1. Strong reasoning models typically fail (low pass rate among top agents)
2. IRT difficulty varies (sample from different difficulty quintiles)
3. Not completely impossible (some agents solve them)
4. From diverse repositories
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from experiment_pass_at_k.config import ExperimentPassKConfig


def load_response_matrix(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load response matrix from JSONL file.

    Returns:
        Dict mapping agent_id -> {task_id -> 0/1}
    """
    responses = {}
    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            responses[data["subject_id"]] = data["responses"]
    return responses


def compute_task_pass_rates(
    responses: Dict[str, Dict[str, int]],
    agent_filter: List[str] = None,
) -> Dict[str, float]:
    """Compute pass rate for each task.

    Args:
        responses: Response matrix
        agent_filter: If provided, only include these agents

    Returns:
        Dict mapping task_id -> pass_rate (0-1)
    """
    task_successes: Dict[str, List[int]] = defaultdict(list)

    for agent_id, agent_responses in responses.items():
        if agent_filter and agent_id not in agent_filter:
            continue
        for task_id, success in agent_responses.items():
            task_successes[task_id].append(success)

    return {
        task_id: sum(successes) / len(successes) if successes else 0.0
        for task_id, successes in task_successes.items()
    }


def get_top_agents(responses: Dict[str, Dict[str, int]], top_fraction: float = 0.2) -> List[str]:
    """Get the top-performing agents by overall pass rate.

    Args:
        responses: Response matrix
        top_fraction: Fraction of agents to consider "top" (default 20%)

    Returns:
        List of top agent IDs
    """
    agent_pass_rates = {}
    for agent_id, agent_responses in responses.items():
        successes = list(agent_responses.values())
        agent_pass_rates[agent_id] = sum(successes) / len(successes) if successes else 0.0

    # Sort by pass rate descending
    sorted_agents = sorted(agent_pass_rates.items(), key=lambda x: x[1], reverse=True)
    n_top = max(1, int(len(sorted_agents) * top_fraction))

    return [agent_id for agent_id, _ in sorted_agents[:n_top]]


def select_tasks(
    items_df: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    n_tasks: int = 5,
    max_top_agent_pass_rate: float = 0.3,
    min_overall_pass_rate: float = 0.05,
) -> List[Dict]:
    """Select tasks for the experiment.

    Selection criteria:
    1. Top agents fail on them (pass rate < max_top_agent_pass_rate)
    2. Not completely impossible (overall pass rate > min_overall_pass_rate)
    3. IRT difficulty varies (sample from different quintiles)
    4. Different repositories

    Args:
        items_df: IRT items with difficulty (b)
        responses: Response matrix
        n_tasks: Number of tasks to select
        max_top_agent_pass_rate: Max pass rate among top agents
        min_overall_pass_rate: Min overall pass rate

    Returns:
        List of task dicts with task_id, difficulty, pass rates, etc.
    """
    # Get top agents
    top_agents = get_top_agents(responses, top_fraction=0.2)
    print(f"Top agents ({len(top_agents)}): {top_agents[:5]}...")

    # Compute pass rates
    overall_pass_rates = compute_task_pass_rates(responses)
    top_agent_pass_rates = compute_task_pass_rates(responses, agent_filter=top_agents)

    # Filter tasks
    candidate_tasks = []
    for task_id in items_df.index:
        overall_rate = overall_pass_rates.get(task_id, 0)
        top_rate = top_agent_pass_rates.get(task_id, 0)
        difficulty = items_df.loc[task_id, "b"]

        # Check criteria
        if top_rate <= max_top_agent_pass_rate and overall_rate >= min_overall_pass_rate:
            repo = task_id.split("__")[0]
            candidate_tasks.append({
                "task_id": task_id,
                "difficulty": float(difficulty),
                "overall_pass_rate": overall_rate,
                "top_agent_pass_rate": top_rate,
                "repo": repo,
            })

    print(f"Candidate tasks meeting criteria: {len(candidate_tasks)}")

    if len(candidate_tasks) < n_tasks:
        print(f"Warning: Only {len(candidate_tasks)} candidates, need {n_tasks}")
        # Relax criteria
        candidate_tasks = []
        for task_id in items_df.index:
            overall_rate = overall_pass_rates.get(task_id, 0)
            top_rate = top_agent_pass_rates.get(task_id, 0)
            difficulty = items_df.loc[task_id, "b"]
            repo = task_id.split("__")[0]
            candidate_tasks.append({
                "task_id": task_id,
                "difficulty": float(difficulty),
                "overall_pass_rate": overall_rate,
                "top_agent_pass_rate": top_rate,
                "repo": repo,
            })
        # Sort by top_agent_pass_rate (ascending) to get hardest for top agents
        candidate_tasks.sort(key=lambda x: x["top_agent_pass_rate"])

    # Sort candidates by difficulty
    candidate_tasks.sort(key=lambda x: x["difficulty"])

    # Sample from different difficulty quintiles for diversity
    n_candidates = len(candidate_tasks)
    quintile_size = n_candidates // 5

    selected = []
    repos_used = set()

    for quintile in range(5):
        if len(selected) >= n_tasks:
            break

        start_idx = quintile * quintile_size
        end_idx = start_idx + quintile_size if quintile < 4 else n_candidates
        quintile_tasks = candidate_tasks[start_idx:end_idx]

        # Prefer tasks from unused repos
        for task in quintile_tasks:
            if len(selected) >= n_tasks:
                break
            if task["repo"] not in repos_used or len(repos_used) >= 4:
                selected.append(task)
                repos_used.add(task["repo"])
                break

    # If we still need more, just take the hardest remaining
    if len(selected) < n_tasks:
        remaining = [t for t in candidate_tasks if t not in selected]
        remaining.sort(key=lambda x: x["top_agent_pass_rate"])
        for task in remaining:
            if len(selected) >= n_tasks:
                break
            selected.append(task)

    return selected


def main():
    """Run task selection and save results."""
    config = ExperimentPassKConfig()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading IRT items from {config.items_path}")
    items_df = pd.read_csv(config.items_path, index_col=0)
    print(f"Loaded {len(items_df)} tasks with IRT parameters")

    print(f"\nLoading response matrix from {config.responses_path}")
    responses = load_response_matrix(config.responses_path)
    print(f"Loaded responses for {len(responses)} agents")

    # Select tasks
    print(f"\nSelecting {config.n_tasks} tasks...")
    selected_tasks = select_tasks(
        items_df=items_df,
        responses=responses,
        n_tasks=config.n_tasks,
        max_top_agent_pass_rate=config.max_pass_rate_for_selection,
        min_overall_pass_rate=config.min_overall_pass_rate,
    )

    # Print selection
    print(f"\nSelected {len(selected_tasks)} tasks:")
    print("-" * 80)
    for i, task in enumerate(selected_tasks, 1):
        print(f"{i}. {task['task_id']}")
        print(f"   Difficulty (b): {task['difficulty']:.3f}")
        print(f"   Overall pass rate: {task['overall_pass_rate']:.1%}")
        print(f"   Top agent pass rate: {task['top_agent_pass_rate']:.1%}")
        print(f"   Repo: {task['repo']}")

    # Save selection
    output_file = config.output_dir / "selected_tasks.json"
    output_data = {
        "config": config.to_dict(),
        "n_tasks": len(selected_tasks),
        "tasks": selected_tasks,
        "task_ids": [t["task_id"] for t in selected_tasks],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved selection to {output_file}")


if __name__ == "__main__":
    main()
