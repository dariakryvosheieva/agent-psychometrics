"""Data loading and splitting for Experiment A on TerminalBench.

To avoid data leakage, this module trains IRT only on train tasks, ensuring
the ground truth difficulties used for training are not contaminated by test
task information.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Reuse stable_split_tasks from experiment_a
from experiment_a.data_loader import stable_split_tasks


def load_abilities(abilities_path: Path) -> pd.DataFrame:
    """Load agent abilities from 1PL IRT model.

    Args:
        abilities_path: Path to abilities.csv

    Returns:
        DataFrame with index=agent_id, columns=['theta', 'theta_std']
    """
    df = pd.read_csv(abilities_path, index_col=0)
    return df


def load_items(items_path: Path) -> pd.DataFrame:
    """Load IRT item parameters (ground truth difficulties).

    Args:
        items_path: Path to items.csv

    Returns:
        DataFrame with index=task_id, columns=['b', 'b_std']
    """
    df = pd.read_csv(items_path, index_col=0)
    return df


def load_binomial_responses(responses_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load binomial response matrix from JSONL.

    Args:
        responses_path: Path to response matrix JSONL file with binomial data

    Returns:
        Dict mapping agent_id -> {task_id -> {successes: int, trials: int}}
    """
    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def load_task_data_from_repo(
    task_ids: List[str],
    repo_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load task instruction and solution from local terminal-bench repo.

    Args:
        task_ids: List of task IDs to load
        repo_path: Path to cloned terminal-bench repo

    Returns:
        Dict mapping task_id -> {instruction, solution, difficulty, category, tags}
    """
    tasks = {}
    for task_id in task_ids:
        task_dir = repo_path / "tasks" / task_id
        if not task_dir.exists():
            print(f"Warning: Task directory not found: {task_dir}")
            continue

        # Load task.yaml
        task_yaml_path = task_dir / "task.yaml"
        if task_yaml_path.exists():
            with open(task_yaml_path) as f:
                task_yaml = yaml.safe_load(f)
        else:
            task_yaml = {}

        # Load solution.sh
        solution_path = task_dir / "solution.sh"
        if solution_path.exists():
            solution = solution_path.read_text()
        else:
            solution = ""

        tasks[task_id] = {
            "instruction": task_yaml.get("instruction", ""),
            "solution": solution,
            "difficulty": task_yaml.get("difficulty"),
            "category": task_yaml.get("category"),
            "tags": task_yaml.get("tags", []),
        }

    return tasks


def load_task_list_from_items(
    items_path: Path,
    repo_path: Path,
) -> List[Dict[str, Any]]:
    """Load TerminalBench tasks as a list of dicts.

    Convenience function for scripts that need task data in list format
    (e.g., compute_llm_judge_features.py, generate_embeddings.py).

    Args:
        items_path: Path to 1PL items.csv (for task IDs)
        repo_path: Path to cloned terminal-bench repo

    Returns:
        List of task dicts with task_id, instruction, solution, category, tags, difficulty
    """
    # Load task IDs from items.csv
    items_df = pd.read_csv(items_path, index_col=0)
    task_ids = list(items_df.index)

    # Load task data from repo
    task_data = load_task_data_from_repo(task_ids, repo_path)

    # Convert to list of dicts
    tasks = []
    for task_id in task_ids:
        if task_id not in task_data:
            continue

        data = task_data[task_id]
        tasks.append({
            "task_id": task_id,
            "instruction": data["instruction"],
            "solution": data["solution"],
            "category": data.get("category", ""),
            "tags": data.get("tags", []),
            "claimed_difficulty": data.get("difficulty", ""),
        })

    return tasks


@dataclass
class TerminalBenchData:
    """Container for all loaded TerminalBench data.

    To avoid data leakage, we maintain two separate IRT models:

    - train_abilities, train_items: From IRT trained ONLY on train tasks (T1).
      These must be used for all actual methods (embedding, constant, etc.)
      to ensure no information from test tasks (T2) leaks into evaluation.

    - full_abilities, full_items: From IRT trained on ALL tasks (T1 ∪ T2).
      These are used ONLY for the oracle baseline, which serves as a
      reference point showing theoretical best performance. The oracle
      is not a valid method - it's just for comparison.

    Attributes:
        train_abilities: Agent abilities from IRT^train (use for all methods)
        train_items: Task difficulties from IRT^train (use for training predictors)
        full_abilities: Agent abilities from IRT^full (ONLY for oracle)
        full_items: Task difficulties from IRT^full (ONLY for oracle)
        responses: Full binomial response matrix
        task_data: Task metadata (instruction, solution, etc.)
        train_tasks: List of train task IDs (T1)
        test_tasks: List of test task IDs (T2)
        all_agents: List of all agent IDs
    """

    train_abilities: pd.DataFrame  # From IRT^train - USE FOR ALL METHODS
    train_items: pd.DataFrame  # From IRT^train - ground truth for training predictors
    full_abilities: pd.DataFrame  # From IRT^full - ONLY for oracle baseline
    full_items: pd.DataFrame  # From IRT^full - ONLY for oracle baseline
    responses: Dict[str, Dict[str, Dict[str, int]]]  # agent_id -> {task_id -> {successes, trials}}
    task_data: Dict[str, Dict[str, Any]]  # task_id -> {instruction, solution, ...}
    train_tasks: List[str]
    test_tasks: List[str]
    all_agents: List[str]

    @property
    def n_agents(self) -> int:
        return len(self.all_agents)

    @property
    def n_tasks(self) -> int:
        return len(self.train_tasks) + len(self.test_tasks)

    @property
    def n_train_tasks(self) -> int:
        return len(self.train_tasks)

    @property
    def n_test_tasks(self) -> int:
        return len(self.test_tasks)


def load_terminalbench_data(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    repo_path: Path,
    test_fraction: float,
    split_seed: int,
    irt_cache_dir: Optional[Path] = None,
    force_retrain: bool = False,
) -> TerminalBenchData:
    """Load all TerminalBench data with separate IRT models for methods vs oracle.

    This function:
    1. Loads full IRT parameters (ONLY for oracle baseline comparison)
    2. Splits tasks into train/test
    3. Trains (or loads cached) binomial IRT model on train tasks only
    4. Returns data with both IRT models clearly separated

    IMPORTANT: All actual methods must use train_abilities and train_items.
    The full_abilities and full_items are ONLY for the oracle baseline.

    Args:
        abilities_path: Path to full IRT abilities.csv (for oracle only)
        items_path: Path to full IRT items.csv (for oracle only)
        responses_path: Path to binomial response matrix JSONL
        repo_path: Path to cloned terminal-bench repo
        test_fraction: Fraction of tasks for test set
        split_seed: Random seed for splits
        irt_cache_dir: Directory for cached split IRT models
        force_retrain: If True, retrain IRT even if cached

    Returns:
        TerminalBenchData with separate train/full IRT parameters
    """
    from experiment_a.train_irt_split import get_or_train_split_irt

    # Load full IRT parameters (ONLY for oracle baseline - not for actual methods!)
    full_abilities = load_abilities(abilities_path)
    full_items = load_items(items_path)
    responses = load_binomial_responses(responses_path)

    # Get all task IDs from full items
    all_task_ids = list(full_items.index)

    # Load task data from repo
    task_data = load_task_data_from_repo(all_task_ids, repo_path)

    # Create train/test split on tasks (reusing from experiment_a)
    train_tasks, test_tasks = stable_split_tasks(
        all_task_ids, test_fraction, split_seed
    )

    # Get or train split IRT model (binomial for TerminalBench)
    if irt_cache_dir is None:
        # Default to chris_output/experiment_a_terminalbench/irt_splits
        irt_cache_dir = Path(__file__).parent.parent / "chris_output" / "experiment_a_terminalbench" / "irt_splits"

    split_irt_dir = get_or_train_split_irt(
        responses_path=responses_path,
        output_base=irt_cache_dir,
        test_fraction=test_fraction,
        split_seed=split_seed,
        model_type="1pl",
        force_retrain=force_retrain,
        is_binomial=True,  # TerminalBench uses binomial IRT
    )

    # Load train-only IRT parameters
    train_abilities = load_abilities(split_irt_dir / "abilities.csv")
    train_items = load_items(split_irt_dir / "items.csv")

    # Get agents that are in both full abilities and responses
    all_agents = [a for a in full_abilities.index if a in responses]

    return TerminalBenchData(
        train_abilities=train_abilities,
        train_items=train_items,
        full_abilities=full_abilities,
        full_items=full_items,
        responses=responses,
        task_data=task_data,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        all_agents=all_agents,
    )
