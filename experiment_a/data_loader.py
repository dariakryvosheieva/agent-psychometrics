"""Data loading and splitting for Experiment A."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


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


def load_responses(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load response matrix from JSONL.

    Args:
        responses_path: Path to response matrix JSONL file

    Returns:
        Dict mapping agent_id -> {task_id -> 0|1}
    """
    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def stable_split_tasks(
    task_ids: List[str],
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split on tasks using hash-based splitting.

    This reuses the same logic as Daria's stable_split_ids() from
    predict_question_difficulty.py to ensure consistent splits.

    Args:
        task_ids: List of task identifiers
        test_fraction: Fraction of tasks for test set (0 < x < 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_task_ids, test_task_ids)
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1")

    # Compute hash-based scores for each task
    scored: List[Tuple[float, str]] = []
    for task_id in task_ids:
        h = hashlib.md5((str(task_id) + f"::{seed}").encode("utf-8")).hexdigest()
        score = int(h[:8], 16) / float(16**8)
        scored.append((score, task_id))

    # Sort by score
    scored.sort()

    # Split based on test_fraction
    n_test = int(round(len(task_ids) * float(test_fraction)))
    test_tasks = [task_id for _, task_id in scored[:n_test]]
    train_tasks = [task_id for _, task_id in scored[n_test:]]

    return train_tasks, test_tasks


@dataclass
class ExperimentAData:
    """Container for all loaded data."""

    abilities: pd.DataFrame  # Agent abilities (theta), index=agent_id
    items: pd.DataFrame  # Ground truth difficulties (b), index=task_id
    responses: Dict[str, Dict[str, int]]  # agent_id -> {task_id -> 0|1}
    train_tasks: List[str]
    test_tasks: List[str]
    all_agents: List[str]

    @property
    def n_agents(self) -> int:
        return len(self.all_agents)

    @property
    def n_tasks(self) -> int:
        return len(self.items)

    @property
    def n_train_tasks(self) -> int:
        return len(self.train_tasks)

    @property
    def n_test_tasks(self) -> int:
        return len(self.test_tasks)


def load_experiment_data(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    test_fraction: float,
    split_seed: int,
) -> ExperimentAData:
    """Load all data and create train/test splits.

    Args:
        abilities_path: Path to 1PL abilities.csv
        items_path: Path to 1PL items.csv
        responses_path: Path to response matrix JSONL
        test_fraction: Fraction of tasks for test set
        split_seed: Random seed for splits

    Returns:
        ExperimentAData with all loaded data and splits
    """
    abilities = load_abilities(abilities_path)
    items = load_items(items_path)
    responses = load_responses(responses_path)

    # Get all task IDs from items (ground truth)
    all_task_ids = list(items.index)

    # Create train/test split on tasks
    train_tasks, test_tasks = stable_split_tasks(
        all_task_ids, test_fraction, split_seed
    )

    # Get agents that are in both abilities and responses
    all_agents = [a for a in abilities.index if a in responses]

    return ExperimentAData(
        abilities=abilities,
        items=items,
        responses=responses,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        all_agents=all_agents,
    )
