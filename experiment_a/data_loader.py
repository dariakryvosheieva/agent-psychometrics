"""Data loading and splitting for Experiment A.

To avoid data leakage, this module trains IRT only on train tasks, ensuring
the ground truth difficulties used for training are not contaminated by test
task information.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Container for all loaded data.

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
        responses: Full response matrix
        train_tasks: List of train task IDs (T1)
        test_tasks: List of test task IDs (T2)
        all_agents: List of all agent IDs
    """

    train_abilities: pd.DataFrame  # From IRT^train - USE FOR ALL METHODS
    train_items: pd.DataFrame  # From IRT^train - ground truth for training predictors
    full_abilities: pd.DataFrame  # From IRT^full - ONLY for oracle baseline
    full_items: pd.DataFrame  # From IRT^full - ONLY for oracle baseline
    responses: Dict[str, Dict[str, int]]
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


def load_experiment_data(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    test_fraction: float,
    split_seed: int,
    irt_cache_dir: Optional[Path] = None,
    force_retrain: bool = False,
) -> ExperimentAData:
    """Load all data with separate IRT models for methods vs oracle.

    This function:
    1. Loads full IRT parameters (ONLY for oracle baseline comparison)
    2. Splits tasks into train/test
    3. Trains (or loads cached) IRT model on train tasks only
    4. Returns data with both IRT models clearly separated

    IMPORTANT: All actual methods must use train_abilities and train_items.
    The full_abilities and full_items are ONLY for the oracle baseline.

    Args:
        abilities_path: Path to full IRT abilities.csv (for oracle only)
        items_path: Path to full IRT items.csv (for oracle only)
        responses_path: Path to response matrix JSONL
        test_fraction: Fraction of tasks for test set
        split_seed: Random seed for splits
        irt_cache_dir: Directory for cached split IRT models
        force_retrain: If True, retrain IRT even if cached

    Returns:
        ExperimentAData with separate train/full IRT parameters
    """
    from experiment_a.train_irt_split import get_or_train_split_irt

    # Load full IRT parameters (ONLY for oracle baseline - not for actual methods!)
    full_abilities = load_abilities(abilities_path)
    full_items = load_items(items_path)
    responses = load_responses(responses_path)

    # Get all task IDs from full items
    all_task_ids = list(full_items.index)

    # Create train/test split
    train_tasks, test_tasks = stable_split_tasks(
        all_task_ids, test_fraction, split_seed
    )

    # Get or train split IRT model
    if irt_cache_dir is None:
        # Default to chris_output/experiment_a/irt_splits
        irt_cache_dir = Path(__file__).parent.parent / "chris_output" / "experiment_a" / "irt_splits"

    split_irt_dir = get_or_train_split_irt(
        responses_path=responses_path,
        output_base=irt_cache_dir,
        test_fraction=test_fraction,
        split_seed=split_seed,
        model_type="1pl",
        force_retrain=force_retrain,
    )

    # Load train-only IRT parameters
    train_abilities = load_abilities(split_irt_dir / "abilities.csv")
    train_items = load_items(split_irt_dir / "items.csv")

    # Get agents that are in both abilities and responses
    all_agents = [a for a in full_abilities.index if a in responses]

    return ExperimentAData(
        train_abilities=train_abilities,
        train_items=train_items,
        full_abilities=full_abilities,
        full_items=full_items,
        responses=responses,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        all_agents=all_agents,
    )