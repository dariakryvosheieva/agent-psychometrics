"""Abstract dataset interface for Experiment A across different benchmarks.

This module provides:
- ExperimentData: Abstract base class for datasets
- BinaryExperimentData: For binary outcomes (SWE-bench)
- BinomialExperimentData: For binomial outcomes (TerminalBench)
- load_dataset: Factory function to load any dataset type
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

# Response type: int for binary, Dict for binomial
ResponseT = TypeVar("ResponseT")


def filter_unsolved_tasks(
    task_ids: List[str],
    responses: Dict[str, Dict[str, Any]],
    is_binomial: bool = False,
) -> Tuple[List[str], int]:
    """Filter out tasks where no agent achieved any success.

    Args:
        task_ids: List of all task IDs to filter
        responses: Response matrix {agent_id: {task_id: response}}
        is_binomial: If True, responses are {successes: k, trials: n}
                     If False, responses are binary 0/1

    Returns:
        Tuple of (filtered_task_ids, n_excluded)
    """
    solved_tasks = []
    for task_id in task_ids:
        task_solved = False
        for agent_responses in responses.values():
            if task_id not in agent_responses:
                continue
            response = agent_responses[task_id]
            if is_binomial:
                if response.get("successes", 0) > 0:
                    task_solved = True
                    break
            else:
                if response == 1:
                    task_solved = True
                    break
        if task_solved:
            solved_tasks.append(task_id)

    n_excluded = len(task_ids) - len(solved_tasks)
    return solved_tasks, n_excluded


def stable_split_tasks(
    task_ids: List[str],
    test_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split on tasks using hash-based splitting.

    Args:
        task_ids: List of task identifiers
        test_fraction: Fraction of tasks for test set (0 < x < 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_task_ids, test_task_ids)
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1")

    scored: List[Tuple[float, str]] = []
    for task_id in task_ids:
        h = hashlib.md5((str(task_id) + f"::{seed}").encode("utf-8")).hexdigest()
        score = int(h[:8], 16) / float(16**8)
        scored.append((score, task_id))

    scored.sort()
    n_test = int(round(len(task_ids) * float(test_fraction)))
    test_tasks = [task_id for _, task_id in scored[:n_test]]
    train_tasks = [task_id for _, task_id in scored[n_test:]]

    return train_tasks, test_tasks


@dataclass
class ExperimentData(ABC, Generic[ResponseT]):
    """Abstract dataset supporting both binary and binomial responses.

    This is the core abstraction that allows the same evaluation code
    to work with different data formats (binary vs binomial IRT).
    """

    # Response matrix: agent_id -> task_id -> response
    responses: Dict[str, Dict[str, ResponseT]]

    # IRT parameters from train-only model (used for all methods)
    train_abilities: pd.DataFrame  # index=agent_id, columns include 'ability'
    train_items: pd.DataFrame      # index=task_id, columns include 'b' (difficulty)

    # IRT parameters from full model (oracle only)
    full_abilities: pd.DataFrame
    full_items: pd.DataFrame

    # Train/test split
    train_tasks: List[str]
    test_tasks: List[str]

    # Optional metadata (e.g., task descriptions for TerminalBench)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        return len(self.train_abilities)

    @property
    def n_tasks(self) -> int:
        return len(self.train_tasks) + len(self.test_tasks)

    @property
    def n_train_tasks(self) -> int:
        return len(self.train_tasks)

    @property
    def n_test_tasks(self) -> int:
        return len(self.test_tasks)

    @abstractmethod
    def expand_for_auc(
        self, agent_id: str, task_id: str, prob: float
    ) -> Tuple[List[int], List[float]]:
        """Expand a single (agent, task) response to binary observations for AUC.

        Args:
            agent_id: The agent identifier
            task_id: The task identifier
            prob: Predicted probability of success

        Returns:
            Tuple of (y_true, y_scores) where:
            - y_true: List of binary outcomes (0 or 1)
            - y_scores: List of predicted probabilities (all same value)

        For binary data: returns ([0] or [1], [prob])
        For binomial data: returns ([1]*k + [0]*(n-k), [prob]*n)
        """
        pass

    def get_train_difficulties(self) -> np.ndarray:
        """Get ground truth difficulties for training tasks (from train IRT)."""
        return self.train_items.loc[self.train_tasks, "b"].values

    def get_all_agents(self) -> List[str]:
        """Get list of all agent IDs."""
        return list(self.train_abilities.index)


@dataclass
class BinaryExperimentData(ExperimentData[int]):
    """Dataset for binary outcomes (0/1 success per agent-task pair).

    Used for SWE-bench where each agent gets one attempt per task.
    """

    def expand_for_auc(
        self, agent_id: str, task_id: str, prob: float
    ) -> Tuple[List[int], List[float]]:
        """Binary: single observation per (agent, task) pair."""
        actual = self.responses[agent_id][task_id]
        return [int(actual)], [prob]


@dataclass
class BinomialExperimentData(ExperimentData[Dict[str, int]]):
    """Dataset for binomial outcomes (k successes out of n trials).

    Used for TerminalBench where agents may have multiple trials per task.
    Response format: {"successes": k, "trials": n}
    """

    def expand_for_auc(
        self, agent_id: str, task_id: str, prob: float
    ) -> Tuple[List[int], List[float]]:
        """Binomial: expand to individual binary observations."""
        resp = self.responses[agent_id][task_id]
        k = resp["successes"]
        n = resp["trials"]
        # k successes (y=1) and (n-k) failures (y=0), all with same predicted prob
        y_true = [1] * k + [0] * (n - k)
        y_scores = [prob] * n
        return y_true, y_scores


def _load_abilities(abilities_path: Path) -> pd.DataFrame:
    """Load agent abilities from IRT model."""
    df = pd.read_csv(abilities_path, index_col=0)
    # Standardize column name to 'ability'
    if "theta" in df.columns:
        df = df.rename(columns={"theta": "ability"})
    return df


def _load_items(items_path: Path) -> pd.DataFrame:
    """Load IRT item parameters (difficulties)."""
    return pd.read_csv(items_path, index_col=0)


def _load_binary_responses(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load binary response matrix from JSONL."""
    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def _load_binomial_responses(responses_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load binomial response matrix from JSONL."""
    responses = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def load_dataset(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    test_fraction: float,
    split_seed: int,
    is_binomial: bool = False,
    irt_cache_dir: Optional[Path] = None,
    force_retrain: bool = False,
    metadata_loader: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    exclude_unsolved: bool = False,
) -> ExperimentData:
    """Load a dataset with proper train/test split and IRT models.

    This is the unified loader that works for both binary (SWE-bench) and
    binomial (TerminalBench) datasets.

    Args:
        abilities_path: Path to full IRT abilities.csv (for oracle only)
        items_path: Path to full IRT items.csv (for oracle only)
        responses_path: Path to response matrix JSONL
        test_fraction: Fraction of tasks for test set
        split_seed: Random seed for splits
        is_binomial: If True, use binomial IRT and BinomialExperimentData
        irt_cache_dir: Directory for cached split IRT models
        force_retrain: If True, retrain IRT even if cached
        metadata_loader: Optional function to load task metadata (task_ids -> metadata dict)
        exclude_unsolved: If True, filter out tasks no agent solved before splitting

    Returns:
        BinaryExperimentData or BinomialExperimentData
    """
    from experiment_a.train_irt_split import get_or_train_split_irt

    # Load full IRT parameters (ONLY for oracle baseline)
    full_abilities = _load_abilities(abilities_path)
    full_items = _load_items(items_path)

    # Load responses (binary or binomial)
    if is_binomial:
        responses = _load_binomial_responses(responses_path)
    else:
        responses = _load_binary_responses(responses_path)

    # Get all task IDs
    all_task_ids = list(full_items.index)

    # Optionally filter unsolved tasks before splitting
    n_excluded = 0
    if exclude_unsolved:
        all_task_ids, n_excluded = filter_unsolved_tasks(all_task_ids, responses, is_binomial)
        print(f"   Excluded {n_excluded} unsolved tasks ({len(all_task_ids)} remaining)")

    # Create train/test split
    train_tasks, test_tasks = stable_split_tasks(all_task_ids, test_fraction, split_seed)

    # Get or train split IRT model
    if irt_cache_dir is None:
        raise ValueError("irt_cache_dir must be provided")

    split_irt_dir = get_or_train_split_irt(
        responses_path=responses_path,
        output_base=irt_cache_dir,
        test_fraction=test_fraction,
        split_seed=split_seed,
        model_type="1pl",
        force_retrain=force_retrain,
        is_binomial=is_binomial,
        exclude_unsolved=exclude_unsolved,
    )

    # Load train-only IRT parameters
    train_abilities = _load_abilities(split_irt_dir / "abilities.csv")
    train_items = _load_items(split_irt_dir / "items.csv")

    # Load optional metadata
    metadata = {}
    if metadata_loader is not None:
        metadata = metadata_loader(all_task_ids)

    # Create appropriate dataset type
    if is_binomial:
        return BinomialExperimentData(
            responses=responses,
            train_abilities=train_abilities,
            train_items=train_items,
            full_abilities=full_abilities,
            full_items=full_items,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            metadata=metadata,
        )
    else:
        return BinaryExperimentData(
            responses=responses,
            train_abilities=train_abilities,
            train_items=train_items,
            full_abilities=full_abilities,
            full_items=full_items,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            metadata=metadata,
        )


def load_dataset_for_fold(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    train_tasks: List[str],
    test_tasks: List[str],
    fold_idx: int,
    k_folds: int,
    split_seed: int,
    is_binomial: bool = False,
    irt_cache_dir: Optional[Path] = None,
    force_retrain: bool = False,
    metadata_loader: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    exclude_unsolved: bool = False,
) -> ExperimentData:
    """Load a dataset for a specific cross-validation fold.

    Unlike load_dataset(), this takes explicit train/test task lists
    instead of computing them from test_fraction. Used for k-fold CV.

    Args:
        abilities_path: Path to full IRT abilities.csv (for oracle only)
        items_path: Path to full IRT items.csv (for oracle only)
        responses_path: Path to response matrix JSONL
        train_tasks: List of task IDs for training this fold
        test_tasks: List of task IDs for testing this fold
        fold_idx: Index of this fold (0 to k_folds-1)
        k_folds: Total number of folds
        split_seed: Random seed (used for cache naming)
        is_binomial: If True, use binomial IRT and BinomialExperimentData
        irt_cache_dir: Directory for cached split IRT models
        force_retrain: If True, retrain IRT even if cached
        metadata_loader: Optional function to load task metadata
        exclude_unsolved: If True, unsolved tasks were filtered (affects cache key)

    Returns:
        BinaryExperimentData or BinomialExperimentData
    """
    from experiment_a.train_irt_split import get_or_train_split_irt

    # Load full IRT parameters (ONLY for oracle baseline)
    full_abilities = _load_abilities(abilities_path)
    full_items = _load_items(items_path)

    # Load responses (binary or binomial)
    if is_binomial:
        responses = _load_binomial_responses(responses_path)
    else:
        responses = _load_binary_responses(responses_path)

    # Get or train fold-specific IRT model
    if irt_cache_dir is None:
        raise ValueError("irt_cache_dir must be provided")

    split_irt_dir = get_or_train_split_irt(
        responses_path=responses_path,
        output_base=irt_cache_dir,
        split_seed=split_seed,
        model_type="1pl",
        force_retrain=force_retrain,
        is_binomial=is_binomial,
        train_tasks=train_tasks,
        fold_idx=fold_idx,
        k_folds=k_folds,
        exclude_unsolved=exclude_unsolved,
    )

    # Load train-only IRT parameters
    train_abilities = _load_abilities(split_irt_dir / "abilities.csv")
    train_items = _load_items(split_irt_dir / "items.csv")

    # Load optional metadata
    all_task_ids = list(full_items.index)
    metadata = {}
    if metadata_loader is not None:
        metadata = metadata_loader(all_task_ids)

    # Create appropriate dataset type
    if is_binomial:
        return BinomialExperimentData(
            responses=responses,
            train_abilities=train_abilities,
            train_items=train_items,
            full_abilities=full_abilities,
            full_items=full_items,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            metadata=metadata,
        )
    else:
        return BinaryExperimentData(
            responses=responses,
            train_abilities=train_abilities,
            train_items=train_items,
            full_abilities=full_abilities,
            full_items=full_items,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            metadata=metadata,
        )
