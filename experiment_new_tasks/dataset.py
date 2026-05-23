"""Abstract dataset interface for Experiment A across different benchmarks.

This module provides:
- ExperimentData: Dataset with per-cell outcomes. Two response formats are
  supported transparently:
    * binary: response is int 0 or 1 (one observation per cell)
    * binomial: response is {"successes": int, "trials": int} (n attempts per cell)
- load_dataset_for_fold: Load dataset for a specific k-fold CV split
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


ResponseValue = Union[int, Dict[str, int]]


def _cell_has_success(response: ResponseValue) -> bool:
    """True if this cell records at least one successful attempt.

    Accepts both binary (int 0/1) and binomial ({"successes", "trials"}) cells.
    Fails loudly on malformed values.
    """
    if isinstance(response, dict):
        if "successes" not in response or "trials" not in response:
            raise ValueError(
                f"Malformed binomial response (missing 'successes'/'trials'): {response!r}"
            )
        return int(response["successes"]) > 0
    if isinstance(response, (int, float, bool)):
        return int(response) == 1
    raise ValueError(f"Unsupported response value type: {type(response).__name__} ({response!r})")


def filter_unsolved_tasks(
    task_ids: List[str],
    responses: Dict[str, Dict[str, ResponseValue]],
) -> Tuple[List[str], int]:
    """Filter out tasks where no agent achieved any success.

    Args:
        task_ids: List of all task IDs to filter
        responses: Response matrix {agent_id: {task_id: response}} where response
            is either int 0/1 (binary) or {"successes", "trials"} (binomial).

    Returns:
        Tuple of (filtered_task_ids, n_excluded)
    """
    solved_tasks = []
    for task_id in task_ids:
        task_solved = False
        for agent_responses in responses.values():
            if task_id not in agent_responses:
                continue
            if _cell_has_success(agent_responses[task_id]):
                task_solved = True
                break
        if task_solved:
            solved_tasks.append(task_id)

    n_excluded = len(task_ids) - len(solved_tasks)
    return solved_tasks, n_excluded



@dataclass
class ExperimentData:
    """Dataset for per-cell outcomes.

    Supports two response formats transparently:
    - binary: each cell is int 0 or 1 (one observation per cell)
    - binomial: each cell is {"successes": int, "trials": int} (n attempts per cell,
      expanded for AUC into n labeled observations sharing the same predicted score)
    """

    # Response matrix: agent_id -> task_id -> int (binary) or {"successes","trials"} (binomial)
    responses: Dict[str, Dict[str, ResponseValue]]

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

    @property
    def is_binomial(self) -> bool:
        """True iff any response value is in the binomial dict format."""
        for agent_responses in self.responses.values():
            for value in agent_responses.values():
                return isinstance(value, dict)
        return False

    def expand_for_auc(
        self, agent_id: str, task_id: str, prob: float
    ) -> Tuple[List[int], List[float]]:
        """Expand a cell into per-attempt labeled observations for AUC.

        - Binary cell: returns ([0|1], [prob]) — a single observation per cell.
        - Binomial cell {"successes": k, "trials": n}: returns n observations with
          k labels=1 and (n-k) labels=0, all sharing the same predicted probability.
          sklearn's roc_auc_score handles the resulting score ties correctly via
          the Mann-Whitney U formulation.
        """
        actual = self.responses[agent_id][task_id]
        if isinstance(actual, dict):
            successes = int(actual["successes"])
            trials = int(actual["trials"])
            if successes < 0 or trials < 0 or successes > trials:
                raise ValueError(
                    f"Invalid binomial cell for ({agent_id!r}, {task_id!r}): {actual!r}"
                )
            labels = [1] * successes + [0] * (trials - successes)
            scores = [prob] * trials
            return labels, scores
        return [int(actual)], [prob]

    def get_train_difficulties(self) -> np.ndarray:
        """Get ground truth difficulties for training tasks (from train IRT)."""
        return self.train_items.loc[self.train_tasks, "b"].values

    def get_all_agents(self) -> List[str]:
        """Get list of all agent IDs."""
        return list(self.train_abilities.index)


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


def _load_responses(responses_path: Path) -> Dict[str, Dict[str, ResponseValue]]:
    """Load response matrix from JSONL.

    Values per cell may be int 0/1 (binary) or {"successes": int, "trials": int}
    (binomial). Mixed formats within a single file are not supported (the IRT
    trainer auto-detects format based on presence of dict-valued cells).
    """
    responses: Dict[str, Dict[str, ResponseValue]] = {}
    with open(responses_path, "r") as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses



def load_dataset_for_fold(
    abilities_path: Path,
    items_path: Path,
    responses_path: Path,
    train_tasks: List[str],
    test_tasks: List[str],
    fold_idx: int,
    k_folds: int,
    split_seed: int,
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
        irt_cache_dir: Directory for cached split IRT models
        force_retrain: If True, retrain IRT even if cached
        metadata_loader: Optional function to load task metadata
        exclude_unsolved: If True, unsolved tasks were filtered (affects cache key)

    Returns:
        ExperimentData
    """
    from experiment_new_tasks.train_irt_split import get_or_train_split_irt

    # Load full IRT parameters (ONLY for oracle baseline)
    full_abilities = _load_abilities(abilities_path)
    full_items = _load_items(items_path)

    # Load responses
    responses = _load_responses(responses_path)

    # Get or train fold-specific IRT model
    if irt_cache_dir is None:
        raise ValueError("irt_cache_dir must be provided")

    split_irt_dir = get_or_train_split_irt(
        responses_path=responses_path,
        output_base=irt_cache_dir,
        split_seed=split_seed,
        model_type="1pl",
        force_retrain=force_retrain,
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

    return ExperimentData(
        responses=responses,
        train_abilities=train_abilities,
        train_items=train_items,
        full_abilities=full_abilities,
        full_items=full_items,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        metadata=metadata,
    )
