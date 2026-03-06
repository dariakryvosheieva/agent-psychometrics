"""Cross-validation utilities for Experiment A.

Provides k-fold cross-validation support for difficulty prediction experiments.

Key concepts:
- Each fold has 20% held-out test tasks and 80% train tasks
- A fold-specific IRT model is trained on the train tasks to provide ground truth
- Predictors are evaluated by comparing predicted probabilities to actual outcomes

Design principle: A single unified CV function handles ALL predictor types.
Each predictor implements a protocol that provides predicted probabilities
for (agent, task) pairs, allowing flexibility in how predictions are made.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from experiment_ab_shared.dataset import ExperimentData


@dataclass
class CrossValidationResult:
    """Results from k-fold cross-validation."""

    mean_auc: Optional[float]
    std_auc: Optional[float]
    fold_aucs: List[Optional[float]]
    k: int

    # Optional diagnostics collected via callback
    fold_diagnostics: Optional[List[Any]] = None


class CVPredictor(Protocol):
    """Protocol for predictors that can be used in cross-validation.

    Predictors must implement:
    1. fit(): Learn from training data
    2. predict_probability(): Return predicted success probability for (agent, task) pairs

    The ExperimentData object provides access to everything needed:
    - Ground truth difficulties: data.get_train_difficulties()
    - Responses: data.responses (for training)
    - Agent abilities: data.train_abilities
    """

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the predictor on training data.

        Args:
            data: ExperimentData containing all information needed for training
            train_task_ids: List of task IDs to train on
        """
        ...

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict probability of success for a specific (agent, task) pair.

        Args:
            data: ExperimentData for accessing agent abilities, etc.
            agent_id: The agent
            task_id: The task

        Returns:
            Predicted probability of success (0 to 1)
        """
        ...


def k_fold_split_tasks(
    task_ids: List[str],
    k: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """K-fold split using sklearn KFold.

    Uses sklearn's standard KFold implementation for splitting tasks.

    Args:
        task_ids: List of task identifiers
        k: Number of folds (e.g., 5)
        seed: Random seed for reproducibility (used with shuffle=True)

    Returns:
        List of k tuples: [(train_tasks_0, test_tasks_0), ...]
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    task_ids = list(task_ids)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    result: List[Tuple[List[str], List[str]]] = []
    for train_idx, test_idx in kfold.split(task_ids):
        train_tasks = [task_ids[i] for i in train_idx]
        test_tasks = [task_ids[i] for i in test_idx]
        result.append((train_tasks, test_tasks))

    return result


def _run_single_fold(
    predictor: CVPredictor,
    fold_idx: int,
    train_tasks: List[str],
    test_tasks: List[str],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    diagnostics_extractor: Optional[Callable[[CVPredictor, int], Any]],
) -> Dict[str, Any]:
    """Run a single fold of cross-validation.

    Returns:
        Dict with 'auc' and 'diagnostics' (optional)
    """
    # Load fold-specific data
    data = load_fold_data(train_tasks, test_tasks, fold_idx)

    # Train predictor
    predictor.fit(data, train_tasks)

    # Extract diagnostics if callback provided
    diagnostics = None
    if diagnostics_extractor is not None:
        diagnostics = diagnostics_extractor(predictor, fold_idx)

    # Evaluate on test tasks
    y_true: List[int] = []
    y_scores: List[float] = []

    for task_id in test_tasks:
        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            # Get predicted probability
            prob = predictor.predict_probability(data, agent_id, task_id)

            # Get actual outcome
            actual = data.responses[agent_id][task_id]
            y_true.append(int(actual))
            y_scores.append(prob)

    # Compute AUC
    if len(y_true) >= 2 and len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))
    else:
        auc = None

    return {
        "fold_idx": fold_idx,
        "auc": auc,
        "diagnostics": diagnostics,
    }


def evaluate_predictor_cv(
    predictor: CVPredictor,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    verbose: bool = True,
    diagnostics_extractor: Optional[Callable[[CVPredictor, int], Any]] = None,
) -> CrossValidationResult:
    """Run cross-validation for any predictor.

    This is a unified CV function that works with any predictor implementing
    the CVPredictor protocol.

    Args:
        predictor: Any predictor implementing CVPredictor protocol
        folds: List of (train_tasks, test_tasks) tuples from k_fold_split_tasks
        load_fold_data: Function that loads ExperimentData for a specific fold
        verbose: Print per-fold AUC results
        diagnostics_extractor: Optional callback to extract diagnostics from predictor after
            each fold. Called as diagnostics_extractor(predictor, fold_idx) after fitting.
            Results are collected in CrossValidationResult.fold_diagnostics.

    Returns:
        CrossValidationResult with mean/std AUC across folds
    """
    fold_aucs: List[Optional[float]] = []
    fold_diagnostics: List[Any] = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        result = _run_single_fold(
            predictor, fold_idx, train_tasks, test_tasks,
            load_fold_data, diagnostics_extractor
        )
        fold_aucs.append(result["auc"])
        if diagnostics_extractor is not None:
            fold_diagnostics.append(result["diagnostics"])

        if verbose:
            auc = result["auc"]
            if auc is not None:
                print(f"      Fold {fold_idx + 1}: AUC = {auc:.4f}")
            else:
                print(f"      Fold {fold_idx + 1}: AUC = N/A")

    # Aggregate results
    valid_aucs = [a for a in fold_aucs if a is not None]

    return CrossValidationResult(
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else None,
        std_auc=float(np.std(valid_aucs)) if valid_aucs else None,
        fold_aucs=fold_aucs,
        k=len(folds),
        fold_diagnostics=fold_diagnostics if diagnostics_extractor is not None else None,
    )
