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

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_ab_shared.dataset import ExperimentData


@dataclass
class CrossValidationResult:
    """Results from k-fold cross-validation."""

    mean_auc: Optional[float]
    std_auc: Optional[float]
    fold_aucs: List[Optional[float]]
    k: int

    # Optional pass rate MSE (only for binomial data with 5-trial responses)
    mean_pass_rate_mse: Optional[float] = None
    std_pass_rate_mse: Optional[float] = None
    fold_pass_rate_mses: Optional[List[Optional[float]]] = None


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
    """Deterministic k-fold split using hash-based assignment.

    Each task is assigned to exactly one fold based on its hash.
    Returns k (train_tasks, test_tasks) tuples where each test set
    is one fold and train set is the remaining k-1 folds.

    Args:
        task_ids: List of task identifiers
        k: Number of folds (e.g., 5)
        seed: Random seed for reproducibility

    Returns:
        List of k tuples: [(train_tasks_0, test_tasks_0), ...]
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    # Score each task using MD5 hash
    scored: List[Tuple[float, str]] = []
    for task_id in task_ids:
        h = hashlib.md5((str(task_id) + f"::{seed}").encode("utf-8")).hexdigest()
        score = int(h[:8], 16) / float(16**8)
        scored.append((score, task_id))
    scored.sort()

    # Extract sorted task IDs
    all_tasks_sorted = [task_id for _, task_id in scored]

    # Divide into k folds (last fold may be slightly larger)
    fold_size = len(all_tasks_sorted) // k
    folds: List[List[str]] = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(all_tasks_sorted)
        folds.append(all_tasks_sorted[start:end])

    # Generate (train, test) tuples for each fold
    result: List[Tuple[List[str], List[str]]] = []
    for i in range(k):
        test_tasks = folds[i]
        train_tasks = [t for j, fold in enumerate(folds) if j != i for t in fold]
        result.append((train_tasks, test_tasks))

    return result


def run_cv(
    predictor: CVPredictor,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    verbose: bool = True,
    compute_pass_rate_mse: bool = False,
) -> CrossValidationResult:
    """Run cross-validation for any predictor.

    This is a unified CV function that works with any predictor implementing
    the CVPredictor protocol.

    Args:
        predictor: Any predictor implementing CVPredictor protocol
        folds: List of (train_tasks, test_tasks) tuples from k_fold_split_tasks
        load_fold_data: Function that loads ExperimentData for a specific fold
        verbose: Print per-fold AUC results
        compute_pass_rate_mse: If True and data is binomial, compute pass rate MSE

    Returns:
        CrossValidationResult with mean/std AUC across folds
    """
    fold_aucs: List[Optional[float]] = []
    fold_mses: List[Optional[float]] = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        # Load fold-specific data
        data = load_fold_data(train_tasks, test_tasks, fold_idx)

        # Train predictor
        predictor.fit(data, train_tasks)

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

                # Expand outcomes (handles binary vs binomial)
                outcomes, _ = data.expand_for_auc(agent_id, task_id, prob)
                y_true.extend(outcomes)
                y_scores.extend([prob] * len(outcomes))

        # Compute AUC
        if len(y_true) >= 2 and len(set(y_true)) >= 2:
            auc = float(roc_auc_score(y_true, y_scores))
        else:
            auc = None
        fold_aucs.append(auc)

        # Optionally compute pass rate MSE
        if compute_pass_rate_mse:
            mse = _compute_pass_rate_mse_for_fold(predictor, data, test_tasks)
            fold_mses.append(mse)

        if verbose:
            if auc is not None:
                print(f"      Fold {fold_idx + 1}: AUC = {auc:.4f}")
            else:
                print(f"      Fold {fold_idx + 1}: AUC = N/A")

    # Aggregate results
    valid_aucs = [a for a in fold_aucs if a is not None]
    valid_mses = [m for m in fold_mses if m is not None]

    return CrossValidationResult(
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else None,
        std_auc=float(np.std(valid_aucs)) if valid_aucs else None,
        fold_aucs=fold_aucs,
        k=len(folds),
        mean_pass_rate_mse=float(np.mean(valid_mses)) if valid_mses else None,
        std_pass_rate_mse=float(np.std(valid_mses)) if valid_mses else None,
        fold_pass_rate_mses=fold_mses if compute_pass_rate_mse else None,
    )


def _compute_pass_rate_mse_for_fold(
    predictor: CVPredictor,
    data: ExperimentData,
    test_tasks: List[str],
) -> Optional[float]:
    """Compute MSE between predicted and empirical pass rates for 5-trial responses."""
    from experiment_ab_shared.dataset import BinomialExperimentData

    if not isinstance(data, BinomialExperimentData):
        return None

    pred_probs: List[float] = []
    empirical_rates: List[float] = []

    for task_id in test_tasks:
        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            resp = data.responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]

            if n == 5:
                prob = predictor.predict_probability(data, agent_id, task_id)
                pred_probs.append(prob)
                empirical_rates.append(k / 5.0)

    if not pred_probs:
        return None

    return float(np.mean((np.array(pred_probs) - np.array(empirical_rates)) ** 2))
