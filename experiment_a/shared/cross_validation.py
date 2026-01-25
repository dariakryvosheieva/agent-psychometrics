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


def expand_with_mode(
    data: ExperimentData,
    agent_id: str,
    task_id: str,
    prob: float,
    expansion_mode: Optional[str],
    binomial_responses: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
) -> Tuple[List[int], List[float]]:
    """Expand response for AUC with explicit mode control.

    This function allows decoupling the evaluation method from the data type,
    enabling fair comparisons between training methods.

    Args:
        data: The experiment data
        agent_id: Agent identifier
        task_id: Task identifier
        prob: Predicted probability
        expansion_mode: "binary", "expand", or None (use data's default)
            - "binary": Collapse to any_success = (k > 0) for binomial, raw value for binary
            - "expand": Expand to n observations from binomial ground truth
            - None: Use data's natural expand_for_auc method
        binomial_responses: Original binomial responses, required when expansion_mode="expand"
            and data is binary (trained on sampled data)

    Returns:
        (y_true, y_scores) tuple for AUC computation

    Raises:
        ValueError: If expansion_mode is unknown or required data is missing
    """
    if expansion_mode is None:
        return data.expand_for_auc(agent_id, task_id, prob)

    response = data.responses[agent_id][task_id]

    if expansion_mode == "binary":
        # Collapse to any_success = (k > 0) for binomial, or use raw value for binary
        if isinstance(response, dict):
            y = 1 if response["successes"] > 0 else 0
        else:
            y = int(response)
        return [y], [prob]

    elif expansion_mode == "expand":
        # Expand to n observations from binomial ground truth
        if binomial_responses is None:
            raise ValueError(
                "expansion_mode='expand' requires binomial_responses, but None provided"
            )
        if agent_id not in binomial_responses:
            raise ValueError(
                f"Agent {agent_id!r} not found in binomial_responses"
            )
        if task_id not in binomial_responses[agent_id]:
            raise ValueError(
                f"Task {task_id!r} not found for agent {agent_id!r} in binomial_responses"
            )
        resp = binomial_responses[agent_id][task_id]
        k, n = resp["successes"], resp["trials"]
        return [1] * k + [0] * (n - k), [prob] * n

    raise ValueError(
        f"Unknown expansion_mode: {expansion_mode!r}. Must be 'binary', 'expand', or None"
    )


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
    This is more standard and well-tested than custom hash-based splitting.

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


def run_cv(
    predictor: CVPredictor,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    verbose: bool = True,
    compute_pass_rate_mse: bool = False,
    expansion_mode: Optional[str] = None,
    binomial_responses: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
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
        compute_pass_rate_mse: If True and data is binomial, compute pass rate MSE
        expansion_mode: Override AUC expansion method ("binary", "expand", or None for default)
        binomial_responses: Original binomial responses, required for expansion_mode="expand"
            when data is binary (trained on sampled data)
        diagnostics_extractor: Optional callback to extract diagnostics from predictor after
            each fold. Called as diagnostics_extractor(predictor, fold_idx) after fitting.
            Results are collected in CrossValidationResult.fold_diagnostics.

    Returns:
        CrossValidationResult with mean/std AUC across folds
    """
    fold_aucs: List[Optional[float]] = []
    fold_mses: List[Optional[float]] = []
    fold_diagnostics: List[Any] = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        # Load fold-specific data
        data = load_fold_data(train_tasks, test_tasks, fold_idx)

        # Train predictor
        predictor.fit(data, train_tasks)

        # Extract diagnostics if callback provided
        if diagnostics_extractor is not None:
            fold_diagnostics.append(diagnostics_extractor(predictor, fold_idx))

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

                # Expand outcomes (with optional mode override)
                outcomes, _ = expand_with_mode(
                    data, agent_id, task_id, prob, expansion_mode, binomial_responses
                )
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
        fold_diagnostics=fold_diagnostics if diagnostics_extractor is not None else None,
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
