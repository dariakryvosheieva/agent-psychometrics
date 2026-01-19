"""Cross-validation utilities for Experiment A.

Provides k-fold cross-validation support for difficulty prediction experiments.

Key concepts:
- Each fold has 20% held-out test tasks and 80% train tasks
- A fold-specific IRT model is trained on the train tasks to get:
  - Agent abilities (θ) - used to compute predicted probabilities
  - Task difficulties (β) - ground truth for training the predictor
- The predictor learns to predict β from task features (embeddings, LLM features)
- AUC is computed on test tasks by comparing:
  - Predicted probability: P(success) = sigmoid(θ - β̂)
  - Actual outcome from the response matrix

Efficiency feature: predictors are instantiated once (loading embeddings/features),
then fit() and predict() are called k times with different train/test splits.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from experiment_a_common.dataset import ExperimentData
from experiment_a_common.evaluator import compute_auc, PredictorConfig


@dataclass
class CrossValidationResult:
    """Results from k-fold cross-validation."""

    mean_auc: Optional[float]
    std_auc: Optional[float]
    fold_aucs: List[Optional[float]]
    k: int

    # Optional binomial metrics (only for BinomialExperimentData)
    mean_mae: Optional[float] = None
    std_mae: Optional[float] = None
    fold_maes: Optional[List[Optional[float]]] = None

    mean_pass5_accuracy: Optional[float] = None
    std_pass5_accuracy: Optional[float] = None
    fold_pass5_accuracies: Optional[List[Optional[float]]] = None

    # Aggregated confusion matrix across all folds (5-trial responses only)
    pass5_confusion_matrix: Optional[List[List[int]]] = None


def k_fold_split_tasks(
    task_ids: List[str],
    k: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """Deterministic k-fold split using hash-based assignment.

    Each task is assigned to exactly one fold based on its hash.
    Returns k (train_tasks, test_tasks) tuples where each test set
    is one fold and train set is the remaining k-1 folds.

    Uses the same hash-based approach as stable_split_tasks() for consistency.

    Args:
        task_ids: List of task identifiers
        k: Number of folds (e.g., 5)
        seed: Random seed for reproducibility

    Returns:
        List of k tuples: [(train_tasks_0, test_tasks_0), ...]
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    # Score each task using MD5 hash (same approach as stable_split_tasks)
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


def run_cv_for_predictor(
    predictor_config: PredictorConfig,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    verbose: bool = True,
    compute_binomial: bool = False,
) -> CrossValidationResult:
    """Run cross-validation for a single predictor.

    The predictor is instantiated ONCE (loading embeddings/features in __init__),
    then fit() and predict() are called k times with different train/test splits.
    This ensures expensive data loading (embeddings, LLM features) happens only once.

    Args:
        predictor_config: The predictor configuration (class + kwargs)
        folds: List of (train_tasks, test_tasks) tuples from k_fold_split_tasks
        load_fold_data: Function that loads ExperimentData for a specific fold.
                        Signature: (train_tasks, test_tasks, fold_idx) -> ExperimentData
                        The returned ExperimentData contains:
                        - train_abilities: Agent abilities (θ) from fold's IRT model
                        - train_items: Task difficulties (β) from fold's IRT model
                        - test_tasks: The held-out test task IDs for this fold
                        - responses: Full response matrix (agent -> task -> outcome)
        verbose: Print per-fold AUC results
        compute_binomial: If True and data is binomial, compute MAE/accuracy metrics

    Returns:
        CrossValidationResult with mean/std AUC across folds

    How AUC is computed:
        For each test task with predicted difficulty β̂:
            For each agent with ability θ:
                predicted_prob = sigmoid(θ - β̂)
                actual = responses[agent][task]  # 0 or 1
        AUC = ROC-AUC(actual_outcomes, predicted_probs)
    """
    if not predictor_config.enabled:
        return CrossValidationResult(
            mean_auc=None,
            std_auc=None,
            fold_aucs=[None] * len(folds),
            k=len(folds),
        )

    # Instantiate predictor once - this loads embeddings/features
    predictor = predictor_config.predictor_class(**predictor_config.kwargs)

    fold_aucs: List[Optional[float]] = []
    fold_maes: List[Optional[float]] = []
    fold_pass5_accuracies: List[Optional[float]] = []
    # Aggregate confusion matrix across folds
    total_confusion: Optional[List[List[int]]] = None

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        # Load fold-specific data:
        # - IRT model trained on this fold's 80% train tasks
        # - Provides abilities (θ) and ground truth difficulties (β)
        data = load_fold_data(train_tasks, test_tasks, fold_idx)

        # Get ground truth difficulties for training tasks
        # These come from the IRT model trained on this fold's train tasks
        train_b = data.get_train_difficulties()

        # Train predictor to predict β from task features
        predictor.fit(train_tasks, train_b)

        # Predict difficulties for held-out test tasks
        predictions = predictor.predict(test_tasks)

        # Compute AUC on test tasks:
        # For each (agent, test_task) pair:
        #   predicted_prob = sigmoid(θ_agent - β̂_task)
        #   actual = responses[agent][task]
        # AUC = ROC-AUC(actuals, predicted_probs)
        auc_result = compute_auc(
            data, predictions, use_full_abilities=predictor_config.use_full_abilities
        )
        auc = auc_result.get("auc")
        fold_aucs.append(auc)

        # Optionally compute binomial metrics
        if compute_binomial:
            from experiment_a_common.binomial_metrics import compute_binomial_metrics
            from experiment_a_common.dataset import BinomialExperimentData

            if isinstance(data, BinomialExperimentData):
                binom_result = compute_binomial_metrics(
                    data, predictions, use_full_abilities=predictor_config.use_full_abilities
                )
                fold_maes.append(binom_result.mae if not np.isnan(binom_result.mae) else None)
                fold_pass5_accuracies.append(
                    binom_result.pass5_accuracy
                    if not np.isnan(binom_result.pass5_accuracy)
                    else None
                )

                # Aggregate confusion matrix
                if total_confusion is None:
                    total_confusion = [[0] * 6 for _ in range(6)]
                for i in range(6):
                    for j in range(6):
                        total_confusion[i][j] += binom_result.pass5_confusion_matrix[i][j]

        if verbose:
            if auc is not None:
                print(f"      Fold {fold_idx + 1}: AUC = {auc:.4f}")
            else:
                print(f"      Fold {fold_idx + 1}: AUC = N/A")

    # Aggregate results across folds
    valid_aucs = [a for a in fold_aucs if a is not None]
    valid_maes = [m for m in fold_maes if m is not None]
    valid_accs = [a for a in fold_pass5_accuracies if a is not None]

    return CrossValidationResult(
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else None,
        std_auc=float(np.std(valid_aucs)) if valid_aucs else None,
        fold_aucs=fold_aucs,
        k=len(folds),
        mean_mae=float(np.mean(valid_maes)) if valid_maes else None,
        std_mae=float(np.std(valid_maes)) if valid_maes else None,
        fold_maes=fold_maes if compute_binomial else None,
        mean_pass5_accuracy=float(np.mean(valid_accs)) if valid_accs else None,
        std_pass5_accuracy=float(np.std(valid_accs)) if valid_accs else None,
        fold_pass5_accuracies=fold_pass5_accuracies if compute_binomial else None,
        pass5_confusion_matrix=total_confusion,
    )


def run_cv_for_baseline(
    baseline_fn: Callable[[ExperimentData], Dict[str, Any]],
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int], ExperimentData],
    verbose: bool = True,
    compute_binomial: bool = False,
) -> CrossValidationResult:
    """Run cross-validation for a baseline method (like agent-only).

    Args:
        baseline_fn: Function that takes ExperimentData and returns result dict with 'auc'
        folds: List of (train_tasks, test_tasks) tuples
        load_fold_data: Function that loads ExperimentData for a specific fold
        verbose: Print per-fold AUC results
        compute_binomial: If True and data is binomial, compute MAE/accuracy metrics

    Returns:
        CrossValidationResult with mean/std AUC across folds
    """
    fold_aucs: List[Optional[float]] = []
    fold_maes: List[Optional[float]] = []
    fold_pass5_accuracies: List[Optional[float]] = []
    total_confusion: Optional[List[List[int]]] = None

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        data = load_fold_data(train_tasks, test_tasks, fold_idx)
        result = baseline_fn(data)
        auc = result.get("auc")
        fold_aucs.append(auc)

        # Optionally compute binomial metrics for baseline
        if compute_binomial:
            binom_result = result.get("binomial_metrics")
            if binom_result is not None:
                fold_maes.append(
                    binom_result["mae"] if not np.isnan(binom_result["mae"]) else None
                )
                fold_pass5_accuracies.append(
                    binom_result["pass5_accuracy"]
                    if not np.isnan(binom_result["pass5_accuracy"])
                    else None
                )

                # Aggregate confusion matrix
                if total_confusion is None:
                    total_confusion = [[0] * 6 for _ in range(6)]
                for i in range(6):
                    for j in range(6):
                        total_confusion[i][j] += binom_result["pass5_confusion_matrix"][i][j]

        if verbose:
            if auc is not None:
                print(f"      Fold {fold_idx + 1}: AUC = {auc:.4f}")
            else:
                print(f"      Fold {fold_idx + 1}: AUC = N/A")

    valid_aucs = [a for a in fold_aucs if a is not None]
    valid_maes = [m for m in fold_maes if m is not None]
    valid_accs = [a for a in fold_pass5_accuracies if a is not None]

    return CrossValidationResult(
        mean_auc=float(np.mean(valid_aucs)) if valid_aucs else None,
        std_auc=float(np.std(valid_aucs)) if valid_aucs else None,
        fold_aucs=fold_aucs,
        k=len(folds),
        mean_mae=float(np.mean(valid_maes)) if valid_maes else None,
        std_mae=float(np.std(valid_maes)) if valid_maes else None,
        fold_maes=fold_maes if compute_binomial else None,
        mean_pass5_accuracy=float(np.mean(valid_accs)) if valid_accs else None,
        std_pass5_accuracy=float(np.std(valid_accs)) if valid_accs else None,
        fold_pass5_accuracies=fold_pass5_accuracies if compute_binomial else None,
        pass5_confusion_matrix=total_confusion,
    )