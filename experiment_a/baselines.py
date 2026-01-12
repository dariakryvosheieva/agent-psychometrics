"""Baseline methods for Experiment A evaluation."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def agent_only_baseline(
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
) -> Dict[str, Any]:
    """Baseline: P(success) = agent's overall success rate.

    This baseline ignores task difficulty entirely and just uses each agent's
    average performance across ALL tasks as the prediction.

    Args:
        abilities: DataFrame with index=agent_id (used for agent list)
        responses: Dict mapping agent_id -> {task_id -> 0|1}
        task_ids: List of task identifiers to evaluate

    Returns:
        Dict with 'auc', 'n_pairs', 'method'
    """
    y_true: List[int] = []
    y_scores: List[float] = []

    # Pre-compute agent success rates across ALL tasks
    agent_success_rates: Dict[str, float] = {}
    for agent_id in abilities.index:
        if agent_id not in responses:
            continue
        outcomes = list(responses[agent_id].values())
        if outcomes:
            agent_success_rates[agent_id] = float(np.mean(outcomes))
        else:
            agent_success_rates[agent_id] = 0.5  # Default

    # Compute baseline predictions for test tasks
    for task_id in task_ids:
        for agent_id in abilities.index:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue

            actual = responses[agent_id][task_id]
            pred_prob = agent_success_rates.get(agent_id, 0.5)

            y_true.append(int(actual))
            y_scores.append(pred_prob)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"error": "Insufficient data", "n_pairs": len(y_true), "method": "agent_only"}

    auc = roc_auc_score(y_true, y_scores)
    return {"auc": float(auc), "n_pairs": len(y_true), "method": "agent_only"}


def task_only_baseline(
    responses: Dict[str, Dict[str, int]],
    train_tasks: List[str],
    test_tasks: List[str],
) -> Dict[str, Any]:
    """Baseline: P(success) = task's observed success rate from training.

    This baseline ignores agent ability entirely and uses the mean pass rate
    from training tasks as the prediction for test tasks.

    Args:
        responses: Dict mapping agent_id -> {task_id -> 0|1}
        train_tasks: List of training task identifiers
        test_tasks: List of test task identifiers

    Returns:
        Dict with 'auc', 'n_pairs', 'method', 'mean_train_rate'
    """
    # Compute mean pass rate from training tasks
    all_train_outcomes: List[int] = []
    for task_id in train_tasks:
        for agent_id in responses:
            if task_id in responses[agent_id]:
                all_train_outcomes.append(responses[agent_id][task_id])

    if all_train_outcomes:
        mean_train_rate = float(np.mean(all_train_outcomes))
    else:
        mean_train_rate = 0.5  # Default

    # Compute baseline predictions for test tasks
    y_true: List[int] = []
    y_scores: List[float] = []

    for task_id in test_tasks:
        for agent_id in responses:
            if task_id not in responses[agent_id]:
                continue

            actual = responses[agent_id][task_id]
            y_true.append(int(actual))
            y_scores.append(mean_train_rate)  # Same prediction for all

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {
            "error": "Insufficient data",
            "n_pairs": len(y_true),
            "method": "task_only",
            "mean_train_rate": mean_train_rate,
        }

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": len(y_true),
        "method": "task_only",
        "mean_train_rate": mean_train_rate,
    }


def random_baseline(n_pairs: int, seed: int = 42) -> Dict[str, Any]:
    """Baseline: Random predictions (expected AUC ~ 0.5).

    Args:
        n_pairs: Number of (agent, task) pairs
        seed: Random seed for reproducibility

    Returns:
        Dict with 'auc', 'n_pairs', 'method'
    """
    rng = np.random.RandomState(seed)

    # Generate random binary outcomes and predictions
    y_true = rng.randint(0, 2, n_pairs)
    y_scores = rng.random(n_pairs)

    if len(set(y_true)) < 2:
        return {"error": "Only one class in random data", "n_pairs": n_pairs, "method": "random"}

    auc = roc_auc_score(y_true, y_scores)
    return {"auc": float(auc), "n_pairs": n_pairs, "method": "random"}
