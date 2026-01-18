"""IRT-based evaluation metrics for Experiment A."""

from typing import Any, Dict, List

import pandas as pd
from scipy.special import expit  # sigmoid
from sklearn.metrics import roc_auc_score


def compute_irt_probability(theta: float, beta: float) -> float:
    """Compute 1PL IRT success probability.

    P(success) = sigmoid(theta - beta)

    Args:
        theta: Agent ability parameter
        beta: Task difficulty parameter

    Returns:
        Probability of success (0 to 1)
    """
    return float(expit(theta - beta))


def compute_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
) -> Dict[str, Any]:
    """Compute AUC for predicted difficulties on test tasks.

    For each (agent, task) pair:
    - y_true = actual response (0 or 1)
    - y_score = sigmoid(theta_j - predicted_beta_i)

    Args:
        predicted_difficulties: Dict mapping task_id to predicted difficulty
        abilities: DataFrame with index=agent_id, column 'theta'
        responses: Dict mapping agent_id -> {task_id -> 0|1}
        task_ids: List of task identifiers to evaluate

    Returns:
        Dict with 'auc', 'n_pairs', 'n_tasks', 'n_agents' and optionally 'error'
    """
    y_true: List[int] = []
    y_scores: List[float] = []

    n_tasks_used = 0
    n_agents_used = set()

    for task_id in task_ids:
        if task_id not in predicted_difficulties:
            continue

        beta_pred = predicted_difficulties[task_id]
        task_used = False

        for agent_id in abilities.index:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue

            theta = float(abilities.loc[agent_id, "theta"])
            actual = responses[agent_id][task_id]
            prob = compute_irt_probability(theta, beta_pred)

            y_true.append(int(actual))
            y_scores.append(prob)
            n_agents_used.add(agent_id)
            task_used = True

        if task_used:
            n_tasks_used += 1

    # Check we have enough data for AUC
    if len(y_true) < 2:
        return {"error": "Insufficient data for AUC", "n_pairs": len(y_true)}

    if len(set(y_true)) < 2:
        return {"error": "Only one class present in y_true", "n_pairs": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)

    return {
        "auc": float(auc),
        "n_pairs": len(y_true),
        "n_tasks": n_tasks_used,
        "n_agents": len(n_agents_used),
    }
