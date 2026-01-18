"""IRT-based evaluation metrics for Experiment A on TerminalBench with binomial data."""

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


def compute_binomial_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, Dict[str, int]]],
    task_ids: List[str],
) -> Dict[str, Any]:
    """Compute AUC for predicted difficulties on test tasks with binomial data.

    For binomial data (k successes, n trials), we expand to binary observations:
    - k observations of (y=1, score=P)
    - (n-k) observations of (y=0, score=P)

    where P = sigmoid(theta - predicted_beta)

    This properly weights agent-task pairs by their trial count.

    Args:
        predicted_difficulties: Dict mapping task_id to predicted difficulty
        abilities: DataFrame with index=agent_id, column 'theta'
        responses: Dict mapping agent_id -> {task_id -> {successes, trials}}
        task_ids: List of task identifiers to evaluate

    Returns:
        Dict with 'auc', 'n_pairs', 'n_observations', 'n_tasks', 'n_agents' and optionally 'error'
    """
    y_true: List[int] = []
    y_scores: List[float] = []

    n_tasks_used = 0
    n_agents_used = set()
    n_pairs = 0

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
            resp = responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]

            prob = compute_irt_probability(theta, beta_pred)

            # Expand binomial to binary observations
            # k successes (y=1) and (n-k) failures (y=0)
            y_true.extend([1] * k + [0] * (n - k))
            y_scores.extend([prob] * n)

            n_agents_used.add(agent_id)
            n_pairs += 1
            task_used = True

        if task_used:
            n_tasks_used += 1

    # Check we have enough data for AUC
    if len(y_true) < 2:
        return {"error": "Insufficient data for AUC", "n_observations": len(y_true)}

    if len(set(y_true)) < 2:
        return {"error": "Only one class present in y_true", "n_observations": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)

    return {
        "auc": float(auc),
        "n_pairs": n_pairs,  # Number of (agent, task) pairs
        "n_observations": len(y_true),  # Total binary observations (expanded from binomial)
        "n_tasks": n_tasks_used,
        "n_agents": len(n_agents_used),
    }
