"""Binomial metrics for TerminalBench evaluation.

Computes pass rate MSE between predicted and empirical pass rates
for 5-trial binomial responses.
"""

from typing import Dict, List, Optional

import numpy as np

from experiment_ab_shared.dataset import BinomialExperimentData
from experiment_ab_shared.evaluator import compute_irt_probability


def compute_pass_rate_mse(
    data: BinomialExperimentData,
    predicted_difficulties: Dict[str, float],
    use_full_abilities: bool = False,
) -> Optional[float]:
    """Compute MSE between predicted and empirical pass rates for 5-trial responses.

    For each (agent, task) pair with 5 trials:
        - predicted_prob = sigmoid(theta - beta_predicted)
        - empirical_rate = successes / 5
        - MSE = mean((predicted_prob - empirical_rate)^2)

    Args:
        data: BinomialExperimentData with responses and abilities
        predicted_difficulties: Mapping of task_id -> predicted difficulty
        use_full_abilities: If True, use full IRT abilities (oracle only)

    Returns:
        MSE if there are 5-trial responses, None otherwise
    """
    abilities = data.full_abilities if use_full_abilities else data.train_abilities

    pred_probs: List[float] = []
    empirical_rates: List[float] = []

    for task_id in data.test_tasks:
        beta_pred = predicted_difficulties.get(task_id)
        if beta_pred is None:
            continue

        for agent_id in abilities.index:
            if agent_id not in data.responses:
                continue
            if task_id not in data.responses[agent_id]:
                continue

            resp = data.responses[agent_id][task_id]
            k = resp["successes"]
            n = resp["trials"]

            # Only include 5-trial responses
            if n == 5:
                theta = abilities.loc[agent_id, "ability"]
                prob = compute_irt_probability(theta, beta_pred)
                pred_probs.append(prob)
                empirical_rates.append(k / 5.0)

    if not pred_probs:
        return None

    return float(np.mean((np.array(pred_probs) - np.array(empirical_rates)) ** 2))
