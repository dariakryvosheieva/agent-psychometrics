"""Baseline predictors and adapters for Experiment A evaluation.

All predictors implement the CVPredictor protocol from cross_validation.py,
allowing them to be used with the unified cross-validation framework.

This module provides:
1. AgentOnlyPredictor - Predicts based on agent's empirical success rate
2. ConstantPredictor - Predicts using mean training difficulty
3. OraclePredictor - Uses true IRT difficulties (upper bound)
4. DifficultyPredictorAdapter - Wraps any DifficultyPredictorBase to implement CVPredictor
"""

from typing import Dict, List

import numpy as np

from experiment_ab_shared.dataset import ExperimentData
from experiment_ab_shared.evaluator import compute_irt_probability
from experiment_ab_shared.predictor_base import DifficultyPredictorBase


class DifficultyPredictorAdapter:
    """Adapts any DifficultyPredictorBase to the CVPredictor protocol.

    Takes a difficulty-based predictor and wraps it to provide predicted
    probabilities using the IRT formula: P(success) = sigmoid(θ - β).
    """

    def __init__(
        self,
        predictor: DifficultyPredictorBase,
        use_full_abilities: bool = False,
    ):
        """Initialize the adapter.

        Args:
            predictor: Any predictor implementing DifficultyPredictorBase
            use_full_abilities: If True, use full IRT abilities (oracle only)
        """
        self._predictor = predictor
        self._use_full_abilities = use_full_abilities
        self._predicted_difficulties: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the underlying difficulty predictor."""
        ground_truth_b = data.train_items.loc[train_task_ids, "b"].values
        self._predictor.fit(train_task_ids, ground_truth_b)
        # Clear cached predictions for new fold
        self._predicted_difficulties = {}

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability using IRT formula."""
        # Lazily predict difficulty for this task if not already done
        if task_id not in self._predicted_difficulties:
            # Predict for all test tasks at once for efficiency
            test_tasks = data.test_tasks
            predictions = self._predictor.predict(test_tasks)
            self._predicted_difficulties.update(predictions)

        if task_id not in self._predicted_difficulties:
            raise ValueError(f"No predicted difficulty for task {task_id}")

        beta = self._predicted_difficulties[task_id]
        abilities = data.full_abilities if self._use_full_abilities else data.train_abilities

        if agent_id not in abilities.index:
            raise ValueError(f"Agent {agent_id} not found in abilities")

        theta = abilities.loc[agent_id, "ability"]
        return compute_irt_probability(theta, beta)


class AgentOnlyPredictor:
    """Baseline that predicts based on agent success rates.

    For each agent, computes their success rate on training tasks.
    At prediction time, ignores task difficulty and returns the
    agent's empirical success rate.
    """

    def __init__(self) -> None:
        self._agent_success_rates: Dict[str, float] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Compute agent success rates on training tasks."""
        self._agent_success_rates = {}

        for agent_id in data.train_abilities.index:
            if agent_id not in data.responses:
                continue

            # Compute success rate using expand_for_auc to handle binary/binomial
            all_outcomes: List[int] = []
            for task_id in train_task_ids:
                if task_id in data.responses[agent_id]:
                    outcomes, _ = data.expand_for_auc(agent_id, task_id, 0.0)
                    all_outcomes.extend(outcomes)

            if all_outcomes:
                self._agent_success_rates[agent_id] = float(np.mean(all_outcomes))
            else:
                self._agent_success_rates[agent_id] = 0.5

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return agent's empirical success rate (ignores task)."""
        if agent_id not in self._agent_success_rates:
            raise ValueError(f"No success rate computed for agent {agent_id}")
        return self._agent_success_rates[agent_id]


class ConstantPredictor:
    """Baseline that predicts the mean training difficulty for all tasks.

    Uses IRT formula: P(success) = sigmoid(θ - β_mean)
    """

    def __init__(self) -> None:
        self._mean_difficulty: float = 0.0

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Compute mean difficulty from training tasks."""
        difficulties = data.get_train_difficulties()
        self._mean_difficulty = float(np.mean(difficulties))

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return P(success) using mean difficulty and agent's ability."""
        from experiment_ab_shared.evaluator import compute_irt_probability

        theta = data.train_abilities.loc[agent_id, "ability"]
        return compute_irt_probability(theta, self._mean_difficulty)


class OraclePredictor:
    """Oracle baseline that uses true IRT difficulties from full model.

    This represents the best possible performance given the IRT framework.
    Uses the full IRT model (trained on all data) for evaluation.
    """

    def __init__(self) -> None:
        pass

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """No training needed - uses oracle difficulties."""
        pass

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Return P(success) using true difficulty and oracle ability."""
        from experiment_ab_shared.evaluator import compute_irt_probability

        # Use full (oracle) abilities and difficulties
        theta = data.full_abilities.loc[agent_id, "ability"]
        beta = data.full_items.loc[task_id, "b"]
        return compute_irt_probability(theta, beta)
