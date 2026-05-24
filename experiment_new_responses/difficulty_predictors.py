"""CV predictors for the new-responses experiment."""

from typing import List

from experiment_new_responses.cross_validation import probability_from_theta
from experiment_new_responses.dataset import (
    ResponseExperimentData,
)


def _subject_id(agent_key: str) -> str:
    return str(agent_key).split("::", 1)[1] if "::" in str(agent_key) else str(agent_key)


def combine_theta(theta_model: float, theta_scaffold: float, *, combine: str) -> float:
    combine_norm = str(combine or "sum").strip().lower()
    if combine_norm == "sum":
        return float(theta_model) + float(theta_scaffold)
    if combine_norm == "mean":
        return (float(theta_model) + float(theta_scaffold)) / 2.0
    if combine_norm == "l2":
        return (float(theta_model) ** 2 + float(theta_scaffold) ** 2) ** 0.5
    raise ValueError(f"Unknown theta_combine={combine!r}")


class ModelScaffoldPredictor:
    """Use model+scaffold IRT parameters trained without held-out cells."""

    requires_train_irt = True

    def fit(self, data: ResponseExperimentData, train_observation_ids: List[str]) -> None:
        pass

    def predict_probability(
        self, data: ResponseExperimentData, agent_id: str, task_id: str
    ) -> float:
        if agent_id not in data.agent_to_ms_pair:
            raise ValueError(f"Agent {agent_id!r} has no model/scaffold mapping")
        model, scaffold = data.agent_to_ms_pair[agent_id]
        if model not in data.train_model_abilities.index:
            raise ValueError(f"Model {model!r} was not observed in training")
        if scaffold not in data.train_scaffold_abilities.index:
            raise ValueError(f"Scaffold {scaffold!r} was not observed in training")
        if task_id not in data.train_items.index:
            raise ValueError(f"Task {task_id!r} has no train-fold IRT difficulty")
        theta = combine_theta(
            data.train_model_abilities.loc[model, "theta"],
            data.train_scaffold_abilities.loc[scaffold, "theta"],
            combine=data.theta_combine,
        )
        return probability_from_theta(theta, data.train_items.loc[task_id, "b"])


class StandardIrtPredictor:
    """Use standard agent/item IRT parameters trained without held-out cells."""

    requires_train_irt = True

    def fit(self, data: ResponseExperimentData, train_observation_ids: List[str]) -> None:
        pass

    def predict_probability(
        self, data: ResponseExperimentData, agent_id: str, task_id: str
    ) -> float:
        subject_id = _subject_id(agent_id)
        if subject_id not in data.train_abilities.index:
            raise ValueError(f"Agent {subject_id!r} has no train-fold IRT ability")
        if task_id not in data.train_items.index:
            raise ValueError(f"Task {task_id!r} has no train-fold IRT difficulty")
        return probability_from_theta(
            data.train_abilities.loc[subject_id, "ability"],
            data.train_items.loc[task_id, "b"],
        )


class OraclePredictor:
    """Oracle using full standard IRT abilities and item difficulties."""

    requires_train_irt = False

    def fit(self, data: ResponseExperimentData, train_observation_ids: List[str]) -> None:
        pass

    def predict_probability(
        self, data: ResponseExperimentData, agent_id: str, task_id: str
    ) -> float:
        subject_id = _subject_id(agent_id)
        if subject_id not in data.full_abilities.index:
            raise ValueError(f"Agent {subject_id!r} has no full-IRT ability")
        if task_id not in data.full_items.index:
            raise ValueError(f"Task {task_id!r} has no full-IRT difficulty")
        return probability_from_theta(
            data.full_abilities.loc[subject_id, "ability"],
            data.full_items.loc[task_id, "b"],
        )
