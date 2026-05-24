"""CV predictors for the new-agents experiment."""

from typing import List

from experiment_new_agents.cross_validation import probability_from_theta
from experiment_new_agents.dataset import AgentExperimentData, combine_theta


class ModelScaffoldPredictor:
    """Use train-agent model+scaffold IRT parameters for held-out agent pairs."""

    requires_train_irt = True

    def fit(self, data: AgentExperimentData, train_agent_ids: List[str]) -> None:
        pass

    def _theta(self, data: AgentExperimentData, agent_id: str) -> float:
        if agent_id not in data.agent_to_ms_pair:
            raise ValueError(f"Agent {agent_id!r} has no model/scaffold mapping")
        model, scaffold = data.agent_to_ms_pair[agent_id]
        if model not in data.train_model_abilities.index:
            raise ValueError(f"Model {model!r} was not observed in training")
        if scaffold not in data.train_scaffold_abilities.index:
            raise ValueError(f"Scaffold {scaffold!r} was not observed in training")
        return combine_theta(
            float(data.train_model_abilities.loc[model, "theta"]),
            float(data.train_scaffold_abilities.loc[scaffold, "theta"]),
            combine=data.theta_combine,
        )

    def predict_probability(
        self, data: AgentExperimentData, agent_id: str, task_id: str
    ) -> float:
        if task_id not in data.train_items.index:
            raise ValueError(f"Task {task_id!r} has no train-fold IRT difficulty")
        return probability_from_theta(self._theta(data, agent_id), data.train_items.loc[task_id, "b"])


class ConstantPredictor:
    """Baseline using each task's empirical solve rate among training agents."""

    requires_train_irt = False

    def fit(self, data: AgentExperimentData, train_agent_ids: List[str]) -> None:
        train_agent_set = set(train_agent_ids)
        counts = {task_id: [0, 0] for task_id in data.all_item_ids}
        for benchmark, subject_id, responses in data.responses:
            agent_key = f"{benchmark}::{subject_id}"
            if agent_key not in train_agent_set:
                continue
            for task_id, response in responses.items():
                if task_id not in counts:
                    continue
                counts[task_id][0] += 1
                counts[task_id][1] += int(response)
        missing = [task_id for task_id, (n, _) in counts.items() if n == 0]
        if missing:
            raise ValueError(
                f"Cannot compute empirical solve-rate baseline; "
                f"{len(missing)} tasks have no train-agent responses. First 5: {missing[:5]}"
            )
        self._solve_rate_by_task = {
            task_id: float(k) / float(n) for task_id, (n, k) in counts.items()
        }

    def predict_probability(
        self, data: AgentExperimentData, agent_id: str, task_id: str
    ) -> float:
        if task_id not in self._solve_rate_by_task:
            raise ValueError(f"No empirical solve rate for task {task_id!r}")
        return self._solve_rate_by_task[task_id]


class OraclePredictor:
    """Oracle using full standard IRT abilities and item difficulties."""

    requires_train_irt = False

    def fit(self, data: AgentExperimentData, train_agent_ids: List[str]) -> None:
        pass

    def predict_probability(
        self, data: AgentExperimentData, agent_id: str, task_id: str
    ) -> float:
        _, subject_id = agent_id.split("::", 1)
        if subject_id not in data.full_abilities.index:
            raise ValueError(f"Agent {subject_id!r} has no full-IRT ability")
        if task_id not in data.full_items.index:
            raise ValueError(f"Task {task_id!r} has no full-IRT difficulty")
        return probability_from_theta(
            data.full_abilities.loc[subject_id, "ability"],
            data.full_items.loc[task_id, "b"],
        )
