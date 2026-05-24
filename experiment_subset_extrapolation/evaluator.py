"""Per-agent benchmark-score MAE evaluator for the subset extrapolation experiment.

Given an observed subset and a held-out remainder, predict each agent's overall
% correct on the entire benchmark by combining their true successes on the
observed subset with predicted-probability times trials on the held-out subset
(or, for the empirical baseline, by extrapolating their observed rate).

This module is backend-agnostic: it takes a `CellPredictor` Protocol (or `None`
for the empirical baseline) and never assumes a particular IRT or feature
predictor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


ResponseValue = Union[int, Dict[str, int]]


class CellPredictor(Protocol):
    """Minimal predictor interface for this evaluator.

    Implementations: `MultiBenchCellPredictor`, `OracleCellPredictor`.
    The `prepare_for_tasks` hook lets a predictor compute and cache all
    needed held-out predictions up-front (e.g., one batched Ridge call).
    """

    def predict_probability(self, agent_id: str, task_id: str) -> float: ...

    def prepare_for_tasks(self, observed_tasks: List[str], heldout_tasks: List[str]) -> None:
        ...


@dataclass
class CellData:
    """All the per-cell context the evaluator needs.

    agent_ids: canonical target-benchmark agent set we iterate over.
    responses: {agent_id: {task_id: ResponseValue}}
    """

    agent_ids: List[str]
    responses: Dict[str, Dict[str, ResponseValue]]


@dataclass
class AgentPrediction:
    """Per-agent prediction record for one (size, seed) evaluation."""

    agent_id: str
    actual_pct: float
    predicted_pct: float
    observed_trials: int
    total_trials: int


def cell_successes_and_trials(resp: ResponseValue) -> Tuple[int, int]:
    """Unify binary (int 0/1) and binomial ({"successes","trials"}) cells.

    Binary cells become (value, 1). Binomial cells become (successes, trials).
    Fails loudly on malformed input.
    """
    if isinstance(resp, dict):
        if "successes" not in resp or "trials" not in resp:
            raise ValueError(
                f"Malformed binomial response (missing 'successes'/'trials'): {resp!r}"
            )
        s = int(resp["successes"])
        n = int(resp["trials"])
        if s < 0 or n < 0 or s > n:
            raise ValueError(f"Invalid binomial cell: {resp!r}")
        return s, n
    if isinstance(resp, (int, float, bool)):
        v = int(resp)
        if v not in (0, 1):
            raise ValueError(f"Binary response must be 0 or 1, got {resp!r}")
        return v, 1
    raise ValueError(f"Unsupported response value type: {type(resp).__name__} ({resp!r})")


def evaluate_subset_extrapolation(
    predictor: Optional[CellPredictor],
    data: CellData,
    observed_tasks: List[str],
    heldout_tasks: List[str],
    *,
    calibrate: bool = False,
) -> Tuple[float, List[AgentPrediction]]:
    """Evaluate a predictor (or the empirical baseline) on one fold.

    For each agent:
      actual_pct    = total_successes / total_trials  (over observed + heldout)
      empirical     = observed_successes / observed_trials
      method        = (observed_successes + Σ_heldout p(a,t) * trials(a,t)) / total_trials

    When `predictor is None`, computes the empirical baseline (no IRT fit).

    Calibration (per-agent constant shift): forces each agent's predicted
    pass-rate on observed tasks to equal their actual observed rate, by adding
    `shift = (obs_successes − obs_pred_mass) / obs_trials` to held-out
    predictions (clipped to [0, 1]). Ranking of held-out tasks is preserved.
    """
    if predictor is not None:
        predictor.prepare_for_tasks(list(observed_tasks), list(heldout_tasks))

    preds: List[AgentPrediction] = []

    for agent_id in data.agent_ids:
        agent_resp = data.responses.get(agent_id, {})

        # Pass 1: observed tasks. Always tally actuals; also tally predicted
        # success-mass on observed when calibrating (so we can compute the
        # per-agent shift).
        obs_succ = 0
        obs_tr = 0
        obs_pred_mass = 0.0
        for t in observed_tasks:
            if t in agent_resp:
                s, n = cell_successes_and_trials(agent_resp[t])
                obs_succ += s
                obs_tr += n
                if calibrate and predictor is not None:
                    p = predictor.predict_probability(agent_id, t)
                    obs_pred_mass += p * n

        shift = 0.0
        if calibrate and predictor is not None and obs_tr > 0:
            shift = (obs_succ - obs_pred_mass) / obs_tr

        # Pass 2: held-out tasks.
        held_succ = 0
        held_tr = 0
        method_held_pred = 0.0
        for t in heldout_tasks:
            if t in agent_resp:
                s, n = cell_successes_and_trials(agent_resp[t])
                held_succ += s
                held_tr += n
                if predictor is not None:
                    p = predictor.predict_probability(agent_id, t)
                    if calibrate:
                        p = max(0.0, min(1.0, p + shift))
                    method_held_pred += p * n

        total_tr = obs_tr + held_tr
        if total_tr == 0:
            continue
        actual = (obs_succ + held_succ) / total_tr

        if predictor is None:
            if obs_tr == 0:
                continue
            predicted = obs_succ / obs_tr
        else:
            predicted = (obs_succ + method_held_pred) / total_tr

        preds.append(
            AgentPrediction(
                agent_id=str(agent_id),
                actual_pct=float(actual),
                predicted_pct=float(predicted),
                observed_trials=int(obs_tr),
                total_trials=int(total_tr),
            )
        )

    if not preds:
        raise ValueError(
            "evaluate_subset_extrapolation produced zero per-agent predictions; "
            "check that data.agent_ids is non-empty and responses cover the "
            "observed/heldout tasks."
        )

    mae = float(np.mean([abs(p.actual_pct - p.predicted_pct) for p in preds]))
    return mae, preds


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class MultiBenchCellPredictor:
    """Probability backend powered by a multi-benchmark IRT + held-out Ridge.

    P(success | agent, task) = sigmoid(theta_model[m] + theta_scaffold[s] - b(task))
    where:
      - (m, s) come from the agent's parsed (model, scaffold)
      - b(task) is the IRT's b for observed tasks (in training set) and the
        Ridge's predicted b for held-out tasks.

    Agents that don't decompose (split_agent_name returns None, or the IRT
    has no theta for the resulting model/scaffold) are NOT silently handled
    here — the caller MUST filter the agent set up-front. We raise loudly if
    asked to predict for one.
    """

    def __init__(
        self,
        *,
        target_dataset: str,
        target_bench: str,
        irt,
        heldout_predictor,
    ):
        self.target_dataset = target_dataset
        self.target_bench = target_bench
        self._irt = irt
        self._heldout = heldout_predictor
        self._heldout_b: Dict[str, float] = {}
        self._prepared = False

    def prepare_for_tasks(self, observed_tasks: List[str], heldout_tasks: List[str]) -> None:
        if self._prepared:
            return
        if heldout_tasks:
            self._heldout_b = self._heldout.predict(list(heldout_tasks))
        else:
            self._heldout_b = {}
        # Sanity: every observed task should have an IRT b (training items).
        missing_obs = [t for t in observed_tasks if t not in self._irt.diff_by_item]
        if missing_obs:
            raise RuntimeError(
                f"{len(missing_obs)} observed tasks have no IRT difficulty (first: "
                f"{missing_obs[:3]}). The IRT should have trained on every observed task."
            )
        self._prepared = True

    def predict_probability(self, agent_id: str, task_id: str) -> float:
        pair = self._irt.agent_to_ms.get((self.target_bench, agent_id))
        if pair is None:
            raise KeyError(
                f"No (model, scaffold) parse for {self.target_bench}::{agent_id}. "
                "Filter unparseable agents from the evaluation set before constructing "
                "this predictor (see multibench_trainer.parseable_agents_for)."
            )
        model, scaffold = pair
        theta_m = self._irt.theta_by_model.get(model)
        theta_s = self._irt.theta_by_scaffold.get(scaffold)
        if theta_m is None or theta_s is None:
            raise KeyError(
                f"IRT has no theta for model={model!r} or scaffold={scaffold!r} "
                f"(agent {self.target_bench}::{agent_id}). "
                "Filter such agents before evaluation."
            )
        b = self._irt.diff_by_item.get(task_id)
        if b is None:
            b = self._heldout_b.get(task_id)
        if b is None:
            raise KeyError(
                f"No b for task {task_id!r}: not in IRT training items "
                f"({len(self._irt.diff_by_item)}) and not in held-out predictions "
                f"({len(self._heldout_b)}). Did you call prepare_for_tasks first?"
            )
        return _sigmoid(theta_m + theta_s - b)


class OracleCellPredictor:
    """Probability backend using the canonical full single-benchmark IRT.

    Loads abilities.csv + items.csv from the dataset's canonical IRT dir and
    scores p = sigmoid(theta_agent - b_task) for every (agent, task).
    """

    def __init__(self, *, abilities_path, items_path):
        import pandas as pd

        ab = pd.read_csv(abilities_path, index_col=0)
        it = pd.read_csv(items_path, index_col=0)
        if "theta" in ab.columns and "ability" not in ab.columns:
            ab = ab.rename(columns={"theta": "ability"})
        self._abilities = ab
        self._items = it

    def prepare_for_tasks(self, observed_tasks: List[str], heldout_tasks: List[str]) -> None:
        pass

    def predict_probability(self, agent_id: str, task_id: str) -> float:
        theta = float(self._abilities.loc[agent_id, "ability"])
        b = float(self._items.loc[task_id, "b"])
        return _sigmoid(theta - b)
