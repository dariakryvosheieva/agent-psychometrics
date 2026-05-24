"""Per-agent benchmark-score MAE evaluator for the subset extrapolation experiment.

Given an observed subset and a held-out remainder, predict each agent's overall
% correct on the entire benchmark by combining their true successes on the
observed subset with predicted-probability times trials on the held-out subset
(or, for the empirical baseline, by extrapolating their observed rate).
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from experiment_new_tasks.cross_validation import CVPredictor
from experiment_new_tasks.dataset import ExperimentData


ResponseValue = Union[int, Dict[str, int]]


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


@dataclass
class AgentPrediction:
    """Per-agent prediction record for one (size, seed) evaluation."""

    agent_id: str
    actual_pct: float
    predicted_pct: float
    observed_trials: int
    total_trials: int


def evaluate_subset_extrapolation(
    predictor: Optional[CVPredictor],
    data: ExperimentData,
    observed_tasks: List[str],
    heldout_tasks: List[str],
    *,
    calibrate: bool = False,
) -> Tuple[float, List[AgentPrediction]]:
    """Evaluate a predictor on a single (size, seed) subset draw.

    For each agent:
      actual_pct    = total_successes / total_trials  (over observed + heldout)
      empirical     = observed_successes / observed_trials
      method        = (observed_successes + Σ_heldout p(a,t) * trials(a,t)) / total_trials

    When `predictor is None`, the empirical-subset baseline is computed (no IRT
    fit needed).

    Calibration (`calibrate=True`, ignored when `predictor is None`):
    The fold IRT theta lives on a slightly different scale than the full-data
    universe (see investigation in repo history), so raw predictions can carry
    a per-agent location bias. To correct it, we compute each agent's mean
    predicted probability on the OBSERVED subset, take the difference to their
    observed actual rate, and add that constant shift to every held-out
    prediction (clipped to [0, 1]). This is an unbiased correction in the
    sense that it forces the predicted pass-rate on observed tasks to equal
    the actual pass-rate, while preserving the model's relative ranking of
    held-out tasks.

    Args:
        predictor: A fitted-or-fittable CVPredictor, or None for the baseline.
            If non-None, it will be `fit()`-ed on `observed_tasks` here.
        data: ExperimentData with responses and (for non-baseline) train IRT.
        observed_tasks: Task IDs the benchmark designer evaluated.
        heldout_tasks: Task IDs to extrapolate to.
        calibrate: When True, apply per-agent location calibration described above.

    Returns:
        (mean_mae_across_agents, per_agent_predictions). Agents with zero
        observed trials are skipped silently for the empirical baseline (they
        cannot be extrapolated); for the model the baseline-only constraint
        also applies because the IRT fit drops such agents anyway.
    """
    if predictor is not None:
        predictor.fit(data, observed_tasks)

    # When calibrating, we need predicted probabilities on BOTH observed and
    # held-out tasks. DifficultyPredictorAdapter lazily caches predictions for
    # `data.test_tasks` only, so for observed tasks the lookup fails. Force a
    # cache fill over observed + heldout by calling predict_probability with a
    # widened `test_tasks` view once. Subsequent calls in the per-agent loop
    # use the original `data` and hit the cache.
    if calibrate and predictor is not None and observed_tasks:
        widened = replace(data, test_tasks=list(observed_tasks) + list(heldout_tasks))
        sample_agent = data.train_abilities.index[0]
        _ = predictor.predict_probability(widened, sample_agent, observed_tasks[0])

    preds: List[AgentPrediction] = []

    for agent_id in data.train_abilities.index:
        agent_resp = data.responses.get(agent_id, {})

        # Pass 1: observed tasks. Always tally actuals; also tally predicted
        # success-mass on observed when calibrating (so we can compute the
        # per-agent shift).
        obs_succ = 0
        obs_tr = 0
        obs_pred_mass = 0.0  # only used when calibrate=True
        for t in observed_tasks:
            if t in agent_resp:
                s, n = cell_successes_and_trials(agent_resp[t])
                obs_succ += s
                obs_tr += n
                if calibrate and predictor is not None:
                    p = predictor.predict_probability(data, agent_id, t)
                    obs_pred_mass += p * n

        # Per-agent calibration shift in probability space. By construction
        # this forces mean(predicted_p on observed) = mean(actual on observed).
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
                    p = predictor.predict_probability(data, agent_id, t)
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
            "check that data.train_abilities is non-empty and responses cover the "
            "observed/heldout tasks."
        )

    mae = float(np.mean([abs(p.actual_pct - p.predicted_pct) for p in preds]))
    return mae, preds
