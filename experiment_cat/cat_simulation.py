"""Core CAT simulation: MLE ability estimation, Fisher task selection, and evaluation loop.

Convention: P(success) = sigmoid(theta - b), where higher b = harder task.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

from experiment_new_tasks.dataset import _load_binary_responses, _load_items


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predicted_difficulties(predictions_csv: Path) -> Dict[str, float]:
    """Load predicted difficulties from multi-benchmark experiment output.

    Expects CSV with columns: item_id, diff_pred, split, fold.
    """
    diffs: Dict[str, float] = {}
    with open(predictions_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row["diff_pred"]
            if val == "":
                continue
            diffs[row["item_id"]] = float(val)
    return diffs


def load_oracle_difficulties(items_csv: Path) -> Dict[str, float]:
    """Load ground truth IRT difficulties from items.csv (column 'b')."""
    items_df = _load_items(items_csv)
    return {str(tid): float(items_df.loc[tid, "b"]) for tid in items_df.index}


def load_and_verify_data(
    responses_path: Path,
    predictions_csv: Path,
    oracle_items_path: Path,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float], Dict[str, float], List[str], List[str]]:
    """Load all data and verify task coverage matches across sources.

    Returns (responses, pred_diffs, oracle_diffs, task_pool, agent_ids).
    Raises RuntimeError if the three sources don't cover the exact same tasks.
    """
    responses = _load_binary_responses(responses_path)
    pred_diffs = load_predicted_difficulties(predictions_csv)
    oracle_diffs = load_oracle_difficulties(oracle_items_path)

    agent_ids = sorted(responses.keys())

    pred_tasks = set(pred_diffs.keys())
    oracle_tasks = set(oracle_diffs.keys())
    response_tasks = set.intersection(*(set(responses[aid].keys()) for aid in agent_ids))

    missing_from_pred = (oracle_tasks & response_tasks) - pred_tasks
    missing_from_oracle = (pred_tasks & response_tasks) - oracle_tasks
    missing_from_responses = (pred_tasks & oracle_tasks) - response_tasks

    if missing_from_pred:
        raise RuntimeError(
            f"{len(missing_from_pred)} tasks in oracle/responses but not in predictions. "
            f"Ensure the multi-benchmark experiment produced predictions for all SWE-bench Pro tasks."
        )
    if missing_from_oracle:
        raise RuntimeError(
            f"{len(missing_from_oracle)} tasks in predictions/responses but not in oracle. "
            f"Ensure oracle IRT model covers all SWE-bench Pro tasks."
        )
    if missing_from_responses:
        raise RuntimeError(
            f"{len(missing_from_responses)} tasks in predictions/oracle but not in all agents' responses."
        )

    task_pool = sorted(pred_tasks & oracle_tasks & response_tasks)

    print(f"Loaded {len(agent_ids)} agents, {len(task_pool)} tasks")
    print(f"Predicted difficulties range: [{min(pred_diffs.values()):.2f}, {max(pred_diffs.values()):.2f}]")
    print(f"Oracle difficulties range: [{min(oracle_diffs.values()):.2f}, {max(oracle_diffs.values()):.2f}]")

    return responses, pred_diffs, oracle_diffs, task_pool, agent_ids


# ---------------------------------------------------------------------------
# MLE ability estimation (1PL)
# ---------------------------------------------------------------------------

def estimate_theta_mle(
    responses: List[int],
    difficulties: List[float],
    theta_init: float = 0.0,
    prior_sigma: float = 3.0,
    bounds: Tuple[float, float] = (-6.0, 6.0),
) -> float:
    """Estimate ability via MAP with a weak Gaussian prior.

    Minimizes the negative log-posterior:
        nll = -sum[y_j * log(P_j) + (1 - y_j) * log(1 - P_j)] + theta^2 / (2 * sigma^2)
    where P_j = sigmoid(theta - b_j).

    Gradient: -sum(y_j - P_j) + theta / sigma^2
    (from d/dtheta log sigmoid(theta - b) = 1 - P and d/dtheta log(1 - sigmoid) = -P)

    Returns theta_hat. With 0 observations, returns theta_init.
    """
    if len(responses) == 0:
        return theta_init

    y = np.array(responses, dtype=np.float64)
    b = np.array(difficulties, dtype=np.float64)
    sigma_sq = prior_sigma ** 2

    def neg_log_posterior(theta_scalar):
        theta = theta_scalar[0]
        p = np.clip(expit(theta - b), 1e-15, 1.0 - 1e-15)
        nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        nll += 0.5 * (theta ** 2) / sigma_sq
        return nll

    def neg_log_posterior_grad(theta_scalar):
        theta = theta_scalar[0]
        p = expit(theta - b)
        return np.array([-np.sum(y - p) + theta / sigma_sq])

    result = minimize(
        neg_log_posterior,
        x0=[theta_init],
        jac=neg_log_posterior_grad,
        method="L-BFGS-B",
        bounds=[bounds],
    )
    return float(result.x[0])


# ---------------------------------------------------------------------------
# Task selectors
# ---------------------------------------------------------------------------

class TaskSelector(ABC):
    """Common interface for task selection strategies."""

    @abstractmethod
    def reset(self) -> None:
        """Reset state for a new agent."""
        ...

    @abstractmethod
    def select_next(self) -> str:
        """Select and return the next task_id to administer."""
        ...

    @abstractmethod
    def update(self, task_id: str, response: int) -> None:
        """Update internal state after observing a response."""
        ...

    @abstractmethod
    def score(self) -> float:
        """Current score for this agent (used for Spearman correlation)."""
        ...


class FisherSelector(TaskSelector):
    """Select tasks by maximizing Fisher information; score via MLE ability."""

    def __init__(self, difficulties: Dict[str, float], task_pool: List[str],
                 prior_sigma: float = 3.0):
        self.difficulties = difficulties
        self.task_pool = task_pool
        self.prior_sigma = prior_sigma
        self.remaining: List[str] = []
        self.theta_hat: float = 0.0
        self.administered_responses: List[int] = []
        self.administered_diffs: List[float] = []

    def reset(self) -> None:
        self.remaining = list(self.task_pool)
        self.theta_hat = 0.0
        self.administered_responses = []
        self.administered_diffs = []

    def select_next(self) -> str:
        best_idx = 0
        best_info = -1.0
        for i, tid in enumerate(self.remaining):
            p = expit(self.theta_hat - self.difficulties[tid])
            info = float(p * (1.0 - p))
            if info > best_info:
                best_info = info
                best_idx = i
        return self.remaining.pop(best_idx)

    def update(self, task_id: str, response: int) -> None:
        self.administered_responses.append(response)
        self.administered_diffs.append(self.difficulties[task_id])
        self.theta_hat = estimate_theta_mle(
            self.administered_responses, self.administered_diffs,
            theta_init=self.theta_hat, prior_sigma=self.prior_sigma,
        )

    def score(self) -> float:
        return self.theta_hat


class RandomSelector(TaskSelector):
    """Select tasks in a fixed random order; score via accuracy."""

    def __init__(self, task_order: List[str]):
        self.task_order = task_order
        self.step: int = 0
        self.n_correct: int = 0
        self.n_total: int = 0

    def reset(self) -> None:
        self.step = 0
        self.n_correct = 0
        self.n_total = 0

    def select_next(self) -> str:
        tid = self.task_order[self.step]
        return tid

    def update(self, task_id: str, response: int) -> None:
        self.step += 1
        self.n_correct += response
        self.n_total += 1

    def score(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_correct / self.n_total


# ---------------------------------------------------------------------------
# Common simulation loop
# ---------------------------------------------------------------------------

def run_method(
    selector: TaskSelector,
    agent_ids: List[str],
    responses: Dict[str, Dict[str, int]],
    gt_values: np.ndarray,
    max_steps: int,
    label: str,
) -> List[float]:
    """Run a selection method for all agents, return Spearman at each step."""
    print(f"Running {label}...")
    agent_scores: Dict[str, List[float]] = {}
    for aid in agent_ids:
        selector.reset()
        scores: List[float] = []
        for _ in range(max_steps):
            tid = selector.select_next()
            selector.update(tid, responses[aid][tid])
            scores.append(selector.score())
        agent_scores[aid] = scores

    spearman_values: List[float] = []
    for step in range(max_steps):
        step_scores = np.array([agent_scores[aid][step] for aid in agent_ids])
        corr, _ = spearmanr(step_scores, gt_values)
        spearman_values.append(float(corr))

    return spearman_values


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    responses_path: Path
    oracle_items_path: Path
    predictions_csv: Path
    max_steps: int = 200
    seed: int = 42
    prior_sigma: float = 3.0


def run_experiment(config: ExperimentConfig) -> Dict[str, List[float]]:
    """Run the full CAT experiment with three methods.

    Returns dict with keys 'step', 'fisher_predicted', 'fisher_oracle', 'random',
    each a list of floats (Spearman correlations at each step).
    """
    responses, pred_diffs, oracle_diffs, task_pool, agent_ids = load_and_verify_data(
        config.responses_path, config.predictions_csv, config.oracle_items_path,
    )

    max_steps = min(config.max_steps, len(task_pool))

    # Ground truth: each agent's full-benchmark accuracy over the task pool
    gt_values = np.array([
        sum(responses[aid][tid] for tid in task_pool) / len(task_pool)
        for aid in agent_ids
    ])
    print(f"Ground truth accuracies: min={gt_values.min():.3f}, "
          f"max={gt_values.max():.3f}, mean={gt_values.mean():.3f}")

    # Fisher (Predicted)
    fisher_pred_spearman = run_method(
        FisherSelector(pred_diffs, task_pool, config.prior_sigma),
        agent_ids, responses, gt_values, max_steps, "Fisher (Predicted)",
    )

    # Fisher (Oracle)
    fisher_oracle_spearman = run_method(
        FisherSelector(oracle_diffs, task_pool, config.prior_sigma),
        agent_ids, responses, gt_values, max_steps, "Fisher (Oracle)",
    )

    # Random
    rng = np.random.default_rng(config.seed)
    random_order = list(task_pool)
    rng.shuffle(random_order)
    random_spearman = run_method(
        RandomSelector(random_order),
        agent_ids, responses, gt_values, max_steps, "Random",
    )

    return {
        "step": list(range(1, max_steps + 1)),
        "fisher_predicted": fisher_pred_spearman,
        "fisher_oracle": fisher_oracle_spearman,
        "random": random_spearman,
    }
