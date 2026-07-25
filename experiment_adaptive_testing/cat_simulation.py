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

from experiment_new_tasks.dataset import _load_responses, _load_items


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
    responses = _load_responses(responses_path)
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
    prior_mean: float = 0.0,
    prior_sigma: float = 3.0,
    bounds: Tuple[float, float] = (-6.0, 6.0),
) -> float:
    """Estimate ability via MAP with a Gaussian prior N(prior_mean, prior_sigma^2).

    Minimizes the negative log-posterior:
        nll = -sum[y_j * log(P_j) + (1 - y_j) * log(1 - P_j)] + (theta - mu)^2 / (2 * sigma^2)
    where P_j = sigmoid(theta - b_j) and mu = prior_mean.

    Returns theta_hat. With 0 observations, returns prior_mean (the prior mode).
    """
    if len(responses) == 0:
        return prior_mean

    y = np.array(responses, dtype=np.float64)
    b = np.array(difficulties, dtype=np.float64)
    sigma_sq = prior_sigma ** 2

    def neg_log_posterior(theta_scalar):
        theta = theta_scalar[0]
        p = np.clip(expit(theta - b), 1e-15, 1.0 - 1e-15)
        nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        nll += 0.5 * ((theta - prior_mean) ** 2) / sigma_sq
        return nll

    def neg_log_posterior_grad(theta_scalar):
        theta = theta_scalar[0]
        p = expit(theta - b)
        return np.array([-np.sum(y - p) + (theta - prior_mean) / sigma_sq])

    init = theta_init if theta_init != 0.0 else prior_mean
    result = minimize(
        neg_log_posterior,
        x0=[init],
        jac=neg_log_posterior_grad,
        method="L-BFGS-B",
        bounds=[bounds],
    )
    return float(result.x[0])


# ---------------------------------------------------------------------------
# Reliability evaluation
# ---------------------------------------------------------------------------

def compute_empirical_reliability(
    theta_hats: np.ndarray,
    fisher_infos: np.ndarray,
) -> float:
    """Empirical reliability: 1 - mean(1/I) / var(θ̂).

    Mirrors mirt::empirical_rxx from the R reference implementation.
    1/I is the asymptotic variance of each θ̂ estimate (no SE intermediate).
    """
    theta_var = np.var(theta_hats, ddof=1)
    if theta_var < 1e-15:
        return float("nan")
    mean_estimate_var = np.mean(1.0 / fisher_infos)
    return float(1.0 - mean_estimate_var / theta_var)


def compute_agent_stats(
    administered_ids: Dict[str, List[str]],
    responses: Dict[str, Dict[str, int]],
    oracle_diffs: Dict[str, float],
    agent_ids: List[str],
    prior_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-agent (theta_hat, fisher_info) for one snapshot of administered tasks.

    All agents are evaluated using oracle (true IRT) difficulties for MLE
    and Fisher information, regardless of how tasks were selected.
    """
    theta_hats = []
    fisher_infos = []
    for aid in agent_ids:
        task_ids = administered_ids[aid]
        diffs = [oracle_diffs[tid] for tid in task_ids]
        resps = [responses[aid][tid] for tid in task_ids]
        theta = estimate_theta_mle(resps, diffs, prior_sigma=prior_sigma)
        b = np.array(diffs)
        p = expit(theta - b)
        info = float(np.sum(p * (1.0 - p)))
        theta_hats.append(theta)
        fisher_infos.append(info)
    return np.array(theta_hats), np.array(fisher_infos)


def summarize_reliability_bands(
    per_trajectory: np.ndarray,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step mean and percentile band across trajectories.

    `per_trajectory` has shape (n_trajectories, n_steps). Returns
    (mean, lo, hi), each of length n_steps, where lo/hi are the
    2.5/97.5 percentiles for ci=0.95.
    """
    alpha = (1.0 - ci) / 2.0
    mean = np.nanmean(per_trajectory, axis=0)
    lo = np.nanpercentile(per_trajectory, 100.0 * alpha, axis=0)
    hi = np.nanpercentile(per_trajectory, 100.0 * (1.0 - alpha), axis=0)
    return mean, lo, hi


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
                 prior_sigma: float = 3.0, prior_mean: float = 0.0):
        self.difficulties = difficulties
        self.task_pool = task_pool
        self.prior_sigma = prior_sigma
        self.prior_mean = prior_mean
        self.remaining: List[str] = []
        self.theta_hat: float = prior_mean
        self.administered_responses: List[int] = []
        self.administered_diffs: List[float] = []

    def reset(self) -> None:
        self.remaining = list(self.task_pool)
        self.theta_hat = self.prior_mean
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
            theta_init=self.theta_hat, prior_mean=self.prior_mean,
            prior_sigma=self.prior_sigma,
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
    max_steps: int,
    label: str,
) -> Dict[str, List[str]]:
    """Run a selection method for all agents, return administered task IDs."""
    print(f"Running {label}...")
    administered_ids: Dict[str, List[str]] = {}
    for aid in agent_ids:
        selector.reset()
        task_ids: List[str] = []
        for _ in range(max_steps):
            tid = selector.select_next()
            selector.update(tid, responses[aid][tid])
            task_ids.append(tid)
        administered_ids[aid] = task_ids
    return administered_ids


def run_fisher_from_init(
    initial_tid: str,
    difficulties: Dict[str, float],
    task_pool: List[str],
    agent_ids: List[str],
    responses: Dict[str, Dict[str, int]],
    max_steps: int,
    prior_sigma: float,
) -> Dict[str, List[str]]:
    """Run greedy Fisher for all agents, forcing the first administered task."""
    administered_ids: Dict[str, List[str]] = {}
    for aid in agent_ids:
        selector = FisherSelector(difficulties, task_pool, prior_sigma)
        selector.reset()
        selector.remaining.remove(initial_tid)
        selector.update(initial_tid, responses[aid][initial_tid])
        task_ids = [initial_tid]
        for _ in range(max_steps - 1):
            tid = selector.select_next()
            selector.update(tid, responses[aid][tid])
            task_ids.append(tid)
        administered_ids[aid] = task_ids
    return administered_ids


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
    n_init_tasks: int = 100
    n_random_subsets: int = 100


def _fisher_reliability_per_trajectory(
    label: str,
    difficulties: Dict[str, float],
    task_pool: List[str],
    agent_ids: List[str],
    responses: Dict[str, Dict[str, int]],
    oracle_diffs: Dict[str, float],
    max_steps: int,
    prior_sigma: float,
    initial_tasks: List[str],
) -> np.ndarray:
    """For each initial task, run greedy Fisher and return reliability per step.

    Shape: (len(initial_tasks), max_steps).
    """
    print(f"Running {label}: {len(initial_tasks)} initial tasks × {max_steps} steps...")
    per_traj = np.empty((len(initial_tasks), max_steps), dtype=np.float64)
    for t_idx, initial_tid in enumerate(initial_tasks):
        admin = run_fisher_from_init(
            initial_tid, difficulties, task_pool, agent_ids, responses,
            max_steps, prior_sigma,
        )
        for step in range(1, max_steps + 1):
            prefix = {aid: admin[aid][:step] for aid in agent_ids}
            theta_hats, fisher_infos = compute_agent_stats(
                prefix, responses, oracle_diffs, agent_ids, prior_sigma,
            )
            per_traj[t_idx, step - 1] = compute_empirical_reliability(theta_hats, fisher_infos)
    return per_traj


def _random_reliability_per_subset(
    task_pool: List[str],
    agent_ids: List[str],
    responses: Dict[str, Dict[str, int]],
    oracle_diffs: Dict[str, float],
    max_steps: int,
    prior_sigma: float,
    n_subsets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each subset size k, draw n_subsets independent random subsets of
    size k and return reliability per (subset, size).

    Shape: (n_subsets, max_steps).
    """
    print(f"Running Random: {n_subsets} subsets × {max_steps} sizes...")
    per_subset = np.empty((n_subsets, max_steps), dtype=np.float64)
    pool_array = np.array(task_pool)
    for step in range(1, max_steps + 1):
        for s_idx in range(n_subsets):
            subset = rng.choice(pool_array, size=step, replace=False).tolist()
            admin = {aid: subset for aid in agent_ids}
            theta_hats, fisher_infos = compute_agent_stats(
                admin, responses, oracle_diffs, agent_ids, prior_sigma,
            )
            per_subset[s_idx, step - 1] = compute_empirical_reliability(theta_hats, fisher_infos)
    return per_subset


def run_experiment(config: ExperimentConfig) -> Dict[str, List[float]]:
    """Run the full CAT experiment with three methods.

    Returns a dict with keys 'step' plus a (mean, lo, hi) triple per method:
    'fisher_predicted_reliability', 'fisher_predicted_reliability_lo',
    'fisher_predicted_reliability_hi', and likewise for fisher_oracle and random.
    For the Fisher methods the band is over `n_init_tasks` trajectories that
    each start from a different uniformly-sampled initial task; for Random it
    is over `n_random_subsets` independent random subsets per size. All bands
    are 2.5/97.5 percentiles.
    """
    responses, pred_diffs, oracle_diffs, task_pool, agent_ids = load_and_verify_data(
        config.responses_path, config.predictions_csv, config.oracle_items_path,
    )

    max_steps = min(config.max_steps, len(task_pool))

    rng = np.random.default_rng(config.seed)

    n_init = min(config.n_init_tasks, len(task_pool))
    if n_init < config.n_init_tasks:
        print(f"Warning: requested {config.n_init_tasks} initial tasks but pool has "
              f"only {len(task_pool)}; using {n_init}.")
    initial_tasks = rng.choice(task_pool, size=n_init, replace=False).tolist()

    fisher_pred_traj = _fisher_reliability_per_trajectory(
        "Fisher (Predicted)", pred_diffs, task_pool, agent_ids, responses,
        oracle_diffs, max_steps, config.prior_sigma, initial_tasks,
    )
    fisher_oracle_traj = _fisher_reliability_per_trajectory(
        "Fisher (Oracle)", oracle_diffs, task_pool, agent_ids, responses,
        oracle_diffs, max_steps, config.prior_sigma, initial_tasks,
    )
    random_traj = _random_reliability_per_subset(
        task_pool, agent_ids, responses, oracle_diffs, max_steps,
        config.prior_sigma, config.n_random_subsets, rng,
    )

    results: Dict[str, List[float]] = {"step": list(range(1, max_steps + 1))}
    for method_key, traj in [
        ("fisher_predicted_reliability", fisher_pred_traj),
        ("fisher_oracle_reliability", fisher_oracle_traj),
        ("random_reliability", random_traj),
    ]:
        mean, lo, hi = summarize_reliability_bands(traj)
        results[method_key] = mean.tolist()
        results[method_key + "_lo"] = lo.tolist()
        results[method_key + "_hi"] = hi.tolist()

    return results
