"""New-Agent adaptive testing simulation.

Pivot of the original adaptive-testing experiment from the New-Benchmarks
setting (predicted item difficulties) to the New-Agents setting:

    A developer has built a new (LLM, scaffold) agent and wants to estimate its
    ability on an existing benchmark using as few task evaluations as possible.

For each test agent we know binary responses on all benchmark tasks (used as
the "oracle" of what the agent would have answered). We simulate selecting
K = 1 ... max_tasks tasks adaptively, re-estimating theta after each response,
and record |theta_hat_K - theta_true|. theta_true is the MAP estimate using
all benchmark tasks under the same prior and item difficulties as the simulator.

We compare three methods:
    1. Random + weak prior N(0, 3^2)
    2. Fisher + weak prior N(0, 3^2)
    3. Fisher + IRT-Agent prior N(theta_prior, sigma_prior^2)
       where theta_prior = combine_theta(LLM, scaffold, "sum") from a
       Model+Scaffold IRT trained on the OTHER agents in the fold, and
       sigma_prior is the global RMSE of (theta_prior, theta_true) across
       folds.

Item difficulties for the simulator come from the same fold's train_items
(no leakage from the held-out agent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from experiment_adaptive_testing.cat_simulation import (
    FisherSelector,
    estimate_theta_mle,
)
from experiment_new_agents.config import ExperimentNewAgentsConfig
from experiment_new_agents.cross_validation import stable_k_fold_split_agent_pairs
from experiment_new_agents.dataset import (
    load_agent_model_scaffold_map,
    load_dataset_for_agent_fold,
    load_tagged_responses,
)
from swebench_irt.model_scaffold_combine import combine_theta


@dataclass
class NewAgentExperimentConfig:
    dataset: str
    responses_path: Path
    output_dir: Path
    irt_cache_dir: Path
    oracle_cache_dir: Path
    k_folds: int = 5
    split_seed: int = 0
    irt_epochs: int = 2000
    irt_device: str = "cuda"
    irt_lr: float = 0.01
    irt_model: str = "1d_1pl"
    theta_combine: str = "sum"
    max_tasks: int = 100
    weak_prior_sigma: float = 3.0
    n_random_subsets: int = 100
    seed: int = 42
    n_bootstrap: int = 10000
    bootstrap_seed: int = 0


@dataclass
class AgentSimulationRecord:
    fold_idx: int
    agent_key: str
    llm: str
    scaffold: str
    theta_true: float
    theta_prior: float
    response_dict: Dict[str, int]


def _theta_true_from_full_data(
    responses: Dict[str, int],
    item_difficulties: Dict[str, float],
    prior_sigma: float,
) -> float:
    """MAP estimate using all observed responses; the 'full-benchmark' reference."""

    task_ids = sorted(responses.keys() & item_difficulties.keys())
    if not task_ids:
        raise RuntimeError("Agent shares no tasks with the fold's item difficulties")
    y = [int(responses[t]) for t in task_ids]
    b = [float(item_difficulties[t]) for t in task_ids]
    return estimate_theta_mle(y, b, prior_mean=0.0, prior_sigma=prior_sigma)


def _collect_fold_records(
    config: NewAgentExperimentConfig,
    root: Path,
) -> Tuple[List[AgentSimulationRecord], Dict[int, Dict[str, float]]]:
    """Build per-fold IRT and produce one AgentSimulationRecord per test agent.

    Returns:
        records: one per (fold, test_agent).
        fold_item_diffs: {fold_idx: {item_id: b}} for use during simulation.
    """

    responses_path = root / config.responses_path
    tagged = load_tagged_responses(responses_path, config.dataset)
    agent_to_ms_pair = load_agent_model_scaffold_map(
        responses_path, config.dataset, tagged
    )
    agent_keys = sorted(agent_to_ms_pair.keys())

    folds = stable_k_fold_split_agent_pairs(
        agent_keys,
        agent_to_ms_pair,
        k=config.k_folds,
        seed=config.split_seed,
    )
    n_pairs = len(set(agent_to_ms_pair.values()))
    print(f"Agents: {len(agent_keys)}, (LLM, scaffold) pairs: {n_pairs}")
    print(f"Built {len(folds)} folds for {config.k_folds}-fold CV")

    response_by_key: Dict[str, Dict[str, int]] = {}
    for benchmark, subject_id, resp in tagged:
        response_by_key[f"{benchmark}::{subject_id}"] = {
            str(t): int(v) for t, v in resp.items()
        }

    records: List[AgentSimulationRecord] = []
    fold_item_diffs: Dict[int, Dict[str, float]] = {}

    for fold_idx, (train_agents, test_agents) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/{len(folds)}: "
              f"{len(train_agents)} train agents, {len(test_agents)} test agents")

        data = load_dataset_for_agent_fold(
            dataset=config.dataset,
            responses_path=responses_path,
            train_agents=train_agents,
            test_agents=test_agents,
            fold_idx=fold_idx,
            k_folds=config.k_folds,
            split_seed=config.split_seed,
            irt_cache_dir=root / config.irt_cache_dir,
            oracle_cache_dir=root / config.oracle_cache_dir,
            irt_model=config.irt_model,
            irt_epochs=config.irt_epochs,
            irt_device=config.irt_device,
            irt_lr=config.irt_lr,
            theta_combine=config.theta_combine,
            load_train_irt=True,
        )

        train_items = data.train_items
        if "b" not in train_items.columns:
            raise RuntimeError(f"Fold {fold_idx} train_items missing 'b' column")
        item_diff_map = {str(tid): float(train_items.loc[tid, "b"])
                         for tid in train_items.index}
        fold_item_diffs[fold_idx] = item_diff_map

        for agent_key in test_agents:
            if agent_key not in response_by_key:
                raise RuntimeError(f"No responses found for test agent {agent_key}")
            llm, scaffold = data.agent_to_ms_pair[agent_key]
            if llm not in data.train_model_abilities.index:
                raise RuntimeError(
                    f"Test agent {agent_key} has LLM {llm!r} not in train_model_abilities"
                )
            if scaffold not in data.train_scaffold_abilities.index:
                raise RuntimeError(
                    f"Test agent {agent_key} has scaffold {scaffold!r} "
                    "not in train_scaffold_abilities"
                )
            theta_prior = combine_theta(
                float(data.train_model_abilities.loc[llm, "theta"]),
                float(data.train_scaffold_abilities.loc[scaffold, "theta"]),
                combine=config.theta_combine,
                model_id=llm,
            )
            theta_true = _theta_true_from_full_data(
                response_by_key[agent_key], item_diff_map, config.weak_prior_sigma,
            )
            records.append(AgentSimulationRecord(
                fold_idx=fold_idx,
                agent_key=agent_key,
                llm=llm,
                scaffold=scaffold,
                theta_true=theta_true,
                theta_prior=theta_prior,
                response_dict=response_by_key[agent_key],
            ))

    return records, fold_item_diffs


def _bootstrap_mean_bands(
    errors: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI on the mean MAE across agents at each K.

    `errors` has shape (n_agents, n_steps). For each step:
        - point estimate: mean across agents
        - band: 2.5/97.5 percentiles of bootstrap-resampled means
          (resample agents with replacement; recompute mean).
    """
    n_agents = errors.shape[0]
    idx = rng.integers(0, n_agents, size=(n_bootstrap, n_agents))
    boot_means = errors[idx].mean(axis=1)  # (n_bootstrap, n_steps)
    alpha = (1.0 - ci) / 2.0
    mean = errors.mean(axis=0)
    lo = np.percentile(boot_means, 100.0 * alpha, axis=0)
    hi = np.percentile(boot_means, 100.0 * (1.0 - alpha), axis=0)
    return mean, lo, hi


def _compute_sigma_prior(records: List[AgentSimulationRecord]) -> float:
    """Global sigma_prior = RMSE(theta_prior, theta_true) across all records."""
    if not records:
        raise RuntimeError("No records to compute sigma_prior from")
    diffs = np.array([r.theta_prior - r.theta_true for r in records], dtype=np.float64)
    return float(np.sqrt(np.mean(diffs ** 2)))


def _simulate_random(
    record: AgentSimulationRecord,
    item_diffs: Dict[str, float],
    max_tasks: int,
    weak_prior_sigma: float,
    n_subsets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mean |theta_hat - theta_true| across n_subsets independent random orders.

    For each subset, follow a fixed random order task by task and record per-step
    absolute error. Returns array of length max_tasks (mean over subsets).
    """
    task_ids = sorted(item_diffs.keys() & record.response_dict.keys())
    if len(task_ids) < max_tasks:
        raise RuntimeError(
            f"Agent {record.agent_key} has only {len(task_ids)} shared tasks, "
            f"need {max_tasks}"
        )
    pool = np.array(task_ids)
    errors = np.zeros((n_subsets, max_tasks), dtype=np.float64)

    for s_idx in range(n_subsets):
        order = rng.permutation(pool)[:max_tasks].tolist()
        running_resps: List[int] = []
        running_diffs: List[float] = []
        for step in range(max_tasks):
            tid = order[step]
            running_resps.append(int(record.response_dict[tid]))
            running_diffs.append(float(item_diffs[tid]))
            theta_hat = estimate_theta_mle(
                running_resps, running_diffs,
                prior_mean=0.0, prior_sigma=weak_prior_sigma,
            )
            errors[s_idx, step] = abs(theta_hat - record.theta_true)
    return errors.mean(axis=0)


def _simulate_fisher(
    record: AgentSimulationRecord,
    item_diffs: Dict[str, float],
    max_tasks: int,
    prior_mean: float,
    prior_sigma: float,
) -> np.ndarray:
    """Greedy Fisher selection with a fixed Gaussian prior. Deterministic per agent.

    Returns an array of length max_tasks: |theta_hat_K - theta_true| at each K.
    """
    task_ids = sorted(item_diffs.keys() & record.response_dict.keys())
    if len(task_ids) < max_tasks:
        raise RuntimeError(
            f"Agent {record.agent_key} has only {len(task_ids)} shared tasks, "
            f"need {max_tasks}"
        )
    diffs_subset = {t: item_diffs[t] for t in task_ids}
    selector = FisherSelector(
        difficulties=diffs_subset,
        task_pool=task_ids,
        prior_sigma=prior_sigma,
        prior_mean=prior_mean,
    )
    selector.reset()
    errors = np.zeros(max_tasks, dtype=np.float64)
    for step in range(max_tasks):
        tid = selector.select_next()
        selector.update(tid, int(record.response_dict[tid]))
        errors[step] = abs(selector.theta_hat - record.theta_true)
    return errors


def run_new_agent_experiment(
    config: NewAgentExperimentConfig,
    root: Path,
) -> Dict[str, object]:
    print("=" * 70)
    print("ADAPTIVE TESTING: NEW AGENT SETTING")
    print("=" * 70)

    records, fold_item_diffs = _collect_fold_records(config, root)
    if not records:
        raise RuntimeError("No test-agent records collected")
    print(f"\nTotal test agents across folds: {len(records)}")

    sigma_prior = _compute_sigma_prior(records)
    prior_errors = np.array(
        [r.theta_prior - r.theta_true for r in records], dtype=np.float64,
    )
    prior_rmse = float(np.sqrt(np.mean(prior_errors ** 2)))
    prior_mae = float(np.mean(np.abs(prior_errors)))
    print(f"Global IRT-Agent prior RMSE  = {prior_rmse:.4f}")
    print(f"Global IRT-Agent prior MAE   = {prior_mae:.4f}")
    print(f"Using sigma_prior            = {sigma_prior:.4f}")

    rng = np.random.default_rng(config.seed)
    max_tasks = config.max_tasks

    random_errors = np.zeros((len(records), max_tasks), dtype=np.float64)
    fisher_weak_errors = np.zeros((len(records), max_tasks), dtype=np.float64)
    fisher_informed_errors = np.zeros((len(records), max_tasks), dtype=np.float64)

    print(f"\nSimulating {len(records)} agents x 3 methods x {max_tasks} steps...")
    for r_idx, record in enumerate(records):
        item_diffs = fold_item_diffs[record.fold_idx]
        random_errors[r_idx] = _simulate_random(
            record, item_diffs, max_tasks, config.weak_prior_sigma,
            config.n_random_subsets, rng,
        )
        fisher_weak_errors[r_idx] = _simulate_fisher(
            record, item_diffs, max_tasks,
            prior_mean=0.0, prior_sigma=config.weak_prior_sigma,
        )
        fisher_informed_errors[r_idx] = _simulate_fisher(
            record, item_diffs, max_tasks,
            prior_mean=record.theta_prior, prior_sigma=sigma_prior,
        )
        if (r_idx + 1) % 10 == 0 or r_idx + 1 == len(records):
            print(f"  {r_idx + 1}/{len(records)} agents simulated")

    steps = list(range(1, max_tasks + 1))
    results: Dict[str, object] = {
        "step": steps,
        "n_agents": len(records),
        "sigma_prior": sigma_prior,
        "prior_rmse": prior_rmse,
        "prior_mae": prior_mae,
        "n_bootstrap": config.n_bootstrap,
    }
    boot_rng = np.random.default_rng(config.bootstrap_seed)
    for label, arr in [
        ("random", random_errors),
        ("fisher_weak", fisher_weak_errors),
        ("fisher_informed", fisher_informed_errors),
    ]:
        mean, lo, hi = _bootstrap_mean_bands(arr, config.n_bootstrap, boot_rng)
        results[f"{label}_mae"] = mean.tolist()
        results[f"{label}_mae_lo"] = lo.tolist()
        results[f"{label}_mae_hi"] = hi.tolist()

    records_df = pd.DataFrame([
        {
            "fold_idx": r.fold_idx,
            "agent_key": r.agent_key,
            "llm": r.llm,
            "scaffold": r.scaffold,
            "theta_true": r.theta_true,
            "theta_prior": r.theta_prior,
        }
        for r in records
    ])
    results["per_agent_records"] = records_df.to_dict(orient="records")
    return results


def save_results(results: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    steps = results["step"]
    for i, step in enumerate(steps):
        rows.append({
            "step": step,
            "random_mae": results["random_mae"][i],
            "random_mae_lo": results["random_mae_lo"][i],
            "random_mae_hi": results["random_mae_hi"][i],
            "fisher_weak_mae": results["fisher_weak_mae"][i],
            "fisher_weak_mae_lo": results["fisher_weak_mae_lo"][i],
            "fisher_weak_mae_hi": results["fisher_weak_mae_hi"][i],
            "fisher_informed_mae": results["fisher_informed_mae"][i],
            "fisher_informed_mae_lo": results["fisher_informed_mae_lo"][i],
            "fisher_informed_mae_hi": results["fisher_informed_mae_hi"][i],
        })
    pd.DataFrame(rows).to_csv(output_dir / "error_curves.csv", index=False)

    summary = {
        "n_agents": int(results["n_agents"]),
        "sigma_prior": float(results["sigma_prior"]),
        "prior_rmse": float(results["prior_rmse"]),
        "prior_mae": float(results["prior_mae"]),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    per_agent = pd.DataFrame(results["per_agent_records"])
    per_agent.to_csv(output_dir / "per_agent_records.csv", index=False)
    print(f"Saved error_curves.csv, summary.json, per_agent_records.csv to {output_dir}")
