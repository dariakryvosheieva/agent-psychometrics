"""Per-cell and sweep-level orchestration for the subset extrapolation experiment.

A "cell" is one (dataset, subset_size, seed) configuration. For each cell we
score every requested method:

  - empirical: no IRT needed. Score per-agent extrapolated mean.
  - llm_judge: fit fold IRT on observed subset, fit RidgeCV on LLM judge
    features, predict held-out difficulties, score IRT probabilities.
  - combined: same as llm_judge but with the Grouped (Embedding + LLM Judge)
    predictor.
  - oracle: same fold IRT step, but the predictor reads agent ability + task
    difficulty from the full IRT (loaded from `config.abilities_path` /
    `config.items_path`).

The fold IRT is trained at most once per cell (`load_dataset_for_fold`), and is
shared across all model-based methods.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from experiment_new_tasks.dataset import (
    ExperimentData,
    _load_responses,
    load_dataset_for_fold,
)
from experiment_new_tasks.pipeline import build_cv_predictors

from experiment_subset_extrapolation.config import SubsetExtrapolationConfig
from experiment_subset_extrapolation.evaluator import (
    AgentPrediction,
    evaluate_subset_extrapolation,
)
from experiment_subset_extrapolation.subset_sampler import sample_subset


# Map our method names -> the internal predictor name produced by
# experiment_new_tasks.pipeline.build_cv_predictors.
METHOD_TO_INTERNAL: Dict[str, str] = {
    "oracle": "oracle",
    "llm_judge": "llm_judge",
    "combined": "grouped",
}


def _load_task_universe(items_path: Path) -> List[str]:
    """All task IDs from the canonical full-IRT items.csv."""
    full_items = pd.read_csv(items_path, index_col=0)
    return list(full_items.index)


def _build_minimal_data_for_empirical(
    responses_path: Path,
    abilities_path: Path,
    items_path: Path,
    observed: List[str],
    heldout: List[str],
) -> ExperimentData:
    """Build an ExperimentData with only the fields needed by the empirical baseline.

    The empirical baseline does NOT need a fold IRT — only responses and the
    canonical agent universe (so the per-agent loop covers the same agents that
    the model methods would cover).
    """
    responses = _load_responses(responses_path)
    full_abilities = pd.read_csv(abilities_path, index_col=0)
    full_items = pd.read_csv(items_path, index_col=0)
    # Standardize column name to 'ability' (matches dataset._load_abilities).
    if "theta" in full_abilities.columns and "ability" not in full_abilities.columns:
        full_abilities = full_abilities.rename(columns={"theta": "ability"})

    # train_abilities is what the evaluator iterates over. For empirical we
    # want the canonical agent set, so use full_abilities here. train_items
    # is unused for the empirical path.
    return ExperimentData(
        responses=responses,
        train_abilities=full_abilities,
        train_items=full_items,
        full_abilities=full_abilities,
        full_items=full_items,
        train_tasks=observed,
        test_tasks=heldout,
    )


def _evaluate_methods(
    methods: List[str],
    data_for_empirical: Optional[ExperimentData],
    data_for_model: Optional[ExperimentData],
    base_cfg: Any,
    root: Path,
    observed: List[str],
    heldout: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Run each requested method, returning {method_name: {mae, n_agents}} or
    {method_name: {error}}."""
    out: Dict[str, Dict[str, Any]] = {}

    if "empirical" in methods:
        if data_for_empirical is None:
            raise ValueError("data_for_empirical must be provided when 'empirical' is requested")
        try:
            mae, preds = evaluate_subset_extrapolation(
                None, data_for_empirical, observed, heldout
            )
            out["empirical"] = {"mae": mae, "n_agents": len(preds)}
        except Exception as e:
            out["empirical"] = {"error": f"{type(e).__name__}: {e}"}

    model_methods = [m for m in methods if m != "empirical"]
    if model_methods:
        if data_for_model is None:
            for m in model_methods:
                out[m] = {"skipped": "fold IRT not available"}
            return out

        predictor_configs = build_cv_predictors(base_cfg, root)
        pc_by_name = {pc.name: pc for pc in predictor_configs}

        for method in model_methods:
            internal = METHOD_TO_INTERNAL.get(method)
            if internal is None or internal not in pc_by_name:
                out[method] = {"error": f"unknown method {method!r} (internal {internal!r})"}
                continue
            try:
                mae, preds = evaluate_subset_extrapolation(
                    pc_by_name[internal].predictor, data_for_model, observed, heldout
                )
                out[method] = {"mae": mae, "n_agents": len(preds)}
            except Exception as e:
                out[method] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
    return out


def run_one_cell(
    dataset: str,
    sweep_cfg: SubsetExtrapolationConfig,
    size: float,
    seed: int,
    root: Path,
) -> Dict[str, Any]:
    """Run a single (dataset, size, seed) configuration end-to-end.

    Returns a dict with per-method MAE and bookkeeping fields. The empirical
    baseline is computed even if the fold IRT fails.
    """
    t0 = time.time()
    base_cfg = sweep_cfg.base_config(dataset)
    abilities_path = root / base_cfg.abilities_path
    items_path = root / base_cfg.items_path
    responses_path = root / base_cfg.responses_path

    all_task_ids = _load_task_universe(items_path)
    observed, heldout = sample_subset(all_task_ids, size, seed)

    methods = list(sweep_cfg.methods)

    out: Dict[str, Any] = {
        "dataset": dataset,
        "size": float(size),
        "seed": int(seed),
        "n_total_tasks": len(all_task_ids),
        "n_observed": len(observed),
        "n_heldout": len(heldout),
    }

    # Empirical-only path (no IRT, fast). Always attempted.
    data_for_empirical: Optional[ExperimentData] = None
    if "empirical" in methods:
        try:
            data_for_empirical = _build_minimal_data_for_empirical(
                responses_path, abilities_path, items_path, observed, heldout
            )
        except Exception as e:
            out["empirical_setup_error"] = f"{type(e).__name__}: {e}"

    # Model-method path requires fold IRT.
    data_for_model: Optional[ExperimentData] = None
    model_methods = [m for m in methods if m != "empirical"]
    if model_methods:
        try:
            irt_cache_dir = sweep_cfg.cache_dir_for(dataset, size, seed)
            data_for_model = load_dataset_for_fold(
                abilities_path=abilities_path,
                items_path=items_path,
                responses_path=responses_path,
                train_tasks=observed,
                test_tasks=heldout,
                fold_idx=0,
                k_folds=1,
                split_seed=seed,
                irt_cache_dir=irt_cache_dir,
                exclude_unsolved=False,
            )
            out["irt_status"] = "trained"
        except Exception as e:
            out["irt_status"] = "failed"
            out["irt_error"] = f"{type(e).__name__}: {e}"

    out["methods"] = _evaluate_methods(
        methods=methods,
        data_for_empirical=data_for_empirical,
        data_for_model=data_for_model,
        base_cfg=base_cfg,
        root=root,
        observed=observed,
        heldout=heldout,
    )
    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


# ----- Sweep-level aggregation --------------------------------------------------


def _summarize_method_seeds(
    results_for_cell: List[Dict[str, Any]],
    method: str,
    target_n_seeds: int,
) -> Dict[str, Any]:
    """Pick first `target_n_seeds` successful runs for `method` (sorted by seed)
    and compute mean/std MAE across them."""
    import numpy as np

    rows: List[Tuple[int, float]] = []
    failures: List[Tuple[int, str]] = []
    for r in sorted(results_for_cell, key=lambda x: x["seed"]):
        method_result = r["methods"].get(method, {})
        if "mae" in method_result:
            rows.append((r["seed"], float(method_result["mae"])))
        else:
            failures.append(
                (r["seed"], method_result.get("error") or method_result.get("skipped") or "missing")
            )
        if len(rows) >= target_n_seeds:
            break

    if not rows:
        return {
            "method": method,
            "n_successful_seeds": 0,
            "seeds_used": [],
            "maes": [],
            "mean_mae": None,
            "std_mae": None,
            "failures_examined": failures[:10],
        }

    seeds_used, maes = zip(*rows)
    return {
        "method": method,
        "n_successful_seeds": len(rows),
        "seeds_used": list(seeds_used),
        "maes": list(maes),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        "failures_examined": failures[:10],
    }


def aggregate_sweep_results(
    raw_cell_results: List[Dict[str, Any]],
    sweep_cfg: SubsetExtrapolationConfig,
) -> Dict[str, Any]:
    """Group raw per-cell results by (dataset, size) and summarize per method.

    Args:
        raw_cell_results: Output of run_one_cell, one entry per (dataset, size, seed).
        sweep_cfg: The sweep configuration (for target_n_seeds, methods, etc.).

    Returns:
        Nested dict:
        {
            "config": sweep_cfg.to_dict(),
            "results": {
                dataset: {
                    str(size): {
                        method: {"mean_mae", "std_mae", "n_successful_seeds", ...},
                        ...,
                        "n_cells_attempted": int,
                        "n_irt_trained": int,
                        "n_irt_failed": int,
                    }
                }
            },
            "raw_cells": [...],
        }
    """
    by_cell: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in raw_cell_results:
        by_cell.setdefault((r["dataset"], float(r["size"])), []).append(r)

    out_results: Dict[str, Any] = {}
    for dataset in sweep_cfg.datasets:
        out_results[dataset] = {}
        for size in sweep_cfg.subset_sizes:
            if sweep_cfg.is_excluded(dataset, size):
                continue
            cell = by_cell.get((dataset, float(size)), [])
            method_summaries: Dict[str, Any] = {}
            for method in sweep_cfg.methods:
                method_summaries[method] = _summarize_method_seeds(
                    cell, method, sweep_cfg.target_n_seeds
                )
            n_irt_trained = sum(1 for r in cell if r.get("irt_status") == "trained")
            n_irt_failed = sum(1 for r in cell if r.get("irt_status") == "failed")
            out_results[dataset][f"{size:.4f}"] = {
                "size": float(size),
                "n_cells_attempted": len(cell),
                "n_irt_trained": n_irt_trained,
                "n_irt_failed": n_irt_failed,
                "methods": method_summaries,
            }

    return {
        "config": sweep_cfg.to_dict(),
        "results": out_results,
    }


def save_summary(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
