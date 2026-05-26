"""Per-cell and sweep-level orchestration for the subset extrapolation experiment.

A "cell" is one (dataset, count, seed) configuration. For each cell we
score every requested method:

  - empirical: no IRT needed. Score per-agent extrapolated mean from observed.
  - combined_calibrated: train a multi-benchmark 1PL IRT on the 3 non-target
    benchmarks + the target subset; cross-benchmark Ridge predicts held-out
    target difficulties from features; combine via sigmoid + per-agent
    calibration shift to estimate each agent's overall %.
  - oracle: same scoring formula but with θ and b from the canonical full
    single-benchmark IRT — represents the IRT model's best possible
    extrapolation.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from experiment_subset_extrapolation.config import SubsetExtrapolationConfig
from experiment_subset_extrapolation.evaluator import (
    AgentPrediction,
    CellData,
    MultiBenchCellPredictor,
    OracleCellPredictor,
    evaluate_subset_extrapolation,
)
from experiment_subset_extrapolation.multibench_trainer import (
    DATASET_TO_BENCH,
    RESPONSES_PATHS,
    parseable_agents_for,
)
from experiment_subset_extrapolation.subset_sampler import sample_subset_by_count


# --- Per-target-dataset response loading (CellData) -------------------------

def _load_response_iter(target_dataset: str):
    """Iterate (subject_id, responses) tuples from the dataset's response file."""
    from utils import difficulty_prediction as base
    from utils.difficulty_prediction import (
        load_all_responses_terminal,
        load_all_responses_generic,
    )

    bench = DATASET_TO_BENCH[target_dataset]
    pth = str(RESPONSES_PATHS[bench])
    if bench == "terminal_bench":
        return load_all_responses_terminal(pth)
    if bench == "gso":
        return load_all_responses_generic(path=pth, normalize_item_ids=True)
    return base.load_all_responses(pth)


def _build_cell_data(target_dataset: str) -> Tuple[CellData, int, int]:
    """Build CellData restricted to agents that decompose into (model, scaffold).

    Every method in this experiment (empirical, combined_calibrated, oracle)
    is evaluated on the same parseable-agent set so the comparison is fair.
    Unparseable agents (e.g., swebench_verified proprietary stacks like
    `factory_code_droid`) are excluded up-front rather than papered over with
    a fallback in the predictor.

    Returns (data, n_total_agents, n_parseable_agents).
    """
    rows = _load_response_iter(target_dataset)
    responses: Dict[str, Dict[str, Any]] = {}
    for sid, resp in rows:
        responses[str(sid)] = dict(resp)
    if not responses:
        raise RuntimeError(f"No responses loaded for {target_dataset}")

    all_agents = set(responses.keys())
    parseable = parseable_agents_for(target_dataset) & all_agents
    if not parseable:
        raise RuntimeError(
            f"No parseable agents found for {target_dataset}; cannot run the "
            "model+scaffold IRT comparison."
        )

    return (
        CellData(agent_ids=sorted(parseable), responses=responses),
        len(all_agents),
        len(parseable),
    )


def _load_task_universe(items_path: Path) -> List[str]:
    """All task IDs from the canonical full-IRT items.csv (the target benchmark's task set)."""
    full_items = pd.read_csv(items_path, index_col=0)
    return list(full_items.index)


# --- Method evaluation ------------------------------------------------------

def _evaluate_methods(
    methods: List[str],
    data: CellData,
    target_dataset: str,
    sweep_cfg: SubsetExtrapolationConfig,
    seed: int,
    cache_dir: Path,
    observed: List[str],
    heldout: List[str],
    abilities_path: Path,
    items_path: Path,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    multibench_irt = None  # lazily trained, shared between methods that need it

    if "empirical" in methods:
        try:
            mae, preds = evaluate_subset_extrapolation(None, data, observed, heldout)
            out["empirical"] = {"mae": mae, "n_agents": len(preds)}
        except Exception as e:
            out["empirical"] = {"error": f"{type(e).__name__}: {e}"}

    if "combined_calibrated" in methods:
        try:
            from experiment_subset_extrapolation.multibench_trainer import train_fold
            from experiment_subset_extrapolation.heldout_predictor import (
                HeldoutDifficultyPredictor,
            )

            multibench_irt = train_fold(
                target_dataset=target_dataset,
                observed_target_tasks=observed,
                seed=seed,
                cache_dir=cache_dir,
                epochs=sweep_cfg.irt_epochs,
                lr=sweep_cfg.irt_lr,
                device=sweep_cfg.irt_device,
            )
            heldout_pred = HeldoutDifficultyPredictor(target_dataset=target_dataset, seed=seed)
            heldout_pred.fit(multibench_irt)
            predictor = MultiBenchCellPredictor(
                target_dataset=target_dataset,
                target_bench=DATASET_TO_BENCH[target_dataset],
                irt=multibench_irt,
                heldout_predictor=heldout_pred,
            )
            mae, preds = evaluate_subset_extrapolation(
                predictor, data, observed, heldout, calibrate=True
            )
            out["combined_calibrated"] = {
                "mae": mae,
                "n_agents": len(preds),
                "calibrated": True,
            }
        except Exception as e:
            out["combined_calibrated"] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    if "oracle" in methods:
        try:
            predictor = OracleCellPredictor(abilities_path=abilities_path, items_path=items_path)
            mae, preds = evaluate_subset_extrapolation(
                predictor, data, observed, heldout, calibrate=False
            )
            out["oracle"] = {"mae": mae, "n_agents": len(preds)}
        except Exception as e:
            out["oracle"] = {"error": f"{type(e).__name__}: {e}"}

    return out


# --- Per-cell runner --------------------------------------------------------

def run_one_cell(
    dataset: str,
    sweep_cfg: SubsetExtrapolationConfig,
    count: int,
    seed: int,
    root: Path,
) -> Dict[str, Any]:
    """Run a single (dataset, count, seed) configuration end-to-end."""
    t0 = time.time()
    base_cfg = sweep_cfg.base_config(dataset)
    abilities_path = root / base_cfg.abilities_path
    items_path = root / base_cfg.items_path

    all_task_ids = _load_task_universe(items_path)
    observed, heldout = sample_subset_by_count(all_task_ids, int(count), int(seed))

    methods = list(sweep_cfg.methods)
    out: Dict[str, Any] = {
        "dataset": dataset,
        "count": int(count),
        "seed": int(seed),
        "n_total_tasks": len(all_task_ids),
        "n_observed": len(observed),
        "n_heldout": len(heldout),
    }

    try:
        data, n_total_agents, n_parseable_agents = _build_cell_data(dataset)
        out["n_total_agents"] = n_total_agents
        out["n_parseable_agents"] = n_parseable_agents
    except Exception as e:
        out["setup_error"] = f"{type(e).__name__}: {e}"
        out["methods"] = {m: {"error": "cell data load failed"} for m in methods}
        out["elapsed_sec"] = round(time.time() - t0, 2)
        return out

    cache_dir = sweep_cfg.cache_dir_for(dataset, int(count), int(seed))
    out["methods"] = _evaluate_methods(
        methods=methods,
        data=data,
        target_dataset=dataset,
        sweep_cfg=sweep_cfg,
        seed=int(seed),
        cache_dir=cache_dir,
        observed=observed,
        heldout=heldout,
        abilities_path=abilities_path,
        items_path=items_path,
    )

    cc = out["methods"].get("combined_calibrated", {})
    if "mae" in cc:
        out["irt_status"] = "trained"
    elif "error" in cc:
        out["irt_status"] = "failed"

    out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


# --- Sweep aggregation ------------------------------------------------------

def _summarize_method_seeds(
    results_for_cell: List[Dict[str, Any]],
    method: str,
    target_n_seeds: int,
) -> Dict[str, Any]:
    """Pick first `target_n_seeds` successful runs for `method` and compute mean/std MAE."""
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
    """Group raw per-cell results by (dataset, count) and summarize per method."""
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in raw_cell_results:
        by_cell.setdefault((r["dataset"], int(r["count"])), []).append(r)

    out_results: Dict[str, Any] = {}
    for dataset in sweep_cfg.datasets:
        out_results[dataset] = {}
        for count in sweep_cfg.counts_for(dataset):
            cell = by_cell.get((dataset, int(count)), [])
            method_summaries: Dict[str, Any] = {}
            for method in sweep_cfg.methods:
                method_summaries[method] = _summarize_method_seeds(
                    cell, method, sweep_cfg.target_n_seeds
                )
            n_irt_trained = sum(1 for r in cell if r.get("irt_status") == "trained")
            n_irt_failed = sum(1 for r in cell if r.get("irt_status") == "failed")
            out_results[dataset][f"{int(count):04d}"] = {
                "count": int(count),
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
