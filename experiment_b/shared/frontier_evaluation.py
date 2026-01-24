"""Frontier evaluation pipeline for Experiment B.

This module handles:
- Date forecasting setup and evaluation
- Evaluation across multiple frontier definitions
- Consolidation of results for reporting
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiment_b.shared.data_preparation import ExperimentData
from experiment_b.shared.date_forecasting import (
    compute_first_capable_dates,
    compute_ground_truth_days,
    split_tasks_by_first_capable_date,
    DateForecastModel,
    compute_date_forecast_metrics,
    compute_dates_from_predicted_difficulties,
    compute_frontier_ability_intervals,
    parse_date,
)
import pandas as pd
from experiment_b.shared.evaluation import (
    compute_method_metrics,
    compute_scale_offset,
    shift_to_oracle_scale,
    compute_mean_per_agent_auc,
    filter_agents_with_frontier_variance,
)


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class DateForecastingData:
    """Data for date forecasting evaluation."""

    date_models: Dict[str, Tuple[DateForecastModel, Dict[str, Any]]]  # method -> (model, fit_stats)
    ground_truth_days: Dict[str, float]  # task_id -> days since earliest agent
    first_capable_dates: Dict[str, datetime]  # task_id -> datetime
    earliest_agent_date: str  # YYYY-MM-DD
    latest_agent_date: str  # YYYY-MM-DD
    gt_date_min: str  # YYYY-MM-DD (min ground truth date for eval set)
    gt_date_max: str  # YYYY-MM-DD (max ground truth date for eval set)
    tasks_without_capable: int  # Count of tasks without any capable agent
    oracle_abilities: pd.DataFrame  # Oracle abilities for direct date lookup


# =============================================================================
# Date Forecasting Setup
# =============================================================================


def setup_date_forecasting(
    data: ExperimentData,
    raw_predictions: Dict[str, Dict[str, float]],
    method_abilities: Dict[str, Dict[str, float]],
) -> Optional[DateForecastingData]:
    """Prepare date forecasting models and ground truth.

    Fits a separate date model for each method that has abilities.

    Args:
        data: Experiment data
        raw_predictions: All collected predictions
        method_abilities: Abilities for methods that have their own IRT

    Returns:
        DateForecastingData or None if not enough data
    """
    print("\nRunning date forecasting evaluation...")

    gt_result = compute_first_capable_dates(
        data.oracle_items, data.oracle_abilities, data.config.agent_dates
    )
    first_capable_dates = gt_result.first_capable_dates
    tasks_without_capable = gt_result.tasks_without_capable_agent
    earliest_agent_date = gt_result.earliest_agent_date
    latest_agent_date = gt_result.latest_agent_date

    cutoff_datetime = parse_date(data.cutoff_date)
    _, post_cutoff_tasks = split_tasks_by_first_capable_date(
        first_capable_dates, cutoff_datetime
    )

    print(f"  Tasks with ground truth (first capable agent exists): {len(first_capable_dates)}")
    print(f"  Post-cutoff tasks (for evaluation): {len(post_cutoff_tasks)}")
    print(f"  Tasks without any capable agent: {len(tasks_without_capable)}")

    # Compute frontier ability interval statistics
    intervals = compute_frontier_ability_intervals(
        data.oracle_abilities, data.config.agent_dates
    )
    print(f"  Frontier model intervals: mean={intervals['mean_days']:.1f}, median={intervals['median_days']:.1f}, "
          f"range={intervals['min_days']:.0f}-{intervals['max_days']:.0f} days ({intervals['n_frontier_jumps']} jumps)")

    if len(post_cutoff_tasks) < 3:
        print("  Not enough post-cutoff tasks for date forecasting")
        return None

    ground_truth_days = compute_ground_truth_days(
        data.train_task_ids, first_capable_dates, earliest_agent_date
    )

    post_cutoff_gt_dates = [first_capable_dates[t] for t in post_cutoff_tasks]
    gt_date_min = min(post_cutoff_gt_dates).strftime("%Y-%m-%d")
    gt_date_max = max(post_cutoff_gt_dates).strftime("%Y-%m-%d")

    # Fit date models for each method with abilities
    date_models: Dict[str, Tuple[DateForecastModel, Dict[str, Any]]] = {}
    for method_name, abilities in method_abilities.items():
        try:
            date_model = DateForecastModel()
            fit_stats = date_model.fit(abilities, data.config.agent_dates)
            date_models[method_name] = (date_model, fit_stats)
        except Exception as e:
            print(f"  Warning: Could not fit date model for {method_name}: {e}")

    return DateForecastingData(
        date_models=date_models,
        ground_truth_days=ground_truth_days,
        first_capable_dates=first_capable_dates,
        earliest_agent_date=earliest_agent_date,
        latest_agent_date=latest_agent_date,
        gt_date_min=gt_date_min,
        gt_date_max=gt_date_max,
        tasks_without_capable=len(tasks_without_capable),
        oracle_abilities=data.oracle_abilities,
    )


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================


def evaluate_all_frontier_definitions(
    frontier_definitions: List[str],
    data: ExperimentData,
    raw_predictions: Dict[str, Dict[str, float]],
    date_info: Optional[DateForecastingData],
    alignment_method: str = "affine",
    verbose: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run evaluation for all frontier definitions and methods.

    Evaluates ALL methods uniformly. For reporting, the caller can consolidate
    SAD-IRT runs into a single "best" entry.

    Args:
        frontier_definitions: List of frontier definitions to evaluate
        data: Experiment data
        raw_predictions: All collected predictions
        date_info: Date forecasting data (optional)
        alignment_method: Scale alignment method
        verbose: Print verbose output

    Returns:
        Dict mapping frontier_def -> results dict (includes all SAD-IRT runs)
    """
    # Import here to avoid circular dependency
    from experiment_b.shared.output_formatting import print_comparison_table

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for frontier_def in frontier_definitions:
        frontier_task_ids = data.frontier_tasks_by_def[frontier_def]
        print(f"\n{'='*90}")
        print(f"Evaluating metrics for frontier definition: {frontier_def}")
        print(f"{'='*90}")

        results: Dict[str, Dict[str, Any]] = {}

        # Get filtered eval agents for this frontier definition (if filtering applied)
        eval_agents = data.eval_agents_by_def.get(frontier_def, data.post_frontier_agents)

        # Filter to agents with frontier variance (at least one success on frontier tasks)
        eval_agents_with_variance = filter_agents_with_frontier_variance(
            responses=data.config.responses,
            frontier_task_ids=frontier_task_ids,
            candidate_agents=eval_agents,
        )
        print(f"  Eval agents: {len(eval_agents)} total, {len(eval_agents_with_variance)} with frontier variance")

        # Compute metrics for each method uniformly
        for method_name, pred_beta in raw_predictions.items():
            try:
                # Old pooled AUC (requires scale alignment)
                metrics = compute_method_metrics(
                    predicted_beta=pred_beta,
                    oracle_items=data.oracle_items,
                    oracle_abilities=data.oracle_abilities,
                    responses=data.config.responses,
                    frontier_task_ids=frontier_task_ids,
                    anchor_task_ids=data.anchor_task_ids,
                    eval_agents=eval_agents,
                    alignment_method=alignment_method,
                )

                # New scale-free Mean Per-Agent AUC
                per_agent_metrics = compute_mean_per_agent_auc(
                    predicted_beta=pred_beta,
                    responses=data.config.responses,
                    frontier_task_ids=frontier_task_ids,
                    eval_agents=eval_agents_with_variance,
                )
                metrics["mean_per_agent_auc"] = per_agent_metrics["mean_auc"]
                metrics["mean_per_agent_auc_std"] = per_agent_metrics["std_auc"]
                metrics["mean_per_agent_auc_sem"] = per_agent_metrics["sem_auc"]
                metrics["mean_per_agent_auc_n_agents"] = per_agent_metrics["n_agents"]

                results[method_name] = metrics
            except Exception as e:
                print(f"  Error computing metrics for {method_name}: {e}")
                results[method_name] = {
                    "auc": None,
                    "mean_per_agent_auc": None,
                    "mean_per_agent_auc_std": None,
                    "mean_per_agent_auc_sem": None,
                    "mean_per_agent_auc_n_agents": 0,
                    "num_frontier_tasks": 0,
                }

        all_results[frontier_def] = results

        # Compute date forecasting for this frontier definition
        date_results_for_def: Optional[Dict[str, Dict[str, Any]]] = None
        oracle_date_results_for_def: Optional[Dict[str, Dict[str, Any]]] = None
        if date_info is not None:
            # Only evaluate on frontier tasks that have a capable agent (ground truth exists)
            eval_tasks = [t for t in frontier_task_ids if t in date_info.first_capable_dates]

            if len(eval_tasks) >= 3:
                date_results_for_def = {}
                oracle_date_results_for_def = {}

                # Get oracle beta for alignment
                oracle_beta = data.oracle_items["b"].to_dict()

                for method_name, pred_beta in raw_predictions.items():
                    # Align predictions to oracle scale for Oracle MAE computation
                    # This ensures predicted β values are on the same scale as Oracle θ
                    alignment_params = compute_scale_offset(
                        pred_beta, oracle_beta, data.anchor_task_ids, method="affine"
                    )
                    aligned_beta = shift_to_oracle_scale(pred_beta, alignment_params)

                    # Compute Oracle MAE for all methods (direct lookup using Oracle abilities)
                    try:
                        oracle_predictions = compute_dates_from_predicted_difficulties(
                            predicted_beta=aligned_beta,
                            oracle_abilities=date_info.oracle_abilities,
                            agent_dates=data.config.agent_dates,
                            task_ids=eval_tasks,
                        )
                        oracle_metrics = compute_date_forecast_metrics(
                            oracle_predictions, date_info.ground_truth_days, eval_tasks
                        )
                        oracle_date_results_for_def[method_name] = oracle_metrics
                    except Exception:
                        oracle_date_results_for_def[method_name] = {
                            "mae_days": float("nan"),
                            "n_tasks": 0,
                        }

                    # Compute regular MAE (regression-based) only for methods with date models
                    if method_name not in date_info.date_models:
                        continue

                    date_model, fit_stats = date_info.date_models[method_name]
                    try:
                        predictions = date_model.predict(pred_beta, eval_tasks)
                        metrics = compute_date_forecast_metrics(
                            predictions, date_info.ground_truth_days, eval_tasks
                        )
                        date_results_for_def[method_name] = {
                            **metrics,
                            "r_squared_fit": fit_stats["r_squared"],
                        }
                    except Exception:
                        date_results_for_def[method_name] = {
                            "mae_days": float("nan"),
                            "pearson_r": float("nan"),
                            "r_squared_fit": float("nan"),
                            "n_tasks": 0,
                        }

        # Consolidate SAD-IRT runs and print comparison table
        # Find best SAD-IRT run by AUC for this frontier definition
        consolidated_results = {}
        best_sad_irt_auc = -1.0
        best_sad_irt_name: Optional[str] = None

        for method_name, metrics in results.items():
            if method_name.startswith("SAD-IRT ("):
                auc = metrics.get('auc') or 0
                if auc > best_sad_irt_auc:
                    best_sad_irt_auc = auc
                    best_sad_irt_name = method_name
            else:
                consolidated_results[method_name] = metrics

        # Add best SAD-IRT as "SAD-IRT (best)"
        if best_sad_irt_name:
            consolidated_results["SAD-IRT (best)"] = results[best_sad_irt_name]
            print(f"  Best SAD-IRT: {best_sad_irt_name} (AUC: {best_sad_irt_auc:.4f})")

        # Consolidate date results similarly
        consolidated_date_results: Optional[Dict[str, Dict[str, Any]]] = None
        if date_results_for_def:
            consolidated_date_results = {}
            for method_name, date_metrics in date_results_for_def.items():
                if method_name.startswith("SAD-IRT ("):
                    if method_name == best_sad_irt_name:
                        consolidated_date_results["SAD-IRT (best)"] = date_metrics
                else:
                    consolidated_date_results[method_name] = date_metrics

        # Consolidate Oracle date results similarly
        consolidated_oracle_date_results: Optional[Dict[str, Dict[str, Any]]] = None
        if oracle_date_results_for_def:
            consolidated_oracle_date_results = {}
            for method_name, date_metrics in oracle_date_results_for_def.items():
                if method_name.startswith("SAD-IRT ("):
                    if method_name == best_sad_irt_name:
                        consolidated_oracle_date_results["SAD-IRT (best)"] = date_metrics
                else:
                    consolidated_oracle_date_results[method_name] = date_metrics

        print()
        # Get filtering stats for this frontier definition (if any)
        filtering_stats = data.filtering_stats_by_def.get(frontier_def)
        print_comparison_table(
            consolidated_results,
            len(frontier_task_ids),
            len(data.pre_frontier_agents),
            len(eval_agents),  # Use filtered agent count
            anchor_task_count=len(data.anchor_task_ids),
            alignment_method=alignment_method,
            cutoff_date=data.cutoff_date,
            frontier_definition=frontier_def,
            irt_solve_prob=data.config.irt_solve_probability,
            date_results=consolidated_date_results,
            oracle_date_results=consolidated_oracle_date_results,
            last_agent_date=data.config.last_agent_date,
            verbose=verbose,
            dataset_name=data.config.name,
            filtering_stats=filtering_stats,
        )

    return all_results
