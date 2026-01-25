#!/usr/bin/env python3
"""Threshold sweep analysis for frontier task difficulty prediction.

Runs experiment_b evaluation at multiple pre_threshold values (0% to 30%)
and generates plots showing Oracle vs Baseline IRT vs Feature-IRT performance.

This script trains models ONCE per dataset then evaluates across thresholds.
Datasets are processed in parallel using multiprocessing.

Usage:
    python -m experiment_b.threshold_sweep
    python -m experiment_b.threshold_sweep --datasets swebench terminalbench
    python -m experiment_b.threshold_sweep --output_dir chris_output/threshold_sweep
"""

import argparse
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment_b import get_dataset_config, list_datasets
from experiment_b.shared import (
    load_and_prepare_data,
    compute_mean_per_agent_auc,
    filter_agents_with_frontier_variance,
    build_feature_sources,
    FeatureIRTPredictor,
    compute_first_capable_dates,
    compute_ground_truth_days,
    DateForecastModel,
    compute_date_forecast_metrics,
    plot_threshold_sweep_auc,
    plot_threshold_sweep_mae,
    plot_ability_vs_date,
    plot_predicted_vs_oracle_scatter,
)
from experiment_b.shared.data_preparation import (
    _train_baseline_irt_on_agents,
)


# Default thresholds to sweep
DEFAULT_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Default hyperparameters for Feature-IRT (from grid search across thresholds)
# These values optimize for good combined AUC@0 and date forecasting MAE
DEFAULT_L2_WEIGHT = 0.001
DEFAULT_L2_RESIDUAL = 10.0

# Datasets with sufficient agent date diversity for meaningful date forecasting
# Other datasets have too few frontier points for reliable ability-over-time regression
DATE_FORECAST_DATASETS = ["swebench"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run threshold sweep analysis across frontier definitions"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list_datasets(),
        choices=list_datasets(),
        help="Datasets to run sweep on (default: all)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Pre-frontier thresholds to sweep (default: 0.0 0.05 0.10 0.15 0.20 0.25 0.30)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/threshold_sweep"),
        help="Directory to save results (default: chris_output/threshold_sweep)",
    )
    parser.add_argument(
        "--post_frontier_oracle",
        action="store_true",
        help="Train Oracle IRT on post-frontier agents only (instead of all agents)",
    )
    parser.add_argument(
        "--l2_weight",
        type=float,
        default=DEFAULT_L2_WEIGHT,
        help=f"L2 regularization for Feature-IRT weights (default: {DEFAULT_L2_WEIGHT})",
    )
    parser.add_argument(
        "--l2_residual",
        type=float,
        default=DEFAULT_L2_RESIDUAL,
        help=f"L2 regularization for Feature-IRT residuals (default: {DEFAULT_L2_RESIDUAL})",
    )
    parser.add_argument(
        "--date_forecast_all",
        action="store_true",
        help="Enable date forecasting for all datasets (default: only swebench)",
    )
    return parser.parse_args()


def _compute_pass_rates_from_dict(
    responses: Dict[str, Dict[str, Any]],
    task_ids: List[str],
    agents: List[str],
) -> Dict[str, float]:
    """Compute pass rates from in-memory responses dict.

    Args:
        responses: Response matrix {agent: {task: response}}
        task_ids: List of task IDs to compute rates for
        agents: List of agent IDs to consider

    Returns:
        Dict mapping task_id -> pass rate (0.0 to 1.0)
    """
    pass_rates = {}
    for task_id in task_ids:
        successes = 0
        total = 0
        for agent_id in agents:
            resp = responses.get(agent_id, {}).get(task_id)
            if resp is not None:
                # Handle both binary and binomial data
                if isinstance(resp, dict) and "successes" in resp:
                    successes += 1 if resp["successes"] > 0 else 0
                else:
                    successes += 1 if resp > 0 else 0
                total += 1
        pass_rates[task_id] = successes / total if total > 0 else 0.0
    return pass_rates


def identify_frontier_tasks_for_threshold(
    responses: Dict[str, Dict[str, Any]],
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
    all_task_ids: List[str],
    pre_threshold: float,
) -> List[str]:
    """Identify frontier tasks based on pre-frontier pass rate threshold.

    A task is a frontier task if:
    - Pre-frontier pass rate <= pre_threshold
    - Post-frontier pass rate > 0 (at least one success)

    Args:
        responses: Response matrix {agent: {task: response}}
        pre_frontier_agents: List of pre-frontier agent IDs
        post_frontier_agents: List of post-frontier agent IDs
        all_task_ids: List of all task IDs
        pre_threshold: Maximum pre-frontier pass rate for frontier tasks

    Returns:
        List of frontier task IDs
    """
    pre_pass_rates = _compute_pass_rates_from_dict(responses, all_task_ids, pre_frontier_agents)
    post_pass_rates = _compute_pass_rates_from_dict(responses, all_task_ids, post_frontier_agents)

    frontier_tasks = []
    for task_id in all_task_ids:
        pre_rate = pre_pass_rates.get(task_id, 0.0)
        post_rate = post_pass_rates.get(task_id, 0.0)

        if pre_rate <= pre_threshold and post_rate > 0:
            frontier_tasks.append(task_id)

    return frontier_tasks


def run_single_dataset(
    dataset_name: str,
    thresholds: List[float],
    output_dir: Path,
    post_frontier_oracle: bool,
    l2_weight: float,
    l2_residual: float,
    enable_date_forecast: bool,
) -> None:
    """Run full threshold sweep for one dataset.

    This function is designed to run in a separate process.
    It trains models ONCE then evaluates across all thresholds.

    Args:
        dataset_name: Name of the dataset to evaluate
        thresholds: List of pre_threshold values to sweep
        output_dir: Directory to save results
        post_frontier_oracle: If True, train Oracle IRT on post-frontier agents only
        l2_weight: L2 regularization for Feature-IRT weights
        l2_residual: L2 regularization for Feature-IRT residuals
        enable_date_forecast: If True, compute date forecasting metrics and plots
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    config = get_dataset_config(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === TRAIN ONCE (threshold-independent) ===
    print("\n--- Training Phase (once per dataset) ---")

    # Create a minimal args object for load_and_prepare_data
    # Use threshold=0 just to get the data structures
    class Args:
        responses_path = None
        baseline_irt_path = None
        oracle_irt_path = None
        oracle_abilities_path = None
        embeddings_path = None
        llm_judge_path = None
        trajectory_features_path = None
        cutoff_date = None
        pre_threshold = 0.0  # Start with 0 to get all data structures
        post_threshold = None
        filter_bottom_percentile = 0.0
        min_oracle_ability = None
        frontier_definitions = ["pre_only"]

    args = Args()

    # Load data with threshold=0 to get all structures
    data = load_and_prepare_data(args, config)

    # Load Oracle IRT
    oracle_items = data.oracle_items
    oracle_abilities = data.oracle_abilities
    print(f"  Oracle IRT: {len(oracle_items)} tasks, {len(oracle_abilities)} agents")

    # Train post-frontier Oracle if requested
    post_frontier_oracle_beta: Optional[Dict[str, float]] = None
    if post_frontier_oracle:
        print(f"\nTraining Oracle IRT on {len(data.post_frontier_agents)} post-frontier agents...")
        cache_dir = output_dir / f"post_frontier_oracle_{dataset_name}"
        post_frontier_oracle_beta = _train_baseline_irt_on_agents(
            responses_path=config.responses_path,
            agent_subset=data.post_frontier_agents,
            output_dir=cache_dir,
            epochs=2000,
        )
        print(f"  Post-frontier Oracle trained: {len(post_frontier_oracle_beta)} tasks")

    # Baseline IRT (trained on pre-frontier agents - cached)
    baseline_items = data.baseline_items
    baseline_abilities = data.baseline_abilities
    print(f"  Baseline IRT: {len(baseline_items)} tasks, {len(baseline_abilities)} agents")

    # Train Feature-IRT with FIXED hyperparameters
    print(f"\nTraining Feature-IRT (l2_weight={l2_weight}, l2_residual={l2_residual})...")
    feature_sources = build_feature_sources(config)

    # Find embedding source
    embedding_source = None
    for name, source in feature_sources:
        if name == "Embedding":
            embedding_source = source
            break

    feature_irt_preds: Optional[Dict[str, float]] = None
    feature_irt_abilities: Optional[Dict[str, float]] = None

    if embedding_source is not None and baseline_abilities is not None:
        try:
            feature_irt_predictor = FeatureIRTPredictor(
                source=embedding_source,
                use_residuals=True,
                init_from_baseline=True,
                l2_weight=l2_weight,
                l2_residual=l2_residual,
                verbose=True,
            )
            feature_irt_predictor.fit(
                task_ids=data.train_task_ids,
                ground_truth_b=data.baseline_ground_truth_b,
                responses=data.train_responses,
                baseline_abilities=baseline_abilities["theta"].values,
                baseline_agent_ids=list(baseline_abilities.index),
            )
            feature_irt_preds = feature_irt_predictor.predict(config.all_task_ids)
            feature_irt_abilities = feature_irt_predictor.learned_abilities
            print(f"  Feature-IRT: {len(feature_irt_preds)} task predictions")
        except Exception as e:
            print(f"  Feature-IRT training failed: {e}")
    else:
        print("  Skipping Feature-IRT (no embeddings or baseline abilities)")

    # Collect all predictions and abilities
    all_predictions: Dict[str, Dict[str, float]] = {}
    all_abilities: Dict[str, Dict[str, float]] = {}

    # Oracle
    if post_frontier_oracle_beta is not None:
        all_predictions["Oracle (post-frontier only)"] = post_frontier_oracle_beta
        # For post-frontier oracle, we don't have abilities easily accessible
        # Skip date forecasting for this variant
    else:
        all_predictions["Oracle (upper bound)"] = oracle_items["b"].to_dict()
        all_abilities["Oracle (upper bound)"] = oracle_abilities["theta"].to_dict()

    # Baseline IRT
    all_predictions["Baseline IRT (pre-frontier only)"] = baseline_items["b"].to_dict()
    if baseline_abilities is not None:
        all_abilities["Baseline IRT (pre-frontier only)"] = baseline_abilities["theta"].to_dict()

    # Feature-IRT
    if feature_irt_preds is not None:
        all_predictions["Baseline-Init Feature-IRT (Embedding)"] = feature_irt_preds
        if feature_irt_abilities is not None:
            all_abilities["Baseline-Init Feature-IRT (Embedding)"] = feature_irt_abilities

    # Generate scatter plot of predicted vs oracle for zero_pre frontier tasks
    if feature_irt_preds is not None:
        zero_pre_frontier = identify_frontier_tasks_for_threshold(
            responses=config.responses,
            pre_frontier_agents=data.pre_frontier_agents,
            post_frontier_agents=data.post_frontier_agents,
            all_task_ids=config.all_task_ids,
            pre_threshold=0.0,  # zero_pre definition
        )
        if zero_pre_frontier:
            scatter_path = output_dir / f"predicted_vs_oracle_{dataset_name}.png"
            plot_predicted_vs_oracle_scatter(
                predicted_beta=feature_irt_preds,
                oracle_beta=oracle_items["b"].to_dict(),
                frontier_task_ids=zero_pre_frontier,
                dataset_name=config.name,
                method_name="Baseline-Init Feature-IRT (Embedding)",
                output_path=scatter_path,
            )
        else:
            print(f"  No zero_pre frontier tasks for {dataset_name} - skipping scatter plot")

    # Date forecasting (optional, only for datasets with sufficient agent diversity)
    date_models: Dict[str, DateForecastModel] = {}
    fit_results: Dict[str, Dict[str, Any]] = {}
    ground_truth_days: Dict[str, float] = {}

    if enable_date_forecast:
        # Compute ground truth dates (using Oracle IRT)
        print("\nComputing ground truth dates...")
        try:
            gt_result = compute_first_capable_dates(
                oracle_items=oracle_items,
                oracle_abilities=oracle_abilities,
                agent_dates=config.agent_dates,
            )
            ground_truth_days = compute_ground_truth_days(
                task_ids=config.all_task_ids,
                first_capable_dates=gt_result.first_capable_dates,
                reference_date=gt_result.earliest_agent_date,
            )
            print(f"  Ground truth: {len(ground_truth_days)} tasks with capable agents")
            print(f"  Tasks without capable agent: {len(gt_result.tasks_without_capable_agent)}")
        except Exception as e:
            print(f"  Ground truth computation failed: {e}")
            ground_truth_days = {}

        # Fit date models (one per method with abilities)
        print("\nFitting date forecast models...")
        for method_name, abilities in all_abilities.items():
            try:
                model = DateForecastModel()
                fit_stats = model.fit(abilities, config.agent_dates)
                date_models[method_name] = model
                fit_results[method_name] = fit_stats
                print(f"  {method_name}: R²={fit_stats['r_squared']:.3f}, "
                      f"slope={fit_stats['slope']:.4f} theta/day")
            except Exception as e:
                print(f"  {method_name}: Failed to fit ({e})")
                fit_results[method_name] = {}  # Empty dict signals fit failure

        # Generate ability-vs-date plot
        ability_plot_path = output_dir / f"ability_vs_date_{dataset_name}.png"
        try:
            plot_ability_vs_date(
                method_abilities=all_abilities,
                agent_dates=config.agent_dates,
                fit_results=fit_results,
                dataset_name=config.name,
                output_path=ability_plot_path,
            )
        except Exception as e:
            print(f"  Failed to generate ability-vs-date plot: {e}")
    else:
        print("\nSkipping date forecasting (not enabled for this dataset)")

    # === EVALUATE PER THRESHOLD ===
    print("\n--- Evaluation Phase (per threshold) ---")
    results: List[Dict[str, Any]] = []
    thresholds = sorted(thresholds)
    fixed_eval_agents: Optional[List[str]] = None

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold*100:.0f}% ---")

        # Identify frontier tasks for this threshold
        frontier_tasks = identify_frontier_tasks_for_threshold(
            responses=config.responses,
            pre_frontier_agents=data.pre_frontier_agents,
            post_frontier_agents=data.post_frontier_agents,
            all_task_ids=config.all_task_ids,
            pre_threshold=threshold,
        )

        if len(frontier_tasks) == 0:
            print(f"  No frontier tasks at threshold {threshold*100:.0f}%, skipping")
            continue

        # Filter eval agents (fix at first threshold for consistent comparison)
        if fixed_eval_agents is None:
            fixed_eval_agents = filter_agents_with_frontier_variance(
                responses=config.responses,
                frontier_task_ids=frontier_tasks,
                candidate_agents=data.post_frontier_agents,
            )
            print(f"  Fixed eval agent set: {len(fixed_eval_agents)} agents "
                  f"(from {threshold*100:.0f}% threshold)")
        else:
            # Filter to agents with variance on current frontier tasks
            eval_agents = filter_agents_with_frontier_variance(
                responses=config.responses,
                frontier_task_ids=frontier_tasks,
                candidate_agents=fixed_eval_agents,
            )
            if len(eval_agents) == 0:
                print(f"  No eval agents with variance at {threshold*100:.0f}%, skipping")
                continue
            eval_agents_for_threshold = eval_agents

        # Use fixed_eval_agents for first threshold, otherwise use filtered
        eval_agents_for_threshold = fixed_eval_agents if threshold == thresholds[0] else eval_agents

        print(f"  Frontier tasks: {len(frontier_tasks)}")
        print(f"  Eval agents: {len(eval_agents_for_threshold)}")

        # Compute metrics for each method
        for method_name, predictions in all_predictions.items():
            # AUC metrics
            auc_metrics = compute_mean_per_agent_auc(
                predicted_beta=predictions,
                responses=config.responses,
                frontier_task_ids=frontier_tasks,
                eval_agents=eval_agents_for_threshold,
            )

            # Date MAE metrics (if model available)
            mae_days = float("nan")
            r_squared_fit = float("nan")
            n_date_tasks = 0

            if method_name in date_models and ground_truth_days:
                try:
                    date_model = date_models[method_name]
                    date_predictions = date_model.predict(predictions, frontier_tasks)
                    date_metrics = compute_date_forecast_metrics(
                        predicted=date_predictions,
                        ground_truth_days=ground_truth_days,
                        task_ids=frontier_tasks,
                    )
                    mae_days = date_metrics["mae_days"]
                    r_squared_fit = date_model.r_squared
                    n_date_tasks = date_metrics["n_tasks"]
                except Exception as e:
                    print(f"    Date forecast failed for {method_name}: {e}")

            results.append({
                "threshold": threshold,
                "method": method_name,
                "mean_auc": auc_metrics["mean_auc"],
                "sem": auc_metrics["sem_auc"],
                "n_agents": auc_metrics["n_agents"],
                "n_frontier_tasks": len(frontier_tasks),
                "mae_days": mae_days,
                "r_squared_fit": r_squared_fit,
                "n_date_tasks": n_date_tasks,
            })

            print(f"  {method_name}: AUC={auc_metrics['mean_auc']:.4f} ± {auc_metrics['sem_auc']:.4f}"
                  + (f", MAE={mae_days:.1f} days" if not np.isnan(mae_days) else ""))

    # Save results
    if not results:
        print(f"\nNo results for {dataset_name}")
        return

    df = pd.DataFrame(results)

    # Save CSV
    csv_path = output_dir / f"threshold_sweep_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    # Generate plots
    auc_plot_path = output_dir / f"threshold_sweep_{dataset_name}.png"
    plot_threshold_sweep_auc(df, dataset_name, auc_plot_path)

    if enable_date_forecast:
        mae_plot_path = output_dir / f"date_forecast_{dataset_name}.png"
        plot_threshold_sweep_mae(df, dataset_name, mae_plot_path)

    print(f"\nDone with {dataset_name}!")


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Threshold Sweep Analysis")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Thresholds: {[f'{t*100:.0f}%' for t in args.thresholds]}")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature-IRT hyperparameters: l2_weight={args.l2_weight}, l2_residual={args.l2_residual}")
    if args.post_frontier_oracle:
        print("Using POST-FRONTIER Oracle mode")
    if args.date_forecast_all:
        print("Date forecasting enabled for ALL datasets")
    else:
        print(f"Date forecasting enabled for: {DATE_FORECAST_DATASETS}")

    # Launch one process per dataset
    processes = []
    for dataset_name in args.datasets:
        # Determine if date forecasting is enabled for this dataset
        enable_date_forecast = args.date_forecast_all or dataset_name in DATE_FORECAST_DATASETS

        p = Process(
            target=run_single_dataset,
            args=(
                dataset_name,
                args.thresholds,
                args.output_dir,
                args.post_frontier_oracle,
                args.l2_weight,
                args.l2_residual,
                enable_date_forecast,
            ),
        )
        processes.append(p)
        p.start()
        print(f"Started process for {dataset_name}")

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\n" + "="*60)
    print("All datasets complete!")
    print("="*60)


if __name__ == "__main__":
    main()
