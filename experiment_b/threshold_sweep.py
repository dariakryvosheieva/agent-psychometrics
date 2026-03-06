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
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    plot_predicted_vs_actual_dates,
    fit_with_cv_hyperparams,
    L2_GRID,
    SINGLE_SOURCE_GRID,
    make_grouped_source_grid,
)
from experiment_b.shared.data_preparation import (
    _train_baseline_irt_on_agents,
)
from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    GroupedFeatureSource,
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


def make_feature_irt_train_fn(
    source: TaskFeatureSource,
    task_ids: List[str],
    ground_truth_b: np.ndarray,
    baseline_abilities: np.ndarray,
    baseline_agent_ids: List[str],
    use_baseline_init: bool = True,
) -> Callable[[Dict[str, Any], Dict[str, Dict[str, int]]], Tuple[Dict[str, float], Dict[str, float]]]:
    """Create a training function for use with fit_with_cv_hyperparams.

    Args:
        source: Feature source (TaskFeatureSource or GroupedFeatureSource)
        task_ids: Task IDs for training
        ground_truth_b: Baseline IRT difficulties
        baseline_abilities: Baseline IRT abilities
        baseline_agent_ids: Agent IDs corresponding to baseline_abilities
        use_baseline_init: If True, use baseline-init mode

    Returns:
        Train function compatible with fit_with_cv_hyperparams
    """
    def train_fn(
        hyperparams: Dict[str, Any],
        responses: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        # Extract hyperparams
        l2_weight = hyperparams.get("l2_weight", 1.0)
        l2_residual = hyperparams.get("l2_residual", 10.0)
        l2_ability = hyperparams.get("l2_ability", 0.01)

        # For grouped sources, extract per_source_alphas from alpha_{source_name} params
        per_source_alphas = None
        if isinstance(source, GroupedFeatureSource):
            per_source_alphas = {}
            for src in source.sources:
                alpha_key = f"alpha_{src.name}"
                if alpha_key in hyperparams:
                    per_source_alphas[src.name] = hyperparams[alpha_key]
            # With per_source_alphas, set l2_weight to 1.0 (group scaling handles reg)
            l2_weight = 1.0

        predictor = FeatureIRTPredictor(
            source=source,
            use_residuals=use_baseline_init,
            init_from_baseline=use_baseline_init,
            l2_weight=l2_weight,
            l2_residual=l2_residual,
            l2_ability=l2_ability,
            per_source_alphas=per_source_alphas,
            verbose=False,
        )

        predictor.fit(
            task_ids=task_ids,
            ground_truth_b=ground_truth_b,
            responses=responses,
            baseline_abilities=baseline_abilities,
            baseline_agent_ids=baseline_agent_ids,
        )

        return predictor.learned_abilities, {
            task_id: predictor.predict([task_id])[task_id]
            for task_id in task_ids
        }

    return train_fn


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
    parser.add_argument(
        "--use_cv_hyperparams",
        action="store_true",
        help="Use CV-based hyperparameter selection (grid search over L2 params)",
    )
    parser.add_argument(
        "--test_all_feature_configs",
        action="store_true",
        help="Test all feature configurations (Embedding, Trajectory, Embedding+Trajectory)",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Only show Oracle IRT and Baseline IRT (skip Feature-IRT training and date forecasting)",
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
    use_cv_hyperparams: bool = False,
    test_all_feature_configs: bool = False,
    baseline_only: bool = False,
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
        use_cv_hyperparams: If True, use CV-based hyperparameter selection
        test_all_feature_configs: If True, test Embedding, Trajectory, and combined
        baseline_only: If True, only show Oracle IRT and Baseline IRT (skip Feature-IRT)
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Force disable date forecasting in baseline_only mode
    if baseline_only:
        enable_date_forecast = False

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

    # Build feature sources (skip if baseline_only)
    feature_configs: List[Tuple[str, TaskFeatureSource]] = []
    if not baseline_only:
        print(f"\nBuilding feature sources...")
        feature_sources = build_feature_sources(config)
        source_dict = {name: source for name, source in feature_sources}

        # Always test Embedding if available
        if "Embedding" in source_dict:
            feature_configs.append(("Embedding", source_dict["Embedding"]))

        # Test all feature configurations if enabled
        if test_all_feature_configs:
            # Single sources
            if "LLM Judge" in source_dict:
                feature_configs.append(("LLM Judge", source_dict["LLM Judge"]))
            if "Trajectory" in source_dict:
                feature_configs.append(("Trajectory", source_dict["Trajectory"]))

            # Two-source combinations with Embedding
            if "Embedding" in source_dict and "LLM Judge" in source_dict:
                combined_source = GroupedFeatureSource([
                    source_dict["Embedding"],
                    source_dict["LLM Judge"],
                ])
                feature_configs.append(("Embedding + LLM Judge", combined_source))

            if "Embedding" in source_dict and "Trajectory" in source_dict:
                combined_source = GroupedFeatureSource([
                    source_dict["Embedding"],
                    source_dict["Trajectory"],
                ])
                feature_configs.append(("Embedding + Trajectory", combined_source))

            # Three-source combination
            if "Embedding" in source_dict and "LLM Judge" in source_dict and "Trajectory" in source_dict:
                combined_source = GroupedFeatureSource([
                    source_dict["Embedding"],
                    source_dict["LLM Judge"],
                    source_dict["Trajectory"],
                ])
                feature_configs.append(("Embedding + LLM Judge + Trajectory", combined_source))

        print(f"  Feature configs to test: {[name for name, _ in feature_configs]}")
    else:
        print(f"\nSkipping Feature-IRT (baseline_only mode)")

    # Train Feature-IRT for each configuration (skip if baseline_only)
    feature_irt_results: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {}

    if baseline_only:
        pass  # Skip Feature-IRT training entirely
    elif baseline_abilities is None:
        print("  Skipping Feature-IRT (no baseline abilities)")
    else:
        baseline_theta = baseline_abilities["theta"].values
        baseline_agent_ids = list(baseline_abilities.index)

        for config_name, source in feature_configs:
            method_name = f"Baseline-Init Feature-IRT ({config_name})"
            print(f"\nTraining {method_name}...")
            config_start_time = time.time()

            if use_cv_hyperparams:
                # CV-based hyperparameter selection
                print(f"  Using CV-based hyperparameter selection...")

                # Choose hyperparam grid based on source type
                if isinstance(source, GroupedFeatureSource):
                    source_names = [s.name for s in source.sources]
                    hyperparam_grid = make_grouped_source_grid(source_names)
                    print(f"  Grid for grouped sources: {source_names}")
                else:
                    hyperparam_grid = SINGLE_SOURCE_GRID
                    print(f"  Grid for single source")

                try:
                    train_fn = make_feature_irt_train_fn(
                        source=source,
                        task_ids=data.train_task_ids,
                        ground_truth_b=data.baseline_ground_truth_b,
                        baseline_abilities=baseline_theta,
                        baseline_agent_ids=baseline_agent_ids,
                        use_baseline_init=True,
                    )

                    best_params, (abilities, difficulties) = fit_with_cv_hyperparams(
                        train_fn=train_fn,
                        hyperparam_grid=hyperparam_grid,
                        responses=data.train_responses,
                        agent_ids=list(data.train_responses.keys()),
                        task_ids=data.train_task_ids,
                        val_frac=0.2,
                        n_jobs=-1,
                        verbose=True,
                    )
                    feature_irt_results[config_name] = (abilities, difficulties)
                    config_elapsed = time.time() - config_start_time
                    print(f"  Best params: {best_params}")
                    print(f"  Training completed in {config_elapsed:.1f}s")
                except Exception as e:
                    config_elapsed = time.time() - config_start_time
                    print(f"  CV hyperparameter selection failed after {config_elapsed:.1f}s: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Fixed hyperparameters
                print(f"  Using fixed hyperparameters (l2_weight={l2_weight}, l2_residual={l2_residual})...")

                try:
                    # For grouped sources, use default alphas from source
                    per_source_alphas = None
                    if isinstance(source, GroupedFeatureSource):
                        per_source_alphas = {
                            s.name: s.alpha for s in source.sources
                        }

                    predictor = FeatureIRTPredictor(
                        source=source,
                        use_residuals=True,
                        init_from_baseline=True,
                        l2_weight=l2_weight,
                        l2_residual=l2_residual,
                        per_source_alphas=per_source_alphas,
                        verbose=True,
                    )
                    predictor.fit(
                        task_ids=data.train_task_ids,
                        ground_truth_b=data.baseline_ground_truth_b,
                        responses=data.train_responses,
                        baseline_abilities=baseline_theta,
                        baseline_agent_ids=baseline_agent_ids,
                    )
                    preds = predictor.predict(config.all_task_ids)
                    abilities = predictor.learned_abilities
                    feature_irt_results[config_name] = (abilities, preds)
                    config_elapsed = time.time() - config_start_time
                    print(f"  {method_name}: {len(preds)} task predictions")
                    print(f"  Training completed in {config_elapsed:.1f}s")
                except Exception as e:
                    print(f"  Feature-IRT training failed: {e}")
                    config_elapsed = time.time() - config_start_time
                    print(f"  Failed after {config_elapsed:.1f}s")
                    import traceback
                    traceback.print_exc()

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

    # Feature-IRT (all configurations)
    for config_name, (abilities, preds) in feature_irt_results.items():
        method_name = f"Baseline-Init Feature-IRT ({config_name})"
        all_predictions[method_name] = preds
        if abilities is not None:
            all_abilities[method_name] = abilities

    # Generate scatter plot of predicted vs oracle for zero_pre frontier tasks
    # (skip if baseline_only)
    # Default to Embedding if available, otherwise use first available config
    scatter_config_name = "Embedding" if "Embedding" in feature_irt_results else (
        list(feature_irt_results.keys())[0] if feature_irt_results else None
    )

    if not baseline_only and scatter_config_name is not None:
        _, scatter_preds = feature_irt_results[scatter_config_name]

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
                predicted_beta=scatter_preds,
                oracle_beta=oracle_items["b"].to_dict(),
                frontier_task_ids=zero_pre_frontier,
                dataset_name=config.name,
                method_name=f"Baseline-Init Feature-IRT ({scatter_config_name})",
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

                    # Generate predicted vs actual dates scatter plot at threshold=0
                    if (
                        threshold == 0.0
                        and dataset_name == "swebench"
                        and "Feature-IRT" in method_name
                    ):
                        scatter_path = output_dir / f"predicted_vs_actual_dates_{dataset_name}.png"
                        plot_predicted_vs_actual_dates(
                            predicted_dates=date_predictions,
                            ground_truth_days=ground_truth_days,
                            frontier_task_ids=frontier_tasks,
                            dataset_name=config.name,
                            method_name=method_name,
                            output_path=scatter_path,
                        )
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
    if args.use_cv_hyperparams:
        print("Using CV-based hyperparameter selection")
    else:
        print(f"Feature-IRT hyperparameters: l2_weight={args.l2_weight}, l2_residual={args.l2_residual}")
    if args.test_all_feature_configs:
        print("Testing all feature configurations (Embedding, Trajectory, Embedding+Trajectory)")
    if args.post_frontier_oracle:
        print("Using POST-FRONTIER Oracle mode")
    if args.baseline_only:
        print("Baseline-only mode: showing Oracle IRT and Baseline IRT only")
    elif args.date_forecast_all:
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
                args.use_cv_hyperparams,
                args.test_all_feature_configs,
                args.baseline_only,
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
