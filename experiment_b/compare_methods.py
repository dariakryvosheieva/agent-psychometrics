#!/usr/bin/env python3
"""Compare methods for frontier task difficulty prediction.

This script compares:
1. Oracle (upper bound): True IRT difficulties
2. Baseline IRT: Train IRT on pre-frontier agents only
3. Embedding + Ridge: Task embeddings with Ridge regression
4. LLM Judge + Ridge: LLM-extracted semantic features with Ridge
5. SAD-IRT (optional): From experiment_sad_irt extracted beta values

Methods are evaluated by:
- Spearman correlation with oracle IRT difficulties on frontier tasks
- ROC-AUC on frontier tasks using oracle abilities and aligned difficulties
- (Optional) Date forecasting: predicting when tasks become solvable

The AUC metric requires aligning predicted difficulties to the oracle scale using
an affine transformation fitted on "nontrivial" anchor tasks (10-90% pass rate in
both agent groups). This alignment uses oracle information and is ONLY for evaluation.

Usage:
    python -m experiment_b.compare_methods
    python -m experiment_b.compare_methods --embeddings_path path/to/embeddings.npz
    python -m experiment_b.compare_methods --output_csv chris_output/experiment_b_results.csv
    python -m experiment_b.compare_methods --alignment_method affine
    python -m experiment_b.compare_methods --forecast_dates
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from experiment_b.shared.data_splits import (
    get_all_agents_from_responses,
    identify_frontier_tasks,
    identify_frontier_tasks_irt,
    identify_nontrivial_tasks,
    split_agents_by_dates,
)
from experiment_b import get_dataset_config, list_datasets
from experiment_b.shared.evaluate import (
    compute_frontier_difficulty_metrics,
    compute_scale_offset,
    shift_to_oracle_scale,
    compute_frontier_auc,
    load_responses_dict,
)
from experiment_b.shared.baseline_irt import get_or_train_baseline_irt
from experiment_b.shared.date_forecasting import (
    compute_first_capable_dates,
    compute_ground_truth_days,
    split_tasks_by_first_capable_date,
    DateForecastModel,
    compute_date_forecast_metrics,
    parse_date,
)

# Import predictors from shared module
from shared.predictor_base import DifficultyPredictorBase
from shared.feature_source import EmbeddingFeatureSource, CSVFeatureSource
from shared.feature_predictor import FeatureBasedPredictor
from experiment_b.shared.feature_irt_predictor import FeatureIRTPredictor


def compute_method_metrics(
    predicted_beta: Dict[str, float],
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    anchor_task_ids: List[str],
    eval_agents: List[str],
    alignment_method: str = "affine",
) -> Dict[str, Any]:
    """Compute all metrics for a difficulty prediction method.

    Args:
        predicted_beta: Dict mapping task_id -> predicted difficulty
        oracle_items: DataFrame with 'b' column (oracle difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle abilities)
        responses: Response matrix as nested dict
        frontier_task_ids: Tasks to evaluate Spearman/AUC on
        anchor_task_ids: Tasks for fitting the alignment transformation
        eval_agents: Agents to use for AUC computation (post-frontier)
        alignment_method: "constant" or "affine"

    Returns:
        Dict with spearman, pearson, auc, and alignment info
    """
    oracle_beta = oracle_items["b"].to_dict()

    # 1. Spearman correlation (no alignment needed - rank-based)
    spearman_metrics = compute_frontier_difficulty_metrics(
        predicted_beta, oracle_beta, frontier_task_ids
    )

    # 2. Compute alignment parameters using anchor tasks
    alignment_params = compute_scale_offset(
        predicted_beta, oracle_beta, anchor_task_ids, method=alignment_method
    )

    # 3. Shift predictions to oracle scale
    shifted_beta = shift_to_oracle_scale(predicted_beta, alignment_params)

    # 4. Compute AUC on frontier tasks using post-frontier agents
    auc_metrics = compute_frontier_auc(
        oracle_abilities, shifted_beta, responses, frontier_task_ids, eval_agents
    )

    return {
        **spearman_metrics,
        "auc": auc_metrics.get("auc"),
        "auc_n_pairs": auc_metrics.get("n_pairs"),
        "auc_n_positive": auc_metrics.get("n_positive"),
        "auc_n_negative": auc_metrics.get("n_negative"),
        "alignment_method": alignment_method,
        "alignment_params": alignment_params,
    }


def evaluate_predictor(
    predictor: DifficultyPredictorBase,
    baseline_items: pd.DataFrame,
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    train_task_ids: List[str],
    anchor_task_ids: List[str],
    eval_agents: List[str],
    alignment_method: str = "affine",
    sanity_check: bool = True,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Train a difficulty predictor on non-frontier tasks, evaluate with full metrics.

    Args:
        predictor: DifficultyPredictorBase instance (already initialized with data path)
        baseline_items: DataFrame with 'b' column (training targets)
        oracle_items: DataFrame with 'b' column (oracle difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle abilities)
        responses: Response matrix as nested dict
        frontier_task_ids: List of frontier task IDs (evaluation)
        train_task_ids: List of training task IDs
        anchor_task_ids: List of anchor task IDs (for scale alignment)
        eval_agents: Agents to use for AUC (post-frontier)
        alignment_method: "constant" or "affine"
        sanity_check: If True, print train set correlation
        return_predictions: If True, also return raw predictions dict

    Returns:
        Dict with spearman, pearson, auc metrics (and optionally 'raw_predictions')
    """
    # Get training data
    train_tasks_available = [t for t in train_task_ids if t in baseline_items.index]
    ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values

    # Fit predictor
    predictor.fit(train_tasks_available, ground_truth_b)

    # Sanity check: evaluate on training tasks
    if sanity_check:
        train_predictions = predictor.predict(train_tasks_available)
        baseline_dict = baseline_items["b"].to_dict()
        train_metrics = compute_frontier_difficulty_metrics(
            train_predictions, baseline_dict, train_tasks_available
        )
        print(f"    [Sanity check] Train set Spearman rho: {train_metrics['frontier_spearman_rho']:.4f}")

    # Predict for all relevant tasks (frontier + anchor for alignment)
    all_tasks = list(set(frontier_task_ids + anchor_task_ids))
    predictions = predictor.predict(all_tasks)

    # Compute full metrics
    metrics = compute_method_metrics(
        predicted_beta=predictions,
        oracle_items=oracle_items,
        oracle_abilities=oracle_abilities,
        responses=responses,
        frontier_task_ids=frontier_task_ids,
        anchor_task_ids=anchor_task_ids,
        eval_agents=eval_agents,
        alignment_method=alignment_method,
    )

    if return_predictions:
        # For date forecasting, predict for ALL tasks (train + frontier + anchor)
        all_tasks_for_dates = list(set(frontier_task_ids + anchor_task_ids + train_task_ids))
        metrics["raw_predictions"] = predictor.predict(all_tasks_for_dates)

    return metrics


def evaluate_predictor_with_responses(
    predictor,
    baseline_items: pd.DataFrame,
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    train_task_ids: List[str],
    anchor_task_ids: List[str],
    eval_agents: List[str],
    train_agents: List[str],
    alignment_method: str = "affine",
    sanity_check: bool = True,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate predictor that requires response data for training.

    CRITICAL: This function filters responses to only include train_agents
    BEFORE passing to the predictor, ensuring no data leakage.

    Args:
        predictor: Predictor instance (e.g., FeatureIRTPredictor)
        baseline_items: DataFrame with 'b' column (training targets)
        oracle_items: DataFrame with 'b' column (oracle difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle abilities)
        responses: Response matrix as nested dict
        frontier_task_ids: List of frontier task IDs (evaluation)
        train_task_ids: List of training task IDs
        anchor_task_ids: List of anchor task IDs (for scale alignment)
        eval_agents: Agents to use for AUC (post-frontier)
        train_agents: Pre-frontier agents for training (CRITICAL: no data leakage)
        alignment_method: "constant" or "affine"
        sanity_check: If True, print train set correlation
        return_predictions: If True, also return raw predictions dict

    Returns:
        Dict with spearman, pearson, auc metrics (and optionally 'raw_predictions')
    """
    # Get training data
    train_tasks_available = [t for t in train_task_ids if t in baseline_items.index]
    ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values

    # CRITICAL: Filter responses to ONLY include pre-frontier (train) agents
    # This prevents any data leakage from post-frontier agents
    train_responses = {
        agent_id: agent_responses
        for agent_id, agent_responses in responses.items()
        if agent_id in train_agents
    }
    print(f"    Training with {len(train_responses)} pre-frontier agents")

    # Fit with pre-filtered responses only
    predictor.fit(
        task_ids=train_tasks_available,
        ground_truth_b=ground_truth_b,
        responses=train_responses,
    )

    # Sanity check: evaluate on training tasks
    if sanity_check:
        train_predictions = predictor.predict(train_tasks_available)
        baseline_dict = baseline_items["b"].to_dict()
        train_metrics = compute_frontier_difficulty_metrics(
            train_predictions, baseline_dict, train_tasks_available
        )
        print(f"    [Sanity check] Train set Spearman rho: {train_metrics['frontier_spearman_rho']:.4f}")

    # Predict for all relevant tasks (frontier + anchor for alignment)
    all_tasks = list(set(frontier_task_ids + anchor_task_ids))
    # Filter to only tasks that were in training (FeatureIRTPredictor requires this)
    all_tasks = [t for t in all_tasks if t in train_tasks_available]
    predictions = predictor.predict(all_tasks)

    # Compute full metrics
    metrics = compute_method_metrics(
        predicted_beta=predictions,
        oracle_items=oracle_items,
        oracle_abilities=oracle_abilities,
        responses=responses,
        frontier_task_ids=frontier_task_ids,
        anchor_task_ids=anchor_task_ids,
        eval_agents=eval_agents,
        alignment_method=alignment_method,
    )

    if return_predictions:
        # For date forecasting, return predictions for ALL tasks we have
        # (FeatureIRTPredictor can only predict for tasks in train_tasks_available)
        metrics["raw_predictions"] = predictor.predict(train_tasks_available)

    return metrics


def print_comparison_table(
    results: Dict[str, Dict],
    frontier_task_count: int,
    pre_frontier_count: int,
    post_frontier_count: int,
    anchor_task_count: int = 0,
    alignment_method: str = "affine",
    cutoff_date: str = "20250401",
    frontier_definition: str = "passrate",
    irt_solve_prob: float = 0.5,
    date_results: Optional[Dict[str, Dict]] = None,
    last_agent_date: Optional[str] = None,
    verbose: bool = False,
    dataset_name: str = "",
) -> None:
    """Print formatted comparison table."""
    print("=" * 90)
    print("EXPERIMENT B: FRONTIER TASK DIFFICULTY PREDICTION")
    print("=" * 90)
    print()
    if frontier_definition == "irt":
        print("Frontier Task Definition (IRT-based):")
        print(f"  - No pre-frontier agent has >={irt_solve_prob:.0%} solve probability")
    else:
        print("Frontier Task Definition (pass-rate based):")
        print("  - Pre-frontier pass rate <= 10%")
        print("  - Post-frontier pass rate > 10%")
    # Format cutoff date as YYYY-MM-DD for readability
    cutoff_formatted = f"{cutoff_date[:4]}-{cutoff_date[4:6]}-{cutoff_date[6:]}"
    if last_agent_date:
        print(f"  - Date range: {cutoff_formatted} to {last_agent_date}")
    else:
        print(f"  - Cutoff date: {cutoff_formatted}")
    print()
    print("Data Summary:")
    print(f"  - Pre-frontier agents: {pre_frontier_count}")
    print(f"  - Post-frontier agents: {post_frontier_count}")
    print(f"  - Frontier tasks: {frontier_task_count}")
    print(f"  - Anchor tasks (for AUC alignment): {anchor_task_count}")
    print(f"  - Alignment method: {alignment_method}")
    print()

    # Print alignment parameters if verbose
    if verbose:
        print("=" * 90)
        print("ALIGNMENT PARAMETERS (fitted on anchor tasks)")
        print("=" * 90)
        print()
        if alignment_method == "affine":
            print(f"{'Method':<45} {'Slope':>10} {'Intercept':>12} {'R²':>10}")
        else:
            print(f"{'Method':<45} {'Offset':>10}")
        print("-" * 90)

        for method, metrics in results.items():
            params = metrics.get("alignment_params", {})
            if alignment_method == "affine":
                slope = params.get("slope", float("nan"))
                intercept = params.get("intercept", float("nan"))
                r2 = params.get("r_squared", float("nan"))
                print(f"{method:<45} {slope:>10.4f} {intercept:>12.4f} {r2:>10.4f}")
            else:
                offset = params.get("offset", float("nan"))
                print(f"{method:<45} {offset:>10.4f}")

        print()

    # Build table header with dataset and frontier definition
    frontier_label = "IRT" if frontier_definition == "irt" else "Pass-rate"
    header_parts = []
    if dataset_name:
        header_parts.append(dataset_name)
    header_parts.append(f"{frontier_label} definition")
    header_parts.append(f"{frontier_task_count} frontier tasks")
    header_parts.append(f"{post_frontier_count} eval agents")
    header_line = " | ".join(header_parts)

    print("=" * 90)
    print(header_line)
    print("=" * 90)
    print()

    # Always show ROC-AUC and MAE (days) as the primary metrics
    print(f"{'Method':<45} {'ROC-AUC':>10} {'MAE (days)':>12}")
    print("-" * 68)

    # Sort by AUC (descending)
    def sort_key(item):
        auc = item[1].get("auc")
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            return float("-inf")
        return auc

    sorted_methods = sorted(results.items(), key=sort_key, reverse=True)

    for method, metrics in sorted_methods:
        auc = metrics.get("auc")

        # Format AUC
        if auc is None or (isinstance(auc, float) and np.isnan(auc)):
            auc_str = "N/A"
        else:
            auc_str = f"{auc:.4f}"

        # Get MAE from date_results if available
        if date_results:
            date_metrics = date_results.get(method, {})
            mae = date_metrics.get("mae_days", float("nan"))
            if isinstance(mae, float) and np.isnan(mae):
                mae_str = "N/A"
            else:
                mae_str = f"{mae:.1f}"
        else:
            mae_str = "N/A"

        print(f"{method:<45} {auc_str:>10} {mae_str:>12}")

    print()


def save_results_csv(results: Dict[str, Dict], output_path: Path) -> None:
    """Save results to CSV."""
    rows = []
    for method, metrics in results.items():
        rows.append({
            "method": method,
            "auc": metrics.get("auc"),
            "spearman_rho": metrics.get("frontier_spearman_rho"),
            "spearman_p": metrics.get("frontier_spearman_p"),
            "pearson_r": metrics.get("frontier_pearson_r"),
            "pearson_p": metrics.get("frontier_pearson_p"),
            "n_tasks": metrics.get("num_frontier_tasks"),
            "auc_n_pairs": metrics.get("auc_n_pairs"),
            "auc_n_positive": metrics.get("auc_n_positive"),
            "auc_n_negative": metrics.get("auc_n_negative"),
        })

    df = pd.DataFrame(rows)
    # Sort by AUC descending (with NaN at bottom)
    df = df.sort_values("auc", ascending=False, na_position="last")
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def print_date_forecast_table(
    date_results: Dict[str, Dict],
    n_frontier_total: int,
    n_frontier_with_gt: int,
    n_excluded: int,
    earliest_agent_date: str,
    latest_agent_date: str,
    cutoff_date: str,
    gt_date_min: str,
    gt_date_max: str,
) -> None:
    """Print formatted date forecasting results table."""
    print()
    print("=" * 90)
    print("DATE FORECASTING: PREDICT WHEN TASKS BECOME SOLVABLE")
    print("=" * 90)
    print()
    print("Data Summary:")
    print(f"  Post-cutoff tasks (eval set): {n_frontier_total}")
    print(f"  Tasks without any capable agent: {n_excluded}")
    print()
    print("Date Range:")
    print(f"  Earliest agent date: {earliest_agent_date}")
    print(f"  Latest agent date: {latest_agent_date}")
    print(f"  Frontier cutoff: {cutoff_date}")
    print(f"  Ground truth date range: {gt_date_min} to {gt_date_max}")
    print()
    print(f"{'Method':<45} {'MAE (days)':>12} {'Pearson r':>12} {'R²(fit)':>10} {'n':>6}")
    print("-" * 86)

    # Sort by MAE (ascending), NaN at bottom
    def sort_key(item):
        mae = item[1].get("mae_days", float("inf"))
        if isinstance(mae, float) and np.isnan(mae):
            return float("inf")
        return mae

    sorted_results = sorted(date_results.items(), key=sort_key)

    for method, metrics in sorted_results:
        mae = metrics.get("mae_days", float("nan"))
        pearson = metrics.get("pearson_r", float("nan"))
        r2_fit = metrics.get("r_squared_fit", float("nan"))
        n_tasks = metrics.get("n_tasks", 0)

        mae_str = f"{mae:.1f}" if not (isinstance(mae, float) and np.isnan(mae)) else "N/A"
        pearson_str = f"{pearson:.4f}" if not (isinstance(pearson, float) and np.isnan(pearson)) else "N/A"
        r2_str = f"{r2_fit:.4f}" if not (isinstance(r2_fit, float) and np.isnan(r2_fit)) else "N/A"

        print(f"{method:<45} {mae_str:>12} {pearson_str:>12} {r2_str:>10} {n_tasks:>6}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare methods for frontier task difficulty prediction"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swebench",
        choices=list_datasets(),
        help="Dataset to run experiment on (default: swebench)",
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        default=None,
        help="Path to response matrix JSONL (overrides dataset default)",
    )
    parser.add_argument(
        "--baseline_irt_path",
        type=Path,
        default=None,
        help="Path to baseline IRT items CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--oracle_irt_path",
        type=Path,
        default=None,
        help="Path to oracle IRT items CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--oracle_abilities_path",
        type=Path,
        default=None,
        help="Path to oracle IRT abilities CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=Path,
        default=None,
        help="Path to embeddings .npz file (overrides dataset default)",
    )
    parser.add_argument(
        "--llm_judge_path",
        type=Path,
        default=None,
        help="Path to LLM judge features CSV (overrides dataset default)",
    )
    parser.add_argument(
        "--sad_irt_beta_dir",
        type=Path,
        default=Path("chris_output/sad_irt_beta_values"),
        help="Directory containing extracted SAD-IRT beta CSV files",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Frontier cutoff date YYYYMMDD (overrides dataset default)",
    )
    parser.add_argument(
        "--pre_threshold",
        type=float,
        default=None,
        help="Max pre-frontier pass rate for frontier tasks (overrides dataset default)",
    )
    parser.add_argument(
        "--post_threshold",
        type=float,
        default=None,
        help="Min post-frontier pass rate for frontier tasks (overrides dataset default)",
    )
    parser.add_argument(
        "--alignment_method",
        type=str,
        default="affine",
        choices=["constant", "affine"],
        help="Method for aligning predicted difficulties to oracle scale (default: affine)",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional path to save results CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print alignment parameters for each method",
    )
    # Note: --train_on_all_tasks removed - now always trains on all tasks
    # (ground truth from baseline IRT trained on pre-frontier agents ensures no data leakage)
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Run grid search over Feature-IRT hyperparameters",
    )
    parser.add_argument(
        "--no_forecast_dates",
        action="store_true",
        help="Disable date forecasting evaluation (enabled by default)",
    )
    parser.add_argument(
        "--frontier_definitions",
        type=str,
        nargs="+",
        default=["passrate", "irt"],
        choices=["irt", "passrate"],
        help="Frontier definitions to evaluate (default: both). "
             "'passrate' = pass rate thresholds, 'irt' = IRT probability threshold",
    )
    args = parser.parse_args()

    # For backwards compatibility, allow --frontier_definition (singular)
    # by checking if only one definition was provided
    if len(args.frontier_definitions) == 1:
        print(f"Running with single frontier definition: {args.frontier_definitions[0]}")
    else:
        print(f"Running with frontier definitions: {', '.join(args.frontier_definitions)}")

    # Load dataset configuration
    print(f"Loading dataset configuration: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)

    # Override paths from CLI args if provided
    responses_path = args.responses_path or dataset_config.responses_path
    oracle_irt_path = args.oracle_irt_path or dataset_config.oracle_irt_path
    oracle_abilities_path = args.oracle_abilities_path or dataset_config.oracle_abilities_path
    baseline_irt_path = args.baseline_irt_path or dataset_config.baseline_irt_path
    embeddings_path = args.embeddings_path or dataset_config.embeddings_path
    llm_judge_path = args.llm_judge_path or dataset_config.llm_judge_path
    cutoff_date = args.cutoff_date or dataset_config.cutoff_date
    pre_threshold = args.pre_threshold if args.pre_threshold is not None else dataset_config.pre_threshold
    post_threshold = args.post_threshold if args.post_threshold is not None else dataset_config.post_threshold
    output_dir = dataset_config.output_dir

    print(f"  Dataset: {dataset_config.name}")
    print(f"  Cutoff date: {cutoff_date}")

    # Validate required files exist
    required_files = [
        (responses_path, "Response matrix"),
        (oracle_irt_path, "Oracle IRT"),
        (oracle_abilities_path, "Oracle abilities"),
    ]
    for path, name in required_files:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load IRT models and abilities
    print("\nLoading IRT models...")
    oracle_items = pd.read_csv(oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(oracle_abilities_path, index_col=0)
    print(f"  Oracle IRT: {len(oracle_items)} tasks")
    print(f"  Oracle abilities: {len(oracle_abilities)} agents")

    # Load response matrix for AUC computation
    print("\nLoading response matrix...")
    responses = load_responses_dict(responses_path)
    print(f"  Loaded responses for {len(responses)} agents")

    # Get agent dates from dataset config and split by cutoff
    print("\nIdentifying frontier tasks...")
    all_agents = get_all_agents_from_responses(responses_path)
    agent_dates = dataset_config.get_agent_dates(all_agents)
    print(f"  Agents with dates: {len(agent_dates)} / {len(all_agents)}")

    pre_frontier, post_frontier = split_agents_by_dates(all_agents, agent_dates, cutoff_date)
    print(f"  Pre-frontier agents (< {cutoff_date}): {len(pre_frontier)}")
    print(f"  Post-frontier agents (>= {cutoff_date}): {len(post_frontier)}")

    # Compute last agent date for display
    if agent_dates:
        all_dates = [parse_date(d) for d in agent_dates.values()]
        last_agent_date = max(all_dates).strftime("%Y-%m-%d")
    else:
        last_agent_date = None

    # Load or train baseline IRT (pre-frontier agents only)
    # Uses caching based on (responses_file, pre_frontier_agents, cutoff_date)
    if baseline_irt_path and baseline_irt_path.exists():
        # Use explicitly provided baseline IRT path (e.g., SWE-bench pre-computed)
        baseline_items = pd.read_csv(baseline_irt_path, index_col=0)
        # Try to load abilities from the same directory
        baseline_abilities_path = baseline_irt_path.parent / "abilities.csv"
        if baseline_abilities_path.exists():
            baseline_abilities = pd.read_csv(baseline_abilities_path, index_col=0)
        else:
            baseline_abilities = None
        print(f"  Baseline IRT: {len(baseline_items)} tasks (loaded from {baseline_irt_path})")
    else:
        # Use cached baseline IRT or train new one
        # Cache is invalidated if training data changes (responses, agents, or cutoff)
        print("\nLoading/training baseline IRT...")
        baseline_items, baseline_abilities = get_or_train_baseline_irt(
            responses_path=responses_path,
            pre_frontier_agents=pre_frontier,
            cutoff_date=cutoff_date,
            output_dir=output_dir,
        )
        print(f"  Baseline IRT: {len(baseline_items)} tasks, {len(baseline_abilities)} agents")

    # IRT solve probability threshold from config
    irt_solve_prob = dataset_config.irt_solve_probability

    # Identify frontier tasks for each definition
    frontier_tasks_by_def = {}
    for frontier_def in args.frontier_definitions:
        if frontier_def == "irt":
            frontier_task_ids = identify_frontier_tasks_irt(
                oracle_items=oracle_items,
                oracle_abilities=oracle_abilities,
                agent_dates=agent_dates,
                cutoff_date=cutoff_date,
                solve_probability=irt_solve_prob,
            )
            print(f"  Frontier tasks ({frontier_def}: no pre-frontier agent with >={irt_solve_prob:.0%} solve prob): {len(frontier_task_ids)}")
        else:
            frontier_task_ids = identify_frontier_tasks(
                responses_path,
                pre_frontier,
                post_frontier,
                pre_threshold,
                post_threshold,
            )
            print(f"  Frontier tasks ({frontier_def}: <={pre_threshold*100:.0f}% pre, >{post_threshold*100:.0f}% post): {len(frontier_task_ids)}")
        frontier_tasks_by_def[frontier_def] = frontier_task_ids

    # Identify nontrivial anchor tasks for scale alignment
    print("\nIdentifying nontrivial anchor tasks...")
    anchor_task_ids, _, _ = identify_nontrivial_tasks(
        responses_path,
        pre_frontier,
        post_frontier,
        min_pass_rate=0.10,
        max_pass_rate=0.90,
    )
    print(f"  Anchor tasks (10-90% pass rate in both groups): {len(anchor_task_ids)}")

    # Training tasks: use all tasks (ground truth from baseline IRT trained on pre-frontier agents)
    # This simplifies logic when evaluating multiple frontier definitions
    all_task_ids = list(baseline_items.index)
    train_task_ids = all_task_ids
    print(f"  Training tasks: {len(train_task_ids)}")

    # =========================================================================
    # PHASE 1: Collect raw predictions from all methods
    # =========================================================================
    # Store raw predicted betas (before oracle alignment)
    # These will be evaluated against each frontier definition
    raw_predictions = {}

    # 0. Oracle upper bound (uses true oracle beta)
    print("\nCollecting predictions: Oracle (upper bound)...")
    oracle_beta = oracle_items["b"].to_dict()
    raw_predictions["Oracle (upper bound)"] = oracle_beta

    # 1. Baseline IRT
    print("Collecting predictions: Baseline IRT...")
    baseline_beta = baseline_items["b"].to_dict()
    raw_predictions["Baseline IRT (pre-frontier only)"] = baseline_beta

    # 2. SAD-IRT runs (from extracted beta CSV files) - load all valid ones
    # Note: we'll select the best one per frontier definition later
    sad_irt_betas = {}
    if args.sad_irt_beta_dir.exists():
        beta_files = list(args.sad_irt_beta_dir.glob("*.csv"))
        print(f"\nLoading SAD-IRT beta values from {args.sad_irt_beta_dir}...")
        print(f"  Found {len(beta_files)} beta CSV files")

        for beta_file in beta_files:
            beta_df = pd.read_csv(beta_file, index_col=0)
            if "beta" not in beta_df.columns:
                print(f"  Skipping {beta_file.name}: no 'beta' column")
                continue
            sad_irt_betas[beta_file.stem] = beta_df["beta"].to_dict()
        print(f"  Loaded {len(sad_irt_betas)} valid SAD-IRT beta files")
    else:
        print(f"\nSAD-IRT beta directory not found: {args.sad_irt_beta_dir}")
        print("  To include SAD-IRT results, run experiment_sad_irt and extract beta values")

    # Build list of available feature sources
    feature_sources = []
    if embeddings_path.exists():
        feature_sources.append(("Embedding", EmbeddingFeatureSource(embeddings_path)))
    else:
        print(f"\nEmbeddings not found: {embeddings_path}")

    if llm_judge_path.exists():
        feature_sources.append((
            "LLM Judge",
            CSVFeatureSource(
                llm_judge_path,
                feature_cols=dataset_config.llm_judge_feature_cols,
                name="LLM Judge",
            ),
        ))
    else:
        print(f"\nLLM Judge features not found: {llm_judge_path}")

    # 3-4. Feature + Ridge predictors: train and collect predictions
    for source_name, source in feature_sources:
        method_name = f"{source_name} + Ridge"
        print(f"\nTraining {method_name}...")
        print(f"  Training on {len(train_task_ids)} tasks")
        try:
            predictor = FeatureBasedPredictor(source)
            # Fit the predictor
            train_tasks_available = [t for t in train_task_ids if t in baseline_items.index]
            ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values
            predictor.fit(train_tasks_available, ground_truth_b)

            # Sanity check on training data
            train_predictions = predictor.predict(train_tasks_available)
            from experiment_b.shared.evaluate import compute_frontier_difficulty_metrics
            baseline_dict = baseline_items["b"].to_dict()
            train_metrics = compute_frontier_difficulty_metrics(
                train_predictions, baseline_dict, train_tasks_available
            )
            print(f"    [Sanity check] Train set Spearman rho: {train_metrics['frontier_spearman_rho']:.4f}")

            # Get predictions for all tasks
            raw_predictions[method_name] = predictor.predict(all_task_ids)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # 5-6. Feature-IRT predictors: train and collect predictions
    # For grid search, use first frontier definition to select best config
    all_task_ids_baseline = list(baseline_items.index)
    primary_frontier_def = args.frontier_definitions[0]
    primary_frontier_tasks = frontier_tasks_by_def[primary_frontier_def]

    if args.grid_search:
        l2_weight_grid = [0.001, 0.01, 0.1]
        l2_residual_grid = [1.0, 10.0, 100.0]
        use_residuals_grid = [True, False]
    else:
        l2_weight_grid = [0.01]
        l2_residual_grid = [10.0]
        use_residuals_grid = [True]

    # Store Feature-IRT learned abilities for date forecasting
    feature_irt_abilities = {}

    for source_name, source in feature_sources:
        method_name = f"Feature-IRT ({source_name})"
        best_auc = -1
        best_predictions = None
        best_abilities = None

        if args.grid_search:
            print(f"\nRunning {method_name} grid search (using '{primary_frontier_def}' frontier for selection)...")
            print(f"  Training on {len(all_task_ids_baseline)} tasks")

        for l2_w in l2_weight_grid:
            for l2_r in l2_residual_grid:
                for use_res in use_residuals_grid:
                    if args.grid_search:
                        print(f"    Testing: l2_w={l2_w}, l2_r={l2_r}, res={use_res}")

                    try:
                        predictor = FeatureIRTPredictor(
                            source,
                            use_residuals=use_res,
                            l2_weight=l2_w,
                            l2_residual=l2_r,
                            verbose=args.verbose and not args.grid_search,
                        )

                        # Fit with pre-frontier agents only (no data leakage)
                        train_tasks_available = [t for t in all_task_ids_baseline if t in baseline_items.index]
                        ground_truth_b = baseline_items.loc[train_tasks_available, "b"].values
                        train_responses = {
                            agent_id: agent_responses
                            for agent_id, agent_responses in responses.items()
                            if agent_id in pre_frontier
                        }
                        if not args.grid_search:
                            print(f"\nTraining {method_name}...")
                            print(f"  Training on {len(all_task_ids_baseline)} tasks with {len(train_responses)} pre-frontier agents")

                        predictor.fit(
                            task_ids=train_tasks_available,
                            ground_truth_b=ground_truth_b,
                            responses=train_responses,
                        )

                        # Sanity check
                        if not args.grid_search:
                            train_preds = predictor.predict(train_tasks_available)
                            baseline_dict = baseline_items["b"].to_dict()
                            train_metrics = compute_frontier_difficulty_metrics(
                                train_preds, baseline_dict, train_tasks_available
                            )
                            print(f"    [Sanity check] Train set Spearman rho: {train_metrics['frontier_spearman_rho']:.4f}")

                        # Get predictions
                        predictions = predictor.predict(train_tasks_available)

                        # For grid search, compute AUC on primary frontier to select best
                        if args.grid_search:
                            metrics = compute_method_metrics(
                                predicted_beta=predictions,
                                oracle_items=oracle_items,
                                oracle_abilities=oracle_abilities,
                                responses=responses,
                                frontier_task_ids=primary_frontier_tasks,
                                anchor_task_ids=anchor_task_ids,
                                eval_agents=post_frontier,
                                alignment_method=args.alignment_method,
                            )
                            auc = metrics.get('auc', 0) or 0
                            rho = metrics.get('frontier_spearman_rho', 0) or 0
                            print(f"      AUC: {auc:.4f}, Spearman: {rho:.4f}")
                        else:
                            auc = 1.0  # No grid search, just use default config

                        if auc > best_auc:
                            best_auc = auc
                            best_predictions = predictions
                            best_abilities = predictor.learned_abilities

                    except Exception as e:
                        if args.grid_search:
                            print(f"      Error: {e}")
                        else:
                            print(f"  Error: {e}")
                            import traceback
                            traceback.print_exc()

        if best_predictions is not None:
            if args.grid_search:
                print(f"  Best AUC: {best_auc:.4f}")
            raw_predictions[method_name] = best_predictions
            if best_abilities is not None:
                feature_irt_abilities[method_name] = best_abilities

    # Date forecasting setup (enabled by default)
    # NOTE: Date forecasting requires abilities from IRT methods to fit
    # the ability-over-time regression. Methods without their own IRT
    # (Embedding + Ridge, LLM Judge + Ridge) are skipped.
    first_capable_dates = {}
    ground_truth_days = {}
    date_models = {}

    if not args.no_forecast_dates:
        print("\nRunning date forecasting evaluation...")

        # Get agent dates for ground truth computation
        agent_dates = dataset_config.get_agent_dates(all_agents)

        # Compute ground truth solvability dates (when first agent with θ >= β appeared)
        # NOTE: This uses oracle IRT and is ONLY for evaluation, not training
        gt_result = compute_first_capable_dates(oracle_items, oracle_abilities, agent_dates)
        first_capable_dates = gt_result.first_capable_dates
        tasks_without_capable = gt_result.tasks_without_capable_agent
        earliest_agent_date = gt_result.earliest_agent_date
        latest_agent_date = gt_result.latest_agent_date

        # Split tasks by whether first capable agent is pre/post cutoff
        cutoff_datetime = parse_date(cutoff_date)
        pre_cutoff_tasks, post_cutoff_tasks = split_tasks_by_first_capable_date(
            first_capable_dates, cutoff_datetime
        )

        print(f"  Tasks with ground truth (first capable agent exists): {len(first_capable_dates)}")
        print(f"  Post-cutoff tasks (for evaluation): {len(post_cutoff_tasks)}")
        print(f"  Tasks without any capable agent: {len(tasks_without_capable)}")

        if len(post_cutoff_tasks) >= 3:
            # Compute ground truth days for all tasks
            ground_truth_days = compute_ground_truth_days(
                all_task_ids, first_capable_dates, earliest_agent_date
            )

            # Get ground truth date range for post-cutoff tasks (eval set)
            post_cutoff_gt_dates = [first_capable_dates[t] for t in post_cutoff_tasks]
            gt_date_min = min(post_cutoff_gt_dates).strftime("%Y-%m-%d")
            gt_date_max = max(post_cutoff_gt_dates).strftime("%Y-%m-%d")

            # Build abilities dict for each method that has IRT abilities
            # Methods without their own IRT are skipped for date forecasting
            method_abilities = {}

            # Oracle uses oracle abilities (upper bound)
            method_abilities["Oracle (upper bound)"] = oracle_abilities["theta"].to_dict()

            # Baseline IRT uses baseline abilities
            if baseline_abilities is not None:
                method_abilities["Baseline IRT (pre-frontier only)"] = baseline_abilities["theta"].to_dict()
            else:
                print("  Warning: Baseline IRT abilities not available for date forecasting")

            # Feature-IRT learned abilities
            for method_name, abilities in feature_irt_abilities.items():
                method_abilities[method_name] = abilities

            # SAD-IRT: match beta and theta files by stem name
            # Load theta files and match with corresponding beta files
            sad_irt_theta_dir = Path("chris_output/sad_irt_theta_values")
            sad_irt_matched = {}  # stem -> (beta_dict, theta_dict)
            if sad_irt_theta_dir.exists() and sad_irt_betas:
                for theta_file in sad_irt_theta_dir.glob("*.csv"):
                    stem = theta_file.stem
                    if stem in sad_irt_betas:
                        theta_df = pd.read_csv(theta_file, index_col=0)
                        if "theta" in theta_df.columns:
                            sad_irt_matched[stem] = (sad_irt_betas[stem], theta_df["theta"].to_dict())

                # Select best SAD-IRT based on task coverage (use first frontier def)
                if sad_irt_matched:
                    primary_frontier = frontier_tasks_by_def[args.frontier_definitions[0]]
                    best_coverage = 0
                    best_stem = None
                    for stem, (beta_dict, theta_dict) in sad_irt_matched.items():
                        coverage = len([t for t in primary_frontier if t in beta_dict])
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_stem = stem

                    if best_stem:
                        beta_dict, theta_dict = sad_irt_matched[best_stem]
                        raw_predictions["SAD-IRT (best)"] = beta_dict
                        method_abilities["SAD-IRT (best)"] = theta_dict
                        print(f"  Using SAD-IRT run: {best_stem}")

            # Pre-fit date models for each method (fit is the same for all frontier defs)
            # The fit uses pre-frontier abilities, evaluation will use frontier tasks
            date_models = {}
            for method_name in method_abilities:
                abilities = method_abilities[method_name]
                try:
                    date_model = DateForecastModel()
                    fit_stats = date_model.fit(abilities, agent_dates)
                    date_models[method_name] = (date_model, fit_stats)
                except Exception as e:
                    print(f"  Warning: Could not fit date model for {method_name}: {e}")

    # =========================================================================
    # PHASE 2: Compute metrics and print tables for each frontier definition
    # =========================================================================
    all_results = {}  # Store results per frontier definition

    for frontier_def in args.frontier_definitions:
        frontier_task_ids = frontier_tasks_by_def[frontier_def]
        print(f"\n{'='*90}")
        print(f"Evaluating metrics for frontier definition: {frontier_def}")
        print(f"{'='*90}")

        results = {}

        # Compute metrics for each method
        for method_name, pred_beta in raw_predictions.items():
            try:
                metrics = compute_method_metrics(
                    predicted_beta=pred_beta,
                    oracle_items=oracle_items,
                    oracle_abilities=oracle_abilities,
                    responses=responses,
                    frontier_task_ids=frontier_task_ids,
                    anchor_task_ids=anchor_task_ids,
                    eval_agents=post_frontier,
                    alignment_method=args.alignment_method,
                )
                results[method_name] = metrics
            except Exception as e:
                print(f"  Error computing metrics for {method_name}: {e}")
                results[method_name] = {
                    "frontier_spearman_rho": float("nan"),
                    "frontier_spearman_p": float("nan"),
                    "auc": None,
                    "num_frontier_tasks": 0,
                }

        # Add SAD-IRT (select best for this frontier definition)
        # Skip if already in raw_predictions (already selected in date forecasting phase)
        if sad_irt_betas and "SAD-IRT (best)" not in raw_predictions:
            best_sad_irt_auc = -1
            best_sad_irt_metrics = None
            best_sad_irt_name = None

            for sad_name, sad_beta in sad_irt_betas.items():
                # Check coverage
                frontier_overlap = [t for t in frontier_task_ids if t in sad_beta]
                if len(frontier_overlap) == 0:
                    continue

                try:
                    sad_metrics = compute_method_metrics(
                        predicted_beta=sad_beta,
                        oracle_items=oracle_items,
                        oracle_abilities=oracle_abilities,
                        responses=responses,
                        frontier_task_ids=frontier_task_ids,
                        anchor_task_ids=anchor_task_ids,
                        eval_agents=post_frontier,
                        alignment_method=args.alignment_method,
                    )
                    auc = sad_metrics.get('auc') or 0
                    if auc > best_sad_irt_auc:
                        best_sad_irt_auc = auc
                        best_sad_irt_metrics = sad_metrics
                        best_sad_irt_name = sad_name
                except Exception:
                    pass

            if best_sad_irt_metrics is not None:
                results["SAD-IRT (best)"] = best_sad_irt_metrics

        all_results[frontier_def] = results

        # Compute date forecasting for this frontier definition
        # Evaluate on frontier tasks that have ground truth (first capable agent exists)
        date_results_for_def = None
        if not args.no_forecast_dates and date_models:
            # Find frontier tasks with ground truth
            eval_tasks = [t for t in frontier_task_ids if t in first_capable_dates]

            if len(eval_tasks) >= 3:
                date_results_for_def = {}
                for method_name, pred_beta in raw_predictions.items():
                    if method_name not in date_models:
                        continue

                    date_model, fit_stats = date_models[method_name]
                    try:
                        predictions = date_model.predict(pred_beta, eval_tasks)
                        metrics = compute_date_forecast_metrics(
                            predictions, ground_truth_days, eval_tasks
                        )
                        date_results_for_def[method_name] = {
                            **metrics,
                            "r_squared_fit": fit_stats["r_squared"],
                        }
                    except Exception as e:
                        date_results_for_def[method_name] = {
                            "mae_days": float("nan"),
                            "pearson_r": float("nan"),
                            "spearman_rho": float("nan"),
                            "r_squared_fit": float("nan"),
                            "n_tasks": 0,
                        }

        # Print comparison table for this frontier definition
        print()
        print_comparison_table(
            results,
            len(frontier_task_ids),
            len(pre_frontier),
            len(post_frontier),
            anchor_task_count=len(anchor_task_ids),
            alignment_method=args.alignment_method,
            cutoff_date=cutoff_date,
            frontier_definition=frontier_def,
            irt_solve_prob=irt_solve_prob,
            date_results=date_results_for_def,
            last_agent_date=last_agent_date,
            verbose=args.verbose,
            dataset_name=dataset_config.name,
        )

    # Save to CSV if requested (use first frontier definition for backwards compatibility)
    if args.output_csv:
        primary_def = args.frontier_definitions[0]
        save_results_csv(all_results[primary_def], args.output_csv)


if __name__ == "__main__":
    main()
