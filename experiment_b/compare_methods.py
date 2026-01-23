#!/usr/bin/env python3
"""Compare methods for frontier task difficulty prediction.

This script compares:
1. Oracle (upper bound): True IRT difficulties
2. Baseline IRT: Train IRT on pre-frontier agents only
3. Embedding + Ridge: Task embeddings with Ridge regression
4. LLM Judge + Ridge: LLM-extracted semantic features with Ridge
5. Feature-IRT: Joint learning of feature weights and agent abilities
6. SAD-IRT (optional): From experiment_sad_irt extracted beta values

Methods are evaluated by:
- ROC-AUC on frontier tasks using oracle abilities and aligned difficulties
- (Optional) Date forecasting: predicting when tasks become solvable

The AUC metric requires aligning predicted difficulties to the oracle scale using
an affine transformation fitted on "nontrivial" anchor tasks (10-90% pass rate in
both agent groups). This alignment uses oracle information and is ONLY for evaluation.

Usage:
    python -m experiment_b.compare_methods
    python -m experiment_b.compare_methods --dataset terminalbench
    python -m experiment_b.compare_methods --output_csv chris_output/experiment_b_results.csv
    python -m experiment_b.compare_methods --no_forecast_dates
    python -m experiment_b.compare_methods --frontier_definitions passrate
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

from experiment_b import get_dataset_config, list_datasets

# Easy-to-change default: set to 0.2 to remove bottom 20% of agents after validation
DEFAULT_FILTER_BOTTOM_PERCENTILE = 0.0
from experiment_b.shared import (
    # Data preparation
    load_and_prepare_data,
    # Prediction methods
    build_feature_sources,
    collect_ridge_predictions,
    collect_grouped_ridge_predictions,
    collect_feature_irt_predictions,
    collect_sad_irt_predictions,
    # Frontier evaluation
    setup_date_forecasting,
    evaluate_all_frontier_definitions,
    # Output
    save_results_csv,
    save_and_plot_diagnostics,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        "--filter_bottom_percentile",
        type=float,
        default=DEFAULT_FILTER_BOTTOM_PERCENTILE,
        help="Remove bottom X%% of post-frontier agents by frontier success rate (0.0-1.0). "
             "E.g., 0.2 removes bottom 20%%. Default: 0.0 (no filtering).",
    )
    parser.add_argument(
        "--min_oracle_ability",
        type=float,
        default=None,
        help="[Research] Minimum oracle theta for evaluation agents. "
             "Warning: may bias AUC results since oracle theta is used in evaluation.",
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
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Run grid search over Feature-IRT hyperparameters",
    )
    parser.add_argument(
        "--diagnostic_mode",
        action="store_true",
        help="Run extended diagnostics for Feature-IRT: wider hyperparameter grid, "
             "training loss curves, and feature/residual contribution analysis",
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
        choices=["irt", "passrate", "zero_pre"],
        help="Frontier definitions to evaluate (default: passrate, irt). "
             "'passrate' = pass rate thresholds, 'irt' = IRT probability threshold, "
             "'zero_pre' = 0%% pre-frontier, >0%% post-frontier",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Log frontier definitions
    if len(args.frontier_definitions) == 1:
        print(f"Running with single frontier definition: {args.frontier_definitions[0]}")
    else:
        print(f"Running with frontier definitions: {', '.join(args.frontier_definitions)}")

    # Load dataset configuration
    print(f"Loading dataset configuration: {args.dataset}")
    config = get_dataset_config(args.dataset)

    # =========================================================================
    # 1. Load data and prepare experiment
    # =========================================================================
    data = load_and_prepare_data(args, config)

    # =========================================================================
    # 2. Collect predictions from all methods
    # =========================================================================
    raw_predictions: Dict[str, Dict[str, float]] = {
        "Oracle (upper bound)": data.oracle_items["b"].to_dict(),
        "Baseline IRT (pre-frontier only)": data.baseline_items["b"].to_dict(),
    }

    # Track abilities for methods that have their own IRT (for date forecasting)
    method_abilities: Dict[str, Dict[str, float]] = {
        "Oracle (upper bound)": data.oracle_abilities["theta"].to_dict(),
    }
    if data.baseline_abilities is not None:
        method_abilities["Baseline IRT (pre-frontier only)"] = data.baseline_abilities["theta"].to_dict()

    # Load SAD-IRT predictions (each run added as separate method)
    sad_preds, sad_abilities = collect_sad_irt_predictions(args.sad_irt_beta_dir)
    raw_predictions.update(sad_preds)
    method_abilities.update(sad_abilities)

    # Build feature sources and collect Ridge predictions
    feature_sources = build_feature_sources(
        config,
        embeddings_path_override=args.embeddings_path,
        llm_judge_path_override=args.llm_judge_path,
    )
    raw_predictions.update(
        collect_ridge_predictions(
            feature_sources,
            data.train_task_ids,
            data.baseline_ground_truth_b,
            data.config.all_task_ids,
        )
    )

    # Collect grouped ridge predictions (combines all sources with per-source regularization)
    raw_predictions.update(
        collect_grouped_ridge_predictions(
            feature_sources,
            data.train_task_ids,
            data.baseline_ground_truth_b,
            data.config.all_task_ids,
        )
    )

    # Run Feature-IRT methods
    primary_frontier_tasks = data.frontier_tasks_by_def[args.frontier_definitions[0]]
    feature_irt = collect_feature_irt_predictions(
        feature_sources=feature_sources,
        train_task_ids=data.train_task_ids,
        ground_truth_b=data.baseline_ground_truth_b,
        train_responses=data.train_responses,
        oracle_items=data.oracle_items,
        oracle_abilities=data.oracle_abilities,
        responses=data.config.responses,
        frontier_task_ids=primary_frontier_tasks,
        anchor_task_ids=data.anchor_task_ids,
        post_frontier_agents=data.post_frontier_agents,
        alignment_method=args.alignment_method,
        grid_search=args.grid_search,
        diagnostic_mode=args.diagnostic_mode,
        verbose=args.verbose,
    )
    raw_predictions.update(feature_irt.predictions)
    method_abilities.update(feature_irt.abilities)

    # =========================================================================
    # 3. Setup date forecasting (optional)
    # =========================================================================
    date_info = None
    if not args.no_forecast_dates:
        date_info = setup_date_forecasting(data, raw_predictions, method_abilities)

    # =========================================================================
    # 4. Evaluate all methods and print results
    # =========================================================================
    all_results = evaluate_all_frontier_definitions(
        frontier_definitions=args.frontier_definitions,
        data=data,
        raw_predictions=raw_predictions,
        date_info=date_info,
        alignment_method=args.alignment_method,
        verbose=args.verbose,
    )

    # =========================================================================
    # 5. Save outputs
    # =========================================================================
    if args.output_csv:
        primary_def = args.frontier_definitions[0]
        save_results_csv(all_results[primary_def], args.output_csv)

    if args.diagnostic_mode and feature_irt.diagnostics:
        save_and_plot_diagnostics(feature_irt.diagnostics, config.output_dir)


if __name__ == "__main__":
    main()
