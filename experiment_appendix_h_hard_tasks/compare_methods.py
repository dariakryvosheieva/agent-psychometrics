#!/usr/bin/env python3
"""Compare methods for frontier task difficulty prediction.

This script compares:
1. Oracle (upper bound): True IRT difficulties
2. Baseline IRT: Train IRT on pre-frontier agents only
3. Embedding + Ridge: Task embeddings with Ridge regression
4. LLM Judge + Ridge: LLM-extracted semantic features with Ridge
5. Feature-IRT: Joint learning of feature weights and agent abilities

Methods are evaluated by:
- ROC-AUC on frontier tasks using oracle abilities and aligned difficulties
- (Optional) Date forecasting: predicting when tasks become solvable

The AUC metric requires aligning predicted difficulties to the oracle scale using
an affine transformation fitted on "nontrivial" anchor tasks (10-90% pass rate in
both agent groups). This alignment uses oracle information and is ONLY for evaluation.

Usage:
    python -m experiment_appendix_h_hard_tasks.compare_methods
    python -m experiment_appendix_h_hard_tasks.compare_methods --dataset terminalbench
    python -m experiment_appendix_h_hard_tasks.compare_methods --output_csv output/experiment_b_results.csv
    python -m experiment_appendix_h_hard_tasks.compare_methods --forecast_dates
    python -m experiment_appendix_h_hard_tasks.compare_methods --frontier_definitions passrate irt
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

from experiment_appendix_h_hard_tasks import get_dataset_config, list_datasets

# Easy-to-change default: set to 0.2 to remove bottom 20% of agents after validation
DEFAULT_FILTER_BOTTOM_PERCENTILE = 0.0
from experiment_appendix_h_hard_tasks.shared import (
    # Data preparation
    load_and_prepare_data,
    # Prediction methods
    build_feature_sources,
    collect_ridge_predictions,
    collect_grouped_ridge_predictions,
    collect_feature_irt_predictions,
    # Frontier evaluation
    setup_date_forecasting,
    evaluate_all_frontier_definitions,
    # Output
    save_results_csv,
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
        "--trajectory_features_path",
        type=Path,
        default=None,
        help="Path to trajectory features CSV (overrides dataset default)",
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
        "--forecast_dates",
        action="store_true",
        help="Enable date forecasting evaluation (disabled by default)",
    )
    parser.add_argument(
        "--frontier_definitions",
        type=str,
        nargs="+",
        default=["zero_pre"],
        choices=["irt", "passrate", "zero_pre", "pre_only", "human_hard"],
        help="Frontier definitions to evaluate (default: zero_pre). "
             "'zero_pre' = 0%% pre, >0%% post; "
             "'passrate' = <=X%% pre AND >Y%% post; "
             "'pre_only' = <=X%% pre (no post filter, uses --pre_threshold); "
             "'irt' = IRT probability threshold; "
             "'human_hard' = human-labeled difficulty >= '1-4 hours'",
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
    # DIAGNOSTIC: β distributions for zero_pre frontier tasks
    # =========================================================================
    if "zero_pre" in data.frontier_tasks_by_def:
        import numpy as np
        from scipy.stats import spearmanr

        frontier_tasks = data.frontier_tasks_by_def["zero_pre"]
        baseline_betas = []
        oracle_betas = []
        for t in frontier_tasks:
            if t in data.baseline_items.index:
                baseline_betas.append(data.baseline_items.loc[t, "b"])
            if t in data.oracle_items.index:
                oracle_betas.append(data.oracle_items.loc[t, "b"])

        print(f"\n=== DIAGNOSTIC: β distributions for {len(frontier_tasks)} zero_pre frontier tasks ===")
        print(f"Baseline β: mean={np.mean(baseline_betas):.3f}, std={np.std(baseline_betas):.3f}, "
              f"range=[{min(baseline_betas):.3f}, {max(baseline_betas):.3f}]")
        print(f"Oracle β:   mean={np.mean(oracle_betas):.3f}, std={np.std(oracle_betas):.3f}, "
              f"range=[{min(oracle_betas):.3f}, {max(oracle_betas):.3f}]")

        # Check correlation between baseline and oracle
        if len(baseline_betas) >= 3 and len(oracle_betas) >= 3:
            corr, p = spearmanr(baseline_betas, oracle_betas)
            print(f"Correlation (Baseline vs Oracle): Spearman r={corr:.3f}, p={p:.4f}")

            # Check if task index order correlates with oracle difficulty
            corr_idx, p_idx = spearmanr(range(len(oracle_betas)), oracle_betas)
            print(f"Correlation (Task index vs Oracle β): Spearman r={corr_idx:.3f}, p={p_idx:.4f}")

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

    # Build feature sources and collect Ridge predictions
    feature_sources = build_feature_sources(
        config,
        embeddings_path_override=args.embeddings_path,
        llm_judge_path_override=args.llm_judge_path,
        trajectory_features_path_override=args.trajectory_features_path,
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

    # Run Feature-IRT methods (with grid search over hyperparameters)
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
        baseline_abilities=data.baseline_abilities,
        alignment_method=args.alignment_method,
        verbose=args.verbose,
    )
    raw_predictions.update(feature_irt.predictions)
    method_abilities.update(feature_irt.abilities)

    # Print baseline-init diagnostics if available
    if args.verbose and feature_irt.baseline_init_diagnostics:
        for source_name, diag in feature_irt.baseline_init_diagnostics.items():
            print(f"\n=== Diagnostics: Baseline-Init ({source_name}) ===")
            print(f"  Weight norm: {diag['weight_norm']:.4f}")
            print(f"  Difficulty drift (mean): {diag['difficulty_drift_mean']:.4f}")
            print(f"  Ability drift (mean): {diag['ability_drift_mean']:.4f}")
            print(f"  Feature contribution: {diag['feature_contribution_ratio']:.2%}")

    # =========================================================================
    # 3. Setup date forecasting (optional)
    # =========================================================================
    date_info = None
    if args.forecast_dates:
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


if __name__ == "__main__":
    main()
