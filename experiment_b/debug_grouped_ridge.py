#!/usr/bin/env python3
"""Diagnostic script for Grouped Ridge in Experiment B.

This script verifies that Grouped Ridge correctly applies different alpha
regularizations to each feature source and provides diagnostic output:
1. Coefficient contributions per source
2. Correlations between individual sources and predicted difficulty
3. Regularization verification

Usage:
    python -m experiment_b.debug_grouped_ridge
    python -m experiment_b.debug_grouped_ridge --dataset swebench_pro
"""

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from experiment_b import get_dataset_config, list_datasets
from experiment_b.shared import (
    load_and_prepare_data,
    build_feature_sources,
)
from experiment_ab_shared.feature_source import (
    GroupedFeatureSource,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
    GroupedRidgePredictor,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_coefficient_analysis(
    feature_sources: List[Tuple[str, Any]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Analyze coefficient distribution for all Grouped Ridge combinations.

    Returns:
        Dict mapping combo name -> diagnostic info
    """
    print_header("1. COEFFICIENT ANALYSIS")

    results = {}

    # Generate all combinations of size 2+
    all_combinations = []
    for r in range(2, len(feature_sources) + 1):
        all_combinations.extend(combinations(feature_sources, r))

    for source_combo in all_combinations:
        # Build grouped source
        combined = GroupedFeatureSource([source for _, source in source_combo])

        combo_name = combined.name
        print(f"\n--- Grouped Ridge ({combo_name}) ---")

        # Train predictor
        predictor = GroupedRidgePredictor(combined)
        predictor.fit(train_task_ids, ground_truth_b)

        # Get detailed diagnostics
        diag = predictor.get_detailed_diagnostics()
        results[combo_name] = diag

        # Print selected alphas
        print("\nSelected alphas:")
        for name, alpha in diag["selected_alphas"].items():
            print(f"  {name}: {alpha:.1f}")

        # Print coefficient distribution
        print("\nCoefficient distribution by source:")
        print(f"  {'Source':<20} | {'Features':>8} | {'L2 Norm':>10} | {'Mean|c|':>10} | {'Contrib%':>8}")
        print(f"  {'-' * 20}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")

        for name, info in diag["coef_by_source"].items():
            print(
                f"  {name:<20} | {info['n_features']:>8} | {info['l2_norm']:>10.4f} | "
                f"{info['mean_abs']:>10.6f} | {info['contribution_pct']:>7.1f}%"
            )

        # Print LLM Judge feature coefficients if available
        if "LLM Judge" in diag["coef_by_source"]:
            llm_info = diag["coef_by_source"]["LLM Judge"]
            if "named_coefficients" in llm_info:
                print("\nLLM Judge feature coefficients:")
                sorted_coefs = sorted(
                    llm_info["named_coefficients"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                for name, coef in sorted_coefs:
                    print(f"  {name:<35}: {coef:+.6f}")

        # Print Trajectory feature coefficients if available
        if "Trajectory" in diag["coef_by_source"]:
            traj_info = diag["coef_by_source"]["Trajectory"]
            if "named_coefficients" in traj_info:
                print("\nTrajectory feature coefficients (top 10 by magnitude):")
                sorted_coefs = sorted(
                    traj_info["named_coefficients"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:10]
                for name, coef in sorted_coefs:
                    print(f"  {name:<35}: {coef:+.6f}")

    return results


def run_correlation_analysis(
    feature_sources: List[Tuple[str, Any]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
    oracle_b: Dict[str, float],
) -> Dict[str, Any]:
    """Analyze correlations between individual sources and ground truth.

    Returns:
        Dict with correlation metrics
    """
    print_header("2. INDIVIDUAL SOURCE CORRELATIONS")

    # Train individual predictors
    individual_preds = {}
    individual_predictors = {}

    for name, source in feature_sources:
        predictor = FeatureBasedPredictor(source)
        predictor.fit(train_task_ids, ground_truth_b)
        preds = predictor.predict(train_task_ids)
        individual_preds[name] = preds
        individual_predictors[name] = predictor

        # Get selected alpha
        info = predictor.get_model_info()
        print(f"\n{name}: selected alpha = {info['best_alpha']:.1f}")

    # Get arrays for correlation
    oracle_arr = np.array([oracle_b.get(t, np.nan) for t in train_task_ids])
    gt_arr = np.array(ground_truth_b)

    # Filter out NaN values
    valid_mask = ~np.isnan(oracle_arr)
    valid_tasks = [t for t, v in zip(train_task_ids, valid_mask) if v]
    oracle_valid = oracle_arr[valid_mask]

    print("\n--- Correlations with Ground Truth (Baseline IRT β) ---")
    print(f"{'Source':<20} | {'Pearson r':>12} | {'Spearman ρ':>12}")
    print(f"{'-' * 20}-+-{'-' * 12}-+-{'-' * 12}")

    source_corrs_gt = {}
    for name, preds in individual_preds.items():
        pred_arr = np.array([preds[t] for t in train_task_ids])
        pearson_r, _ = stats.pearsonr(pred_arr, gt_arr)
        spearman_r, _ = stats.spearmanr(pred_arr, gt_arr)
        source_corrs_gt[name] = {"pearson": pearson_r, "spearman": spearman_r}
        print(f"{name:<20} | {pearson_r:>12.4f} | {spearman_r:>12.4f}")

    print("\n--- Correlations with Oracle IRT β ---")
    print(f"{'Source':<20} | {'Pearson r':>12} | {'Spearman ρ':>12}")
    print(f"{'-' * 20}-+-{'-' * 12}-+-{'-' * 12}")

    source_corrs_oracle = {}
    for name, preds in individual_preds.items():
        pred_arr = np.array([preds[t] for t in valid_tasks])
        pearson_r, _ = stats.pearsonr(pred_arr, oracle_valid)
        spearman_r, _ = stats.spearmanr(pred_arr, oracle_valid)
        source_corrs_oracle[name] = {"pearson": pearson_r, "spearman": spearman_r}
        print(f"{name:<20} | {pearson_r:>12.4f} | {spearman_r:>12.4f}")

    # Cross-correlations between sources
    print("\n--- Prediction Correlations Between Sources ---")
    source_names = list(individual_preds.keys())
    for i, name1 in enumerate(source_names):
        for name2 in source_names[i + 1 :]:
            pred1 = np.array([individual_preds[name1][t] for t in train_task_ids])
            pred2 = np.array([individual_preds[name2][t] for t in train_task_ids])
            pearson_r, _ = stats.pearsonr(pred1, pred2)

            if pearson_r > 0.7:
                redundancy = "HIGH redundancy"
            elif pearson_r > 0.4:
                redundancy = "moderate redundancy"
            else:
                redundancy = "low redundancy"

            print(f"  {name1} vs {name2}: r={pearson_r:.4f} ({redundancy})")

    return {
        "correlations_gt": source_corrs_gt,
        "correlations_oracle": source_corrs_oracle,
        "individual_preds": individual_preds,
    }


def run_regularization_verification(
    coef_results: Dict[str, Dict[str, Any]],
) -> None:
    """Verify that different alphas produce different coefficient scales."""
    print_header("3. REGULARIZATION VERIFICATION")

    for combo_name, diag in coef_results.items():
        print(f"\n--- {combo_name} ---")

        alphas = diag["selected_alphas"]
        coef_by_source = diag["coef_by_source"]

        print("\nVerification: scaling = 1/sqrt(alpha), expect larger alpha -> smaller coefficients")
        print(f"{'Source':<20} | {'Alpha':>12} | {'1/√α':>12} | {'Mean|c|':>12} | {'Status'}")
        print(f"{'-' * 20}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}")

        source_data = []
        for name, alpha in alphas.items():
            scaling = 1 / np.sqrt(alpha)
            mean_abs = coef_by_source[name]["mean_abs"]
            source_data.append((name, alpha, scaling, mean_abs))

        # Sort by alpha (descending)
        source_data.sort(key=lambda x: x[1], reverse=True)

        # Check if ordering is correct (higher alpha -> smaller mean coef)
        mean_abs_values = [d[3] for d in source_data]
        is_monotonic = all(
            mean_abs_values[i] <= mean_abs_values[i + 1] * 100  # Allow 100x tolerance
            for i in range(len(mean_abs_values) - 1)
        )

        for name, alpha, scaling, mean_abs in source_data:
            status = "✓" if is_monotonic else "?"
            print(f"{name:<20} | {alpha:>12.1f} | {scaling:>12.6f} | {mean_abs:>12.6f} | {status}")

        # Summary
        print("\nEffective regularization (higher = more shrinkage):")
        for name, alpha, scaling, mean_abs in source_data:
            n_features = coef_by_source[name]["n_features"]
            if alpha >= 10000:
                level = "very high"
            elif alpha >= 1000:
                level = "high"
            elif alpha >= 100:
                level = "medium"
            elif alpha >= 10:
                level = "low"
            else:
                level = "very low"
            print(f"  {name}: α={alpha:.0f} ({level} regularization for {n_features} features)")

        # Check for potential issues
        contributions = {name: coef_by_source[name]["contribution_pct"] for name in alphas}
        max_contrib = max(contributions.values())
        min_contrib = min(contributions.values())

        print("\nContribution balance check:")
        if max_contrib > 95:
            print(f"  WARNING: One source dominates ({max_contrib:.1f}% contribution)")
            print("  This may indicate the other source(s) are not contributing meaningfully")
        elif max_contrib > 80:
            print(f"  MODERATE: One source contributes {max_contrib:.1f}%")
            print("  Consider if this imbalance is expected")
        else:
            print(f"  GOOD: Contributions are balanced ({min_contrib:.1f}% - {max_contrib:.1f}%)")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug Grouped Ridge for Experiment B"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swebench",
        choices=list_datasets(),
        help="Dataset to analyze (default: swebench)",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Override cutoff date (YYYYMMDD format)",
    )
    return parser.parse_args()


def main():
    """Run all diagnostic analyses."""
    args = parse_args()

    print_header("GROUPED RIDGE DIAGNOSTICS FOR EXPERIMENT B")

    # Load configuration
    print(f"\nLoading dataset: {args.dataset}")
    config = get_dataset_config(args.dataset)

    # Create a minimal args namespace for load_and_prepare_data
    class MinimalArgs:
        pass

    minimal_args = MinimalArgs()
    minimal_args.cutoff_date = args.cutoff_date
    minimal_args.responses_path = None
    minimal_args.baseline_irt_path = None
    minimal_args.oracle_irt_path = None
    minimal_args.oracle_abilities_path = None
    minimal_args.embeddings_path = None
    minimal_args.llm_judge_path = None
    minimal_args.trajectory_features_path = None
    minimal_args.pre_threshold = None
    minimal_args.post_threshold = None
    minimal_args.filter_bottom_percentile = 0.0
    minimal_args.min_oracle_ability = None
    minimal_args.frontier_definitions = ["passrate"]
    minimal_args.verbose = False

    # Load data
    data = load_and_prepare_data(minimal_args, config)

    # Build feature sources
    feature_sources = build_feature_sources(config)

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Cutoff: {data.cutoff_date}")
    print(f"  Pre-frontier agents: {len(data.pre_frontier_agents)}")
    print(f"  Post-frontier agents: {len(data.post_frontier_agents)}")
    print(f"  Training tasks: {len(data.train_task_ids)}")
    print(f"\nFeature sources:")
    for name, source in feature_sources:
        print(f"  {name}: {source.feature_dim} features")

    if len(feature_sources) < 2:
        print("\nERROR: Need at least 2 feature sources for Grouped Ridge analysis")
        return

    # Run analyses
    coef_results = run_coefficient_analysis(
        feature_sources,
        data.train_task_ids,
        data.baseline_ground_truth_b,
    )

    oracle_b = data.oracle_items["b"].to_dict()
    corr_results = run_correlation_analysis(
        feature_sources,
        data.train_task_ids,
        data.baseline_ground_truth_b,
        oracle_b,
    )

    run_regularization_verification(coef_results)

    print_header("ANALYSIS COMPLETE")

    # Summary
    print("\nKEY FINDINGS:")

    # 1. Check if different alphas are selected
    all_alphas = set()
    for combo_name, diag in coef_results.items():
        for alpha in diag["selected_alphas"].values():
            all_alphas.add(alpha)

    if len(all_alphas) > 1:
        print(f"  1. ✓ Different alphas are being selected: {sorted(all_alphas)}")
    else:
        print(f"  1. ? Only one alpha value used: {all_alphas}")

    # 2. Check contribution balance
    for combo_name, diag in coef_results.items():
        contribs = [info["contribution_pct"] for info in diag["coef_by_source"].values()]
        if max(contribs) < 90:
            print(f"  2. ✓ {combo_name}: balanced contributions ({min(contribs):.1f}% - {max(contribs):.1f}%)")
        else:
            print(f"  2. ? {combo_name}: imbalanced ({max(contribs):.1f}% from one source)")

    # 3. Check source correlations
    corr_gt = corr_results["correlations_gt"]
    best_source = max(corr_gt, key=lambda k: corr_gt[k]["pearson"])
    best_corr = corr_gt[best_source]["pearson"]
    print(f"  3. Best individual source: {best_source} (r={best_corr:.4f} with baseline IRT)")


if __name__ == "__main__":
    main()
