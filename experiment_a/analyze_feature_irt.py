"""Analyze Feature-IRT predictions: correlation with Oracle, feature vs residual contributions.

This script decomposes Feature-IRT predictions into:
- Feature contribution: w^T f + bias
- Difficulty latent (residual): r_i

And compares with Oracle difficulties to understand what each component captures.

Usage:
    python -m experiment_a.analyze_feature_irt
"""

import numpy as np
import pandas as pd
from pathlib import Path

from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared.feature_source import (
    build_feature_sources,
    GroupedFeatureSource,
    RegularizedFeatureSource,
)
from experiment_ab_shared.dataset import _load_binary_responses
from experiment_b.shared.prediction_methods import FeatureIRTPredictor

ROOT = Path(__file__).resolve().parents[1]


def analyze_feature_irt(
    l2_weight: float = 0.001,
    l2_residual: float = 0.0001,
):
    """Analyze Feature-IRT predictions vs Oracle.

    Args:
        l2_weight: L2 regularization on feature weights
        l2_residual: L2 regularization on difficulty latents (residuals)
    """
    config = ExperimentAConfig()

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path
    embeddings_path = ROOT / config.embeddings_path if config.embeddings_path else None
    llm_judge_path = ROOT / config.llm_judge_features_path if config.llm_judge_features_path else None
    trajectory_path = ROOT / config.trajectory_features_path if getattr(config, "trajectory_features_path", None) else None

    # Load data directly
    full_abilities = pd.read_csv(abilities_path, index_col=0)
    full_items = pd.read_csv(items_path, index_col=0)
    responses = _load_binary_responses(responses_path)

    all_task_ids = list(full_items.index)
    oracle_b = full_items.loc[all_task_ids, "b"].values

    print(f"Tasks: {len(all_task_ids)}")
    print(f"Agents: {len(responses)}")
    print(f"Oracle b: mean={oracle_b.mean():.4f}, std={oracle_b.std():.4f}")
    print()

    # Build feature sources
    feature_source_list = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        trajectory_features_path=trajectory_path,
        verbose=False,
    )

    print("=" * 80)
    print("FEATURE-IRT ANALYSIS")
    print(f"Hyperparameters: l2_weight={l2_weight}, l2_residual={l2_residual}")
    print("=" * 80)

    results = []

    for source_name, source in feature_source_list:
        print(f"\n{'='*60}")
        print(f"Source: {source_name}")
        print(f"{'='*60}")

        result = analyze_single_source(
            source_name, source, all_task_ids, oracle_b, responses,
            l2_weight, l2_residual
        )
        results.append(result)

    # Also analyze grouped source
    print(f"\n{'='*60}")
    print(f"Source: Grouped (All Sources)")
    print(f"{'='*60}")

    feature_sources = [source for _, source in feature_source_list]
    grouped_source = GroupedFeatureSource([
        RegularizedFeatureSource(src) for src in feature_sources
    ])

    result = analyze_single_source(
        "Grouped", grouped_source, all_task_ids, oracle_b, responses,
        l2_weight, l2_residual
    )
    results.append(result)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Source':<20} {'Feat Dim':>10} {'Feat Var%':>10} {'Resid Var%':>12} {'Corr(Total)':>12} {'Corr(Feat)':>12} {'RMSE':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['source']:<20} {r['feature_dim']:>10} {r['feature_var_pct']:>10.1f} {r['residual_var_pct']:>12.1f} {r['corr_total']:>12.6f} {r['corr_feature']:>12.6f} {r['rmse']:>10.4f}")

    return results


def analyze_single_source(
    source_name: str,
    source,
    all_task_ids,
    oracle_b,
    responses,
    l2_weight: float,
    l2_residual: float,
):
    """Analyze a single feature source."""
    # Get feature dimension
    features = source.get_features(all_task_ids)
    print(f"Feature dimension: {features.shape[1]}")

    # Train Feature-IRT
    predictor = FeatureIRTPredictor(
        source=source,
        use_residuals=True,
        init_from_baseline=False,
        l2_weight=l2_weight,
        l2_residual=l2_residual,
        l2_ability=0.01,
        verbose=False,
    )

    predictor.fit(
        task_ids=all_task_ids,
        ground_truth_b=oracle_b,
        responses=responses,
    )

    # Get predictions and components
    predicted_b = predictor.predict(all_task_ids)
    predicted_b_arr = np.array([predicted_b[t] for t in all_task_ids])

    # Get residuals (difficulty latents)
    residuals = predictor._residuals
    residual_arr = np.array([residuals[t] for t in all_task_ids])

    # Get feature contribution: w^T f + bias
    # Total = feature_contrib + residual
    feature_contrib = predicted_b_arr - residual_arr

    # Compute statistics
    print(f"\n--- Prediction Components ---")
    print(f"Total predicted b:  mean={predicted_b_arr.mean():.4f}, std={predicted_b_arr.std():.4f}")
    print(f"Feature contrib:    mean={feature_contrib.mean():.4f}, std={feature_contrib.std():.4f}")
    print(f"Difficulty latent:  mean={residual_arr.mean():.4f}, std={residual_arr.std():.4f}")

    # Variance decomposition
    total_var = np.var(predicted_b_arr)
    feature_var = np.var(feature_contrib)
    residual_var = np.var(residual_arr)
    # Covariance term: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
    cov_term = total_var - feature_var - residual_var

    print(f"\n--- Variance Decomposition ---")
    print(f"Total variance:     {total_var:.4f}")
    print(f"Feature variance:   {feature_var:.4f} ({100*feature_var/total_var:.1f}%)")
    print(f"Residual variance:  {residual_var:.4f} ({100*residual_var/total_var:.1f}%)")
    print(f"2*Cov(feat,resid):  {cov_term:.4f} ({100*cov_term/total_var:.1f}%)")

    # Correlation with Oracle
    corr_total = np.corrcoef(predicted_b_arr, oracle_b)[0, 1]
    corr_feature = np.corrcoef(feature_contrib, oracle_b)[0, 1]
    corr_residual = np.corrcoef(residual_arr, oracle_b)[0, 1]

    print(f"\n--- Correlation with Oracle ---")
    print(f"Total prediction:   r={corr_total:.6f}")
    print(f"Feature contrib:    r={corr_feature:.6f}")
    print(f"Difficulty latent:  r={corr_residual:.6f}")

    # Are predictions exactly equal to Oracle?
    diff = predicted_b_arr - oracle_b
    print(f"\n--- Difference from Oracle ---")
    print(f"Mean absolute diff: {np.abs(diff).mean():.6f}")
    print(f"Max absolute diff:  {np.abs(diff).max():.6f}")
    print(f"RMSE:               {np.sqrt(np.mean(diff**2)):.6f}")

    return {
        "source": source_name,
        "feature_dim": features.shape[1],
        "total_var": total_var,
        "feature_var": feature_var,
        "residual_var": residual_var,
        "feature_var_pct": 100 * feature_var / total_var,
        "residual_var_pct": 100 * residual_var / total_var,
        "corr_total": corr_total,
        "corr_feature": corr_feature,
        "corr_residual": corr_residual,
        "rmse": np.sqrt(np.mean(diff**2)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Feature-IRT predictions")
    parser.add_argument("--l2_weight", type=float, default=0.001, help="L2 regularization on feature weights")
    parser.add_argument("--l2_residual", type=float, default=0.0001, help="L2 regularization on difficulty latents")
    args = parser.parse_args()

    analyze_feature_irt(l2_weight=args.l2_weight, l2_residual=args.l2_residual)
