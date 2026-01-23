"""Analyze correlations between LLM Judge features and oracle IRT task difficulties.

This module provides functions to:
1. Load LLM Judge features and IRT difficulties
2. Compute Pearson and Spearman correlations with p-values
3. Run LassoCV feature selection
4. Output formatted tables with significance markers

Usage:
    python -m experiment_ab_shared.llm_judge.analyze_feature_correlations \
        --features-csv chris_output/experiment_a/llm_judge_features/llm_judge_features.csv \
        --irt-items clean_data/swebench_verified_20251120_full/1d/items.csv \
        --dataset swebench
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def load_features_and_difficulties(
    features_path: Path,
    irt_items_path: Path,
    task_id_col: str = "_instance_id",
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and merge LLM Judge features with IRT difficulties.

    Args:
        features_path: Path to LLM Judge features CSV
        irt_items_path: Path to IRT items.csv with difficulty column 'b'
        task_id_col: Column name for task ID in features CSV

    Returns:
        Tuple of (merged DataFrame, list of feature column names)
    """
    # Load features
    features_df = pd.read_csv(features_path)

    # Auto-detect task ID column
    for col in [task_id_col, "_instance_id", "instance_id", "_task_id", "task_id"]:
        if col in features_df.columns:
            task_id_col = col
            break
    else:
        # Use first column as task ID
        task_id_col = features_df.columns[0]

    # Clean task IDs (strip 'instance_' prefix if present)
    features_df[task_id_col] = features_df[task_id_col].str.replace(
        r"^instance_", "", regex=True
    )

    # Load IRT items
    irt_df = pd.read_csv(irt_items_path, index_col=0)
    irt_df.index = irt_df.index.str.replace(r"^instance_", "", regex=True)

    # Identify feature columns (numeric, not metadata)
    meta_cols = [c for c in features_df.columns if c.startswith("_") or c == "reasoning"]
    feature_cols = [
        c
        for c in features_df.columns
        if c not in meta_cols
        and c != task_id_col
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]

    # Merge
    features_df = features_df.set_index(task_id_col)
    merged = features_df.join(irt_df[["b"]], how="inner")

    if len(merged) == 0:
        raise ValueError(
            f"No tasks matched between features ({len(features_df)}) "
            f"and IRT items ({len(irt_df)}). "
            "Check task ID formats."
        )

    print(f"Merged {len(merged)} tasks (features: {len(features_df)}, IRT: {len(irt_df)})")

    return merged, feature_cols


def compute_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "b",
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations with p-values.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name (default: 'b' for IRT difficulty)

    Returns:
        DataFrame with correlation results
    """
    results = []

    for feature in feature_cols:
        x = df[feature].values
        y = df[target_col].values

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 3:
            continue

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

        # Significance markers
        def sig_marker(p: float) -> str:
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            return ""

        results.append(
            {
                "feature": feature,
                "n": len(x_clean),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "pearson_sig": sig_marker(pearson_p),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "spearman_sig": sig_marker(spearman_p),
            }
        )

    return pd.DataFrame(results).sort_values("pearson_p")


def run_lasso_feature_selection(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "b",
    cv: int = 5,
) -> Dict[str, float]:
    """Run LassoCV to identify predictive features.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        cv: Number of cross-validation folds

    Returns:
        Dict mapping feature names to Lasso coefficients
    """
    X = df[feature_cols].values
    y = df[target_col].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run LassoCV
    lasso = LassoCV(cv=cv, random_state=42)
    lasso.fit(X_scaled, y)

    # Get coefficients
    coefs = dict(zip(feature_cols, lasso.coef_))

    return coefs


def print_correlation_table(
    corr_df: pd.DataFrame,
    lasso_coefs: Optional[Dict[str, float]] = None,
    title: str = "Feature Correlations with IRT Difficulty",
) -> None:
    """Print a formatted correlation table.

    Args:
        corr_df: DataFrame from compute_correlations
        lasso_coefs: Optional dict of Lasso coefficients
        title: Table title
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # Header
    if lasso_coefs:
        print(
            f"{'Feature':<35} {'n':>5} {'Pearson':>12} {'Spearman':>12} {'Lasso':>10}"
        )
        print("-" * 80)
    else:
        print(f"{'Feature':<35} {'n':>5} {'Pearson':>12} {'Spearman':>12}")
        print("-" * 70)

    for _, row in corr_df.iterrows():
        pearson_str = f"{row['pearson_r']:>6.3f}{row['pearson_sig']:<3}"
        spearman_str = f"{row['spearman_r']:>6.3f}{row['spearman_sig']:<3}"

        if lasso_coefs:
            lasso_val = lasso_coefs.get(row["feature"], 0.0)
            lasso_str = f"{lasso_val:>8.4f}" if abs(lasso_val) > 0.001 else "        -"
            print(
                f"{row['feature']:<35} {row['n']:>5} {pearson_str:>12} "
                f"{spearman_str:>12} {lasso_str:>10}"
            )
        else:
            print(
                f"{row['feature']:<35} {row['n']:>5} {pearson_str:>12} {spearman_str:>12}"
            )

    print("-" * 80)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")

    # Summary
    sig_features = corr_df[corr_df["pearson_p"] < 0.05]
    print(f"\nSignificant features (p<0.05): {len(sig_features)} / {len(corr_df)}")

    if lasso_coefs:
        selected = [f for f, c in lasso_coefs.items() if abs(c) > 0.001]
        print(f"Lasso-selected features: {len(selected)} / {len(lasso_coefs)}")


def analyze_features(
    features_path: Path,
    irt_items_path: Path,
    output_path: Optional[Path] = None,
    dataset_name: str = "unknown",
) -> Dict[str, Any]:
    """Run full feature correlation analysis.

    Args:
        features_path: Path to LLM Judge features CSV
        irt_items_path: Path to IRT items.csv
        output_path: Optional path to save results JSON
        dataset_name: Name of the dataset for display

    Returns:
        Dict with analysis results
    """
    # Load data
    df, feature_cols = load_features_and_difficulties(features_path, irt_items_path)

    # Compute correlations
    corr_df = compute_correlations(df, feature_cols)

    # Run Lasso
    lasso_coefs = run_lasso_feature_selection(df, feature_cols)

    # Print results
    print_correlation_table(
        corr_df,
        lasso_coefs,
        title=f"Feature Correlations with IRT Difficulty ({dataset_name})",
    )

    # Prepare results
    results = {
        "dataset": dataset_name,
        "n_tasks": len(df),
        "n_features": len(feature_cols),
        "correlations": corr_df.to_dict(orient="records"),
        "lasso_coefs": lasso_coefs,
        "significant_features": corr_df[corr_df["pearson_p"] < 0.05][
            "feature"
        ].tolist(),
        "lasso_selected_features": [
            f for f, c in lasso_coefs.items() if abs(c) > 0.001
        ],
    }

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between LLM Judge features and IRT difficulties"
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to LLM Judge features CSV",
    )
    parser.add_argument(
        "--irt-items",
        type=Path,
        required=True,
        help="Path to IRT items.csv with difficulty column 'b'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="unknown",
        help="Dataset name for display (e.g., 'swebench', 'terminalbench')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results JSON",
    )

    args = parser.parse_args()

    analyze_features(
        features_path=args.features_csv,
        irt_items_path=args.irt_items,
        output_path=args.output,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
