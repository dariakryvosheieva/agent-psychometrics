"""Quick evaluation pipeline for LLM judge features.

This module enables rapid validation of new feature prompts before committing
to full dataset extraction.

Workflow:
1. Sample ~100 tasks (stratified by difficulty tercile)
2. Extract features via batched API calls (or use existing)
3. Compute individual correlations with IRT difficulty
4. Run Lasso to identify predictive features
5. Detect redundant feature pairs (|r| > 0.9)
6. Report: which features to keep/drop

Usage:
    python -m experiment_ab_shared.llm_judge.quick_eval \
        --dataset swebench \
        --features-csv path/to/features.csv \
        --irt-items clean_data/swebench_verified_20251120_full/1d/items.csv
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


@dataclass
class QuickEvalResult:
    """Results from quick feature evaluation."""

    n_tasks: int
    n_features: int
    correlations: pd.DataFrame
    lasso_coefs: Dict[str, float]
    redundant_pairs: List[Tuple[str, str, float]]
    recommended_features: List[str]
    dropped_features: Dict[str, str]  # feature -> reason

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 80)
        print("QUICK EVALUATION REPORT")
        print("=" * 80)

        print(f"\nDataset: {self.n_tasks} tasks, {self.n_features} features")

        # Correlation summary
        print("\n" + "-" * 80)
        print("INDIVIDUAL CORRELATIONS WITH DIFFICULTY")
        print("-" * 80)
        print(f"{'Feature':<40} {'Pearson r':>12} {'p-value':>12} {'Lasso coef':>12}")
        print("-" * 80)

        for _, row in self.correlations.iterrows():
            feature = row["feature"]
            pearson_r = row["pearson_r"]
            pearson_p = row["pearson_p"]
            lasso_coef = self.lasso_coefs.get(feature, 0.0)

            # Significance marker
            if pearson_p < 0.001:
                sig = "***"
            elif pearson_p < 0.01:
                sig = "**"
            elif pearson_p < 0.05:
                sig = "*"
            else:
                sig = ""

            lasso_str = f"{lasso_coef:>10.4f}" if abs(lasso_coef) > 0.001 else "         -"
            print(f"{feature:<40} {pearson_r:>9.3f}{sig:<3} {pearson_p:>12.4f} {lasso_str}")

        print("-" * 80)
        print("Significance: * p<0.05, ** p<0.01, *** p<0.001")

        # Redundant pairs
        if self.redundant_pairs:
            print("\n" + "-" * 80)
            print("REDUNDANT FEATURE PAIRS (|r| > 0.9)")
            print("-" * 80)
            for f1, f2, r in self.redundant_pairs:
                print(f"  {f1} <-> {f2}: r = {r:.3f}")

        # Recommendations
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)
        print(f"\nKEEP ({len(self.recommended_features)} features):")
        for f in self.recommended_features:
            print(f"  + {f}")

        if self.dropped_features:
            print(f"\nDROP ({len(self.dropped_features)} features):")
            for f, reason in self.dropped_features.items():
                print(f"  - {f}: {reason}")

        print("\n" + "=" * 80)

    def to_json(self, path: Path) -> None:
        """Save results to JSON."""
        data = {
            "n_tasks": self.n_tasks,
            "n_features": self.n_features,
            "correlations": self.correlations.to_dict(orient="records"),
            "lasso_coefs": self.lasso_coefs,
            "redundant_pairs": [
                {"feature1": f1, "feature2": f2, "correlation": r}
                for f1, f2, r in self.redundant_pairs
            ],
            "recommended_features": self.recommended_features,
            "dropped_features": self.dropped_features,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {path}")


def sample_tasks(
    task_ids: List[str],
    difficulties: pd.Series,
    n_sample: int = 100,
    seed: int = 42,
    stratify_by_difficulty: bool = True,
) -> List[str]:
    """Sample tasks for quick evaluation.

    Args:
        task_ids: List of all task IDs
        difficulties: Series mapping task_id -> difficulty value
        n_sample: Number of tasks to sample
        seed: Random seed for reproducibility
        stratify_by_difficulty: If True, sample evenly across difficulty terciles

    Returns:
        List of sampled task IDs
    """
    rng = np.random.default_rng(seed)

    # Filter to tasks with difficulty values
    valid_task_ids = [t for t in task_ids if t in difficulties.index]

    if len(valid_task_ids) <= n_sample:
        return valid_task_ids

    if stratify_by_difficulty:
        # Split into terciles
        task_diffs = difficulties.loc[valid_task_ids]
        tercile_bounds = task_diffs.quantile([0.33, 0.67]).values

        easy = [t for t in valid_task_ids if task_diffs[t] < tercile_bounds[0]]
        medium = [
            t
            for t in valid_task_ids
            if tercile_bounds[0] <= task_diffs[t] < tercile_bounds[1]
        ]
        hard = [t for t in valid_task_ids if task_diffs[t] >= tercile_bounds[1]]

        # Sample from each tercile
        n_per_tercile = n_sample // 3
        remainder = n_sample % 3

        sampled = []
        for i, tercile in enumerate([easy, medium, hard]):
            n = n_per_tercile + (1 if i < remainder else 0)
            n = min(n, len(tercile))
            sampled.extend(rng.choice(tercile, size=n, replace=False).tolist())

        return sampled
    else:
        return rng.choice(valid_task_ids, size=n_sample, replace=False).tolist()


def compute_correlations(
    features_df: pd.DataFrame,
    difficulty: pd.Series,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations with p-values.

    Args:
        features_df: DataFrame with features (index = task_id)
        difficulty: Series with difficulty values (index = task_id)
        feature_cols: List of feature column names to analyze

    Returns:
        DataFrame with correlation results sorted by |pearson_r|
    """
    # Align indices
    common_ids = features_df.index.intersection(difficulty.index)
    if len(common_ids) == 0:
        raise ValueError("No common task IDs between features and difficulty")

    features_aligned = features_df.loc[common_ids]
    difficulty_aligned = difficulty.loc[common_ids]

    results = []
    for feature in feature_cols:
        if feature not in features_aligned.columns:
            continue

        x = features_aligned[feature].values
        y = difficulty_aligned.values

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

        results.append(
            {
                "feature": feature,
                "n": len(x_clean),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("pearson_p")
    return df


def compute_pairwise_correlations(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.9,
) -> List[Tuple[str, str, float]]:
    """Find pairs of features with |correlation| > threshold.

    Args:
        features_df: DataFrame with features
        feature_cols: List of feature column names
        threshold: Correlation threshold for redundancy

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    redundant_pairs = []

    # Filter to available columns
    available_cols = [c for c in feature_cols if c in features_df.columns]

    if len(available_cols) < 2:
        return redundant_pairs

    # Compute correlation matrix
    corr_matrix = features_df[available_cols].corr()

    # Find pairs above threshold
    for i, f1 in enumerate(available_cols):
        for f2 in available_cols[i + 1 :]:
            r = corr_matrix.loc[f1, f2]
            if abs(r) > threshold:
                redundant_pairs.append((f1, f2, r))

    return sorted(redundant_pairs, key=lambda x: -abs(x[2]))


def run_lasso_selection(
    features_df: pd.DataFrame,
    difficulty: pd.Series,
    feature_cols: List[str],
    cv: int = 5,
) -> Dict[str, float]:
    """Run LassoCV to identify predictive features.

    Args:
        features_df: DataFrame with features
        difficulty: Series with difficulty values
        feature_cols: List of feature column names
        cv: Number of cross-validation folds

    Returns:
        Dict mapping feature names to Lasso coefficients
    """
    # Align indices
    common_ids = features_df.index.intersection(difficulty.index)
    features_aligned = features_df.loc[common_ids]
    difficulty_aligned = difficulty.loc[common_ids]

    # Filter to available columns
    available_cols = [c for c in feature_cols if c in features_aligned.columns]

    if len(available_cols) == 0:
        return {}

    X = features_aligned[available_cols].values
    y = difficulty_aligned.values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run LassoCV
    lasso = LassoCV(cv=cv, random_state=42)
    lasso.fit(X_scaled, y)

    return dict(zip(available_cols, lasso.coef_))


def evaluate_features(
    features_df: pd.DataFrame,
    irt_items_path: Path,
    feature_cols: Optional[List[str]] = None,
    correlation_threshold: float = 0.05,
    redundancy_threshold: float = 0.9,
    lasso_threshold: float = 0.001,
) -> QuickEvalResult:
    """Evaluate features and provide recommendations.

    Args:
        features_df: DataFrame with features (index = task_id)
        irt_items_path: Path to IRT items.csv with difficulty column 'b'
        feature_cols: List of feature columns to evaluate (auto-detect if None)
        correlation_threshold: p-value threshold for significant correlation
        redundancy_threshold: Correlation threshold for redundant pairs
        lasso_threshold: Coefficient threshold for Lasso selection

    Returns:
        QuickEvalResult with recommendations
    """
    # Load IRT items
    irt_df = pd.read_csv(irt_items_path, index_col=0)
    irt_df.index = irt_df.index.str.replace(r"^instance_", "", regex=True)
    difficulty = irt_df["b"]

    # Clean feature index
    features_df = features_df.copy()
    features_df.index = features_df.index.astype(str).str.replace(
        r"^instance_", "", regex=True
    )

    # Auto-detect feature columns if not provided
    if feature_cols is None:
        meta_cols = [c for c in features_df.columns if c.startswith("_") or c == "reasoning"]
        feature_cols = [
            c
            for c in features_df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(features_df[c])
        ]

    # Compute correlations
    corr_df = compute_correlations(features_df, difficulty, feature_cols)

    # Run Lasso
    lasso_coefs = run_lasso_selection(features_df, difficulty, feature_cols)

    # Find redundant pairs
    redundant_pairs = compute_pairwise_correlations(
        features_df, feature_cols, threshold=redundancy_threshold
    )

    # Build recommendations
    recommended = []
    dropped = {}

    # Track which features to drop due to redundancy
    redundant_drop = set()
    for f1, f2, r in redundant_pairs:
        # Drop the one with lower absolute correlation with difficulty
        r1 = abs(corr_df[corr_df["feature"] == f1]["pearson_r"].values[0]) if f1 in corr_df["feature"].values else 0
        r2 = abs(corr_df[corr_df["feature"] == f2]["pearson_r"].values[0]) if f2 in corr_df["feature"].values else 0

        if r1 >= r2:
            redundant_drop.add(f2)
            dropped[f2] = f"Redundant with {f1} (r={r:.3f})"
        else:
            redundant_drop.add(f1)
            dropped[f1] = f"Redundant with {f2} (r={r:.3f})"

    for _, row in corr_df.iterrows():
        feature = row["feature"]

        if feature in redundant_drop:
            continue

        # Check significance
        if row["pearson_p"] >= correlation_threshold:
            dropped[feature] = f"Not significant (p={row['pearson_p']:.4f})"
            continue

        # Check Lasso selection
        lasso_coef = lasso_coefs.get(feature, 0.0)
        if abs(lasso_coef) < lasso_threshold and row["pearson_p"] > 0.01:
            dropped[feature] = f"Lasso dropped (coef={lasso_coef:.4f})"
            continue

        recommended.append(feature)

    return QuickEvalResult(
        n_tasks=len(features_df.index.intersection(difficulty.index)),
        n_features=len(feature_cols),
        correlations=corr_df,
        lasso_coefs=lasso_coefs,
        redundant_pairs=redundant_pairs,
        recommended_features=recommended,
        dropped_features=dropped,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quick evaluation of LLM judge features"
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Path to features CSV file",
    )
    parser.add_argument(
        "--irt-items",
        type=Path,
        required=True,
        help="Path to IRT items.csv with difficulty column 'b'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results JSON",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.05,
        help="p-value threshold for significant correlation (default: 0.05)",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.9,
        help="Correlation threshold for redundant pairs (default: 0.9)",
    )

    args = parser.parse_args()

    # Load features
    features_df = pd.read_csv(args.features_csv)

    # Find task ID column
    task_id_col = None
    for col in ["_instance_id", "instance_id", "_task_id", "task_id"]:
        if col in features_df.columns:
            task_id_col = col
            break

    if task_id_col is None:
        task_id_col = features_df.columns[0]

    features_df = features_df.set_index(task_id_col)

    # Evaluate
    result = evaluate_features(
        features_df,
        args.irt_items,
        correlation_threshold=args.correlation_threshold,
        redundancy_threshold=args.redundancy_threshold,
    )

    # Print report
    result.print_report()

    # Save if output path provided
    if args.output:
        result.to_json(args.output)


if __name__ == "__main__":
    main()
