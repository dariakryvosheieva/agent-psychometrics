"""Feature quality metrics for iterative prompt refinement.

Provides utilities to evaluate feature quality:
- Entropy: Higher entropy = feature uses more of its scale = better discrimination
- Pairwise correlation: Highly correlated features are redundant
- Correlation with difficulty: Primary quality signal
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_entropy(values: np.ndarray, normalize: bool = True) -> float:
    """Compute entropy of discrete feature values.

    Higher entropy means the feature uses more of its scale, which is good
    for discrimination. A feature that only outputs {1, 2} has lower entropy
    than one that outputs {1, 2, 3, 4, 5}.

    Args:
        values: Array of discrete integer values
        normalize: If True, normalize by max possible entropy (log2 of unique values)

    Returns:
        Entropy in bits. If normalize=True, returns value in [0, 1].
    """
    values = np.asarray(values).flatten()
    values = values[~np.isnan(values)]  # Remove NaN values

    if len(values) == 0:
        return 0.0

    # Count occurrences of each unique value
    unique, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    if normalize and len(unique) > 1:
        max_entropy = np.log2(len(unique))
        return entropy / max_entropy

    return entropy


def compute_entropy_for_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """Compute entropy for multiple features.

    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names (if None, use all numeric columns)
        normalize: If True, normalize entropies to [0, 1]

    Returns:
        Dict mapping feature name to entropy value
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return {
        col: compute_entropy(df[col].values, normalize=normalize)
        for col in feature_cols
        if col in df.columns
    }


def compute_pairwise_correlations(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for feature columns.

    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names (if None, use all numeric columns)

    Returns:
        Correlation matrix as DataFrame
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_df = df[feature_cols].dropna()
    return feature_df.corr(method='pearson')


def find_redundant_features(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.9,
) -> List[Tuple[str, str, float]]:
    """Find pairs of features with correlation above threshold.

    Args:
        corr_matrix: Correlation matrix from compute_pairwise_correlations
        threshold: Correlation threshold for redundancy

    Returns:
        List of (feature1, feature2, correlation) tuples for redundant pairs
    """
    redundant = []
    features = corr_matrix.columns.tolist()

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if i < j:  # Only upper triangle
                corr = corr_matrix.loc[f1, f2]
                if abs(corr) >= threshold:
                    redundant.append((f1, f2, corr))

    return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)


def compute_difficulty_correlations(
    df: pd.DataFrame,
    difficulty_col: str,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute correlation of each feature with ground truth difficulty.

    Args:
        df: DataFrame with feature columns and difficulty column
        difficulty_col: Name of the difficulty column
        feature_cols: List of feature column names (if None, use all numeric except difficulty)

    Returns:
        Dict mapping feature name to {"pearson_r": r, "p_value": p, "significant": bool}
    """
    if feature_cols is None:
        feature_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col != difficulty_col
        ]

    results = {}
    for col in feature_cols:
        if col not in df.columns:
            continue

        # Drop rows with NaN in either column
        valid = df[[col, difficulty_col]].dropna()
        if len(valid) < 3:
            continue

        r, p = stats.pearsonr(valid[col], valid[difficulty_col])
        results[col] = {
            "pearson_r": r,
            "p_value": p,
            "significant": p < 0.05,
            "abs_r": abs(r),
        }

    return results


def compute_feature_quality_summary(
    df: pd.DataFrame,
    difficulty_col: str,
    feature_cols: Optional[List[str]] = None,
    entropy_threshold: float = 1.0,
    redundancy_threshold: float = 0.9,
) -> Dict:
    """Compute comprehensive feature quality summary.

    Args:
        df: DataFrame with features and difficulty
        difficulty_col: Name of difficulty column
        feature_cols: Feature columns to analyze
        entropy_threshold: Minimum normalized entropy for "good" feature
        redundancy_threshold: Correlation threshold for redundancy

    Returns:
        Dict with:
        - entropies: Dict of feature -> entropy
        - difficulty_correlations: Dict of feature -> correlation stats
        - redundant_pairs: List of highly correlated feature pairs
        - low_entropy_features: Features with entropy below threshold
        - best_features: Features ranked by |correlation| with difficulty
    """
    if feature_cols is None:
        feature_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col != difficulty_col
        ]

    # Compute metrics
    entropies = compute_entropy_for_features(df, feature_cols, normalize=True)
    correlations = compute_difficulty_correlations(df, difficulty_col, feature_cols)
    corr_matrix = compute_pairwise_correlations(df, feature_cols)
    redundant_pairs = find_redundant_features(corr_matrix, redundancy_threshold)

    # Identify low-entropy features
    low_entropy = [
        (f, e) for f, e in entropies.items()
        if e < entropy_threshold
    ]

    # Rank features by correlation with difficulty
    best_features = sorted(
        [(f, c["pearson_r"], c["p_value"]) for f, c in correlations.items()],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "entropies": entropies,
        "difficulty_correlations": correlations,
        "correlation_matrix": corr_matrix,
        "redundant_pairs": redundant_pairs,
        "low_entropy_features": low_entropy,
        "best_features": best_features,
        "n_tasks": len(df),
        "n_features": len(feature_cols),
    }


def format_quality_report(summary: Dict) -> str:
    """Format feature quality summary as human-readable report.

    Args:
        summary: Output from compute_feature_quality_summary

    Returns:
        Formatted string report
    """
    lines = [
        f"Feature Quality Report (n={summary['n_tasks']} tasks, {summary['n_features']} features)",
        "=" * 70,
        "",
        "Top Features by Correlation with Difficulty:",
        "-" * 40,
    ]

    for feature, r, p in summary["best_features"][:10]:
        sig = "*" if p < 0.05 else ""
        lines.append(f"  {feature:35s} r={r:+.3f} (p={p:.3f}){sig}")

    lines.extend([
        "",
        "Feature Entropies (normalized, higher=better):",
        "-" * 40,
    ])

    for feature, entropy in sorted(
        summary["entropies"].items(), key=lambda x: x[1], reverse=True
    ):
        status = "LOW" if entropy < 0.5 else ""
        lines.append(f"  {feature:35s} H={entropy:.3f} {status}")

    if summary["redundant_pairs"]:
        lines.extend([
            "",
            f"Redundant Feature Pairs (r > {0.9}):",
            "-" * 40,
        ])
        for f1, f2, r in summary["redundant_pairs"]:
            lines.append(f"  {f1} <-> {f2}: r={r:.3f}")

    if summary["low_entropy_features"]:
        lines.extend([
            "",
            "Low Entropy Features (consider removing/modifying):",
            "-" * 40,
        ])
        for feature, entropy in summary["low_entropy_features"]:
            lines.append(f"  {feature}: H={entropy:.3f}")

    return "\n".join(lines)