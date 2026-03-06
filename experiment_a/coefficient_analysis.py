"""Coefficient analysis for LLM Judge Ridge predictor.

Extracts and displays feature coefficients from fitted Ridge models,
producing data for Table 10 (feature rankings) and Figure 3 (source bar chart)
in the paper.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiment_ab_shared.feature_predictor import FeatureBasedPredictor


# Feature source mapping: feature name → information source category.
# Used to group coefficients by source for Figure 3.
FEATURE_SOURCES = {
    # Solution Patch — require the gold solution patch
    "solution_complexity": "Solution Patch",
    "integration_complexity": "Solution Patch",
    # Environment — require shell access to task environment
    "entry_point_clarity": "Environment",
    "fix_localization": "Environment",
    "change_blast_radius": "Environment",
    # Test Patch — derived from test patch diff
    "test_comprehensiveness": "Test Patch",
    "test_assertion_complexity": "Test Patch",
    "test_edge_case_coverage": "Test Patch",
    # Problem Statement — derived from problem statement only
    "solution_hint": "Problem Statement",
    "problem_clarity": "Problem Statement",
    "domain_knowledge_required": "Problem Statement",
    "logical_reasoning_required": "Problem Statement",
    "atypicality": "Problem Statement",
    "verification_difficulty": "Problem Statement",
    "standard_pattern_available": "Problem Statement",
    "error_specificity": "Problem Statement",
    "reproduction_clarity": "Problem Statement",
    "expected_behavior_clarity": "Problem Statement",
    "debugging_complexity": "Problem Statement",
    "codebase_scope": "Problem Statement",
    "information_completeness": "Problem Statement",
    "similar_issue_likelihood": "Problem Statement",
    "backwards_compatibility_risk": "Problem Statement",
}


def extract_llm_coefficients(predictor: FeatureBasedPredictor) -> Dict[str, Any]:
    """Extract coefficients from a fitted FeatureBasedPredictor (Ridge).

    Args:
        predictor: A fitted FeatureBasedPredictor wrapping a Ridge model.

    Returns:
        Dict with keys:
        - feature_names: List of feature names
        - standardized_coef: Coefficients for standardized features
        - unscaled_coef: Coefficients per 1-unit change in raw features
    """
    if not predictor._is_fitted:
        raise RuntimeError("Predictor must be fitted")

    coef = predictor._model.coef_
    feature_names = predictor.source.feature_names
    scale = predictor._scaler.scale_
    unscaled_coef = coef / scale

    return {
        "feature_names": list(feature_names),
        "standardized_coef": coef.copy(),
        "unscaled_coef": unscaled_coef.copy(),
    }


def print_coefficient_table(
    coeffs_by_fold: List[Dict[str, Any]],
) -> None:
    """Print a ranked coefficient table (Table 10 format).

    Args:
        coeffs_by_fold: List of dicts from extract_llm_coefficients, one per fold.
    """
    if not coeffs_by_fold:
        return

    feature_names = coeffs_by_fold[0]["feature_names"]
    all_coefs = np.array([c["standardized_coef"] for c in coeffs_by_fold])
    mean_coef = all_coefs.mean(axis=0)
    std_coef = all_coefs.std(axis=0)

    # Sort by absolute coefficient
    indices = np.argsort(np.abs(mean_coef))[::-1]

    print(f"\n{'Rank':<5} {'Feature':<35} {'Coeff':>8} {'Std':>8} {'Source':<20}")
    print("-" * 80)
    for rank, idx in enumerate(indices, 1):
        name = feature_names[idx]
        source = FEATURE_SOURCES.get(name, "Unknown")
        print(f"{rank:<5} {name:<35} {mean_coef[idx]:>+8.3f} {std_coef[idx]:>8.3f} {source:<20}")

    # Print source-level summary
    print(f"\n{'Source':<25} {'Mean |Coeff|':>12} {'N features':>12}")
    print("-" * 52)
    source_groups: Dict[str, List[float]] = {}
    for name, coef in zip(feature_names, mean_coef):
        source = FEATURE_SOURCES.get(name, "Unknown")
        source_groups.setdefault(source, []).append(abs(coef))

    for source in ["Problem Statement", "Environment", "Test Patch", "Solution Patch"]:
        if source in source_groups:
            vals = source_groups[source]
            print(f"{source:<25} {np.mean(vals):>12.3f} {len(vals):>12}")


def make_llm_coef_extractor():
    """Create a diagnostics extractor for LLM Judge Ridge coefficients.

    Returns a callback suitable for use as a diagnostics_extractor in
    cross_validate_all_predictors(). The callback extracts coefficients from a
    DifficultyPredictorAdapter wrapping a FeatureBasedPredictor.

    Returns:
        Dict mapping predictor name to extractor callback.
    """
    def _extract(predictor, fold_idx):
        inner = getattr(predictor, "_predictor", None)
        if inner is not None and hasattr(inner, "_is_fitted") and inner._is_fitted:
            return extract_llm_coefficients(inner)
        return None

    return {"llm_judge": _extract}


def save_coefficient_bar_chart(
    coeffs_by_fold: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Mean Coefficient Magnitude by Feature Source",
) -> None:
    """Save a bar chart of mean |coefficient| by feature source (Figure 3).

    Args:
        coeffs_by_fold: List of dicts from extract_llm_coefficients, one per fold.
        output_path: Path to save the PNG figure.
        title: Chart title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not coeffs_by_fold:
        return

    feature_names = coeffs_by_fold[0]["feature_names"]
    all_coefs = np.array([c["standardized_coef"] for c in coeffs_by_fold])
    mean_coef = all_coefs.mean(axis=0)

    # Group by source
    sources = ["Problem Statement", "Environment", "Test Patch", "Solution Patch"]
    source_coefs: Dict[str, List[float]] = {s: [] for s in sources}
    for name, coef in zip(feature_names, mean_coef):
        source = FEATURE_SOURCES.get(name, "Unknown")
        if source in source_coefs:
            source_coefs[source].append(abs(coef))

    means = [np.mean(source_coefs[s]) if source_coefs[s] else 0 for s in sources]
    counts = [len(source_coefs[s]) for s in sources]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    x = np.arange(len(sources))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=1)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Feature Source", fontsize=12)
    ax.set_ylabel("Mean |Coefficient|", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBar chart saved to: {output_path}")
