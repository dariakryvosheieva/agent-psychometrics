"""Analyze Grouped Ridge predictor coefficients across all 4 datasets.

This script uses the existing CV infrastructure to fit GroupedRidgePredictor
and extracts LLM judge coefficients to analyze:
1. Which features have highest/lowest coefficients (ranks)
2. Whether feature importance is consistent across datasets
3. What percentage of predictions come from embeddings vs LLM judge (by L2 norm squared)
4. What regularization alphas were selected per source

Usage:
    python -m experiment_a.analyze_grouped_ridge_coefficients
    python -m experiment_a.analyze_grouped_ridge_coefficients --dataset swebench
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from experiment_a.shared.pipeline import run_cross_validation, ExperimentSpec
from experiment_a.shared.cross_validation import CVPredictor

# Import dataset configs
from experiment_a.swebench.config import ExperimentAConfig
from experiment_a.swebench_pro.config import SWEBenchProConfig
from experiment_a.gso.config import GSOConfig
from experiment_a.terminalbench.config import TerminalBenchConfig

ROOT = Path(__file__).resolve().parents[1]

# Dataset specifications (matches run_all_datasets.py structure)
DATASETS = {
    "swebench": {
        "config_class": ExperimentAConfig,
        "display_name": "SWE-bench Verified",
        "spec": ExperimentSpec(
            name="SWE-bench Verified",
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a" / "irt_splits",
        ),
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "swebench_unified" / "llm_judge_features.csv",
    },
    "swebench_pro": {
        "config_class": SWEBenchProConfig,
        "display_name": "SWE-bench Pro",
        "spec": ExperimentSpec(
            name="SWE-bench Pro",
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a_swebench_pro" / "irt_splits",
        ),
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "swebench_pro_unified" / "llm_judge_features.csv",
    },
    "gso": {
        "config_class": GSOConfig,
        "display_name": "GSO",
        "spec": ExperimentSpec(
            name="GSO",
            is_binomial=False,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a_gso" / "irt_splits",
        ),
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "gso_unified" / "llm_judge_features.csv",
    },
    "terminalbench": {
        "config_class": TerminalBenchConfig,
        "display_name": "TerminalBench",
        "spec": ExperimentSpec(
            name="TerminalBench",
            is_binomial=True,
            irt_cache_dir=ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits",
        ),
        "unified_llm_path": ROOT / "chris_output" / "llm_judge_features" / "terminalbench_unified" / "llm_judge_features.csv",
    },
}


def extract_grouped_ridge_diagnostics(predictor: CVPredictor, fold_idx: int) -> Dict[str, Any]:
    """Extract diagnostics from a fitted GroupedRidgePredictor."""
    inner = predictor._predictor  # Access inner GroupedRidgePredictor via DifficultyPredictorAdapter
    return inner.get_detailed_diagnostics()


def run_analysis_for_dataset(dataset_name: str, k_folds: int = 5) -> Dict[str, Any]:
    """Run CV with diagnostics extraction for one dataset."""
    dataset_info = DATASETS[dataset_name]
    config_class = dataset_info["config_class"]
    spec = dataset_info["spec"]
    unified_llm_path = dataset_info["unified_llm_path"]

    # Create config with unified LLM features
    config = config_class(llm_judge_features_path=unified_llm_path)

    # Run CV with diagnostics extraction for grouped_ridge
    results = run_cross_validation(
        config=config,
        spec=spec,
        root=ROOT,
        k=k_folds,
        diagnostics_extractors={"grouped_ridge": extract_grouped_ridge_diagnostics},
    )

    # Extract and aggregate diagnostics
    grouped_result = results["cv_results"]["grouped_ridge"]
    fold_diagnostics = grouped_result.get("fold_diagnostics", [])

    if not fold_diagnostics:
        raise ValueError(f"No diagnostics extracted for {dataset_name}")

    # Get feature names and aggregate coefficients
    feature_names = fold_diagnostics[0]["coef_by_source"]["LLM Judge"]["feature_names"]
    all_coefs = []
    for diag in fold_diagnostics:
        named_coefs = diag["coef_by_source"]["LLM Judge"]["named_coefficients"]
        all_coefs.append([named_coefs[fn] for fn in feature_names])
    all_coefs = np.array(all_coefs)

    # Extract contributions and alphas
    emb_contribs = [d["coef_by_source"]["Embedding"]["contribution_pct"] for d in fold_diagnostics]
    llm_contribs = [d["coef_by_source"]["LLM Judge"]["contribution_pct"] for d in fold_diagnostics]
    emb_alphas = [d["selected_alphas"]["Embedding"] for d in fold_diagnostics]
    llm_alphas = [d["selected_alphas"]["LLM Judge"] for d in fold_diagnostics]

    return {
        "dataset_name": dataset_name,
        "display_name": dataset_info["display_name"],
        "mean_auc": grouped_result["mean_auc"],
        "std_auc": grouped_result["std_auc"],
        "feature_names": feature_names,
        "mean_coef": np.mean(all_coefs, axis=0).tolist(),
        "std_coef": np.std(all_coefs, axis=0).tolist(),
        "mean_embedding_contribution_pct": float(np.mean(emb_contribs)),
        "mean_llm_contribution_pct": float(np.mean(llm_contribs)),
        "std_embedding_contribution_pct": float(np.std(emb_contribs)),
        "std_llm_contribution_pct": float(np.std(llm_contribs)),
        "mean_embedding_alpha": float(np.mean(emb_alphas)),
        "mean_llm_alpha": float(np.mean(llm_alphas)),
    }


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted results for all datasets."""
    for dataset_name, result in results.items():
        print(f"\n{'='*75}")
        print(f"{result['display_name']} - Grouped Ridge Coefficient Analysis")
        print(f"{'='*75}")
        print(f"Mean AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")

        # Coefficient table
        print(f"\nLLM Judge Coefficients (ranked by |coefficient|):")
        print(f"{'Feature':<35} {'Coef':<15} {'|Coef|':>8} {'Rank':>6}")
        print("-" * 70)

        feature_names = result["feature_names"]
        mean_coef = np.array(result["mean_coef"])
        std_coef = np.array(result["std_coef"])
        indices = np.argsort(np.abs(mean_coef))[::-1]

        for rank, idx in enumerate(indices, 1):
            name = feature_names[idx]
            coef = mean_coef[idx]
            std = std_coef[idx]
            print(f"{name:<35} {coef:+.3f}±{std:.3f}     {abs(coef):>8.3f} {rank:>6}")

        # Contribution and alpha
        print(f"\nContribution (L2 norm squared):")
        print(f"  Embedding: {result['mean_embedding_contribution_pct']:.1f}% ± {result['std_embedding_contribution_pct']:.1f}%")
        print(f"  LLM Judge: {result['mean_llm_contribution_pct']:.1f}% ± {result['std_llm_contribution_pct']:.1f}%")
        print(f"\nSelected Alphas: Embedding={result['mean_embedding_alpha']:.0f}, LLM Judge={result['mean_llm_alpha']:.1f}")


def print_cross_dataset_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """Print cross-dataset feature importance comparison."""
    print(f"\n{'='*85}")
    print("CROSS-DATASET COEFFICIENT COMPARISON (Grouped Ridge)")
    print(f"{'='*85}")

    all_features = results[list(results.keys())[0]]["feature_names"]
    dataset_abbrevs = {"swebench": "SWE", "swebench_pro": "Pro", "gso": "GSO", "terminalbench": "Term"}

    # Build rankings
    rankings = {}
    for ds, result in results.items():
        mean_coef = np.array(result["mean_coef"])
        indices = np.argsort(np.abs(mean_coef))[::-1]
        rankings[ds] = {result["feature_names"][idx]: rank + 1 for rank, idx in enumerate(indices)}

    # Average ranks
    avg_ranks = {f: np.mean([rankings[ds].get(f, len(all_features)) for ds in results]) for f in all_features}
    sorted_features = sorted(all_features, key=lambda f: avg_ranks[f])

    # Print ranking table
    print(f"\nFeature Importance Ranking (by |coefficient|):")
    header = f"{'Feature':<32}" + "".join(f" {dataset_abbrevs.get(ds, ds[:4]):>6}" for ds in results) + f" {'Avg':>6}"
    print(header)
    print("-" * len(header))

    for feature in sorted_features:
        row = f"{feature:<32}" + "".join(f" {rankings[ds].get(feature, '-'):>6}" for ds in results) + f" {avg_ranks[feature]:>6.1f}"
        print(row)

    # Contribution summary
    print(f"\nContribution Summary (L2 norm squared):")
    print(f"{'':20}" + "".join(f" {dataset_abbrevs.get(ds, ds[:4]):>8}" for ds in results))
    print("-" * (20 + 9 * len(results)))
    print(f"{'Embedding %':<20}" + "".join(f" {r['mean_embedding_contribution_pct']:>7.1f}%" for r in results.values()))
    print(f"{'LLM Judge %':<20}" + "".join(f" {r['mean_llm_contribution_pct']:>7.1f}%" for r in results.values()))

    print(f"\nSelected Alphas (mean):")
    print(f"{'Embedding α':<20}" + "".join(f" {r['mean_embedding_alpha']:>8.0f}" for r in results.values()))
    print(f"{'LLM Judge α':<20}" + "".join(f" {r['mean_llm_alpha']:>8.1f}" for r in results.values()))


def main():
    parser = argparse.ArgumentParser(description="Analyze Grouped Ridge coefficients")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), help="Single dataset (default: all)")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output_path", type=str, default=str(ROOT / "chris_output" / "grouped_ridge_coefficient_analysis.json"))
    args = parser.parse_args()

    datasets_to_run = [args.dataset] if args.dataset else list(DATASETS.keys())

    results = {}
    for ds in datasets_to_run:
        results[ds] = run_analysis_for_dataset(ds, k_folds=args.k_folds)

    print_results(results)
    if len(results) > 1:
        print_cross_dataset_comparison(results)

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
