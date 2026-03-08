"""Run information level ablation across all datasets using v8 features.

For each dataset, progressively adds features from higher info levels
(Problem → +Auditor → +Test → +Solution) and measures LLM judge AUC.
Feature count is held constant at 15 via Ridge coefficient-based selection.

Uses v8 features which carry natural info levels (no information leakage):
each feature was extracted seeing only the task data appropriate to its level.

Usage:
    python -m experiment_new_tasks.run_information_ablation
    python -m experiment_new_tasks.run_information_ablation --rebuild_csvs
    python -m experiment_new_tasks.run_information_ablation --datasets swebench_verified gso
    python -m experiment_new_tasks.run_information_ablation --sequential
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_new_tasks.config import ExperimentAConfig, DATASET_DEFAULTS
from experiment_new_tasks.pipeline import cross_validate_all_predictors
from llm_judge_feature_extraction.feature_registry import get_features_by_level
from llm_judge_feature_extraction.prompt_config import InfoLevel


ROOT = Path(__file__).resolve().parents[1]

ALL_DATASETS = ["swebench_verified", "gso", "terminalbench", "swebench_pro"]

ABLATION_OUTPUT_BASE = Path("llm_judge_features/information_ablation")

# Cumulative info levels for the ablation, in order.
# Each entry: (csv_filename, list of InfoLevels included, display_name)
INFO_LEVELS = [
    ("1_problem_15.csv", [InfoLevel.PROBLEM], "Problem"),
    ("2_problem_auditor_15.csv", [InfoLevel.PROBLEM, InfoLevel.ENVIRONMENT], "+ Auditor"),
    ("3_problem_auditor_test_15.csv", [InfoLevel.PROBLEM, InfoLevel.ENVIRONMENT, InfoLevel.TEST], "+ Test"),
    ("4_full_15.csv", [InfoLevel.PROBLEM, InfoLevel.ENVIRONMENT, InfoLevel.TEST, InfoLevel.SOLUTION], "+ Solution (Full)"),
]


def get_feature_names_for_levels(levels: List[InfoLevel]) -> List[str]:
    """Get all feature names for the given info levels, from the registry."""
    names = []
    for level in levels:
        names.extend(f.name for f in get_features_by_level(level))
    return names


def select_top_features(
    df: pd.DataFrame, available_features: List[str], n: int = 15, target_col: str = "b"
) -> List[str]:
    """Select top N features by Ridge coefficient magnitude."""
    X = df[available_features].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    coef_abs = np.abs(ridge.coef_)
    top_indices = np.argsort(coef_abs)[::-1][:n]
    return [available_features[i] for i in top_indices]


def build_ablation_csvs(dataset: str) -> Dict[str, Path]:
    """Build top-15 ablation CSVs for a dataset. Returns {level_filename: output_path}."""
    # Load v8 features (all 28 features at natural info levels)
    v8_path = ROOT / f"llm_judge_features/information_ablation/source/{dataset}.csv"
    features_df = pd.read_csv(v8_path)

    # Load IRT difficulties as regression target
    items_path = ROOT / f"data/{dataset}/irt/1d_1pl/items.csv"
    items = pd.read_csv(items_path, index_col=0)
    items = items.reset_index().rename(columns={"index": "instance_id"})

    # Merge features with IRT difficulties
    merged = items[["instance_id", "b"]].merge(
        features_df, on="instance_id", how="inner"
    )
    print(f"  {dataset}: {len(merged)} tasks with features + IRT difficulties")

    output_dir = ROOT / ABLATION_OUTPUT_BASE / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for filename, levels, display_name in INFO_LEVELS:
        available = get_feature_names_for_levels(levels)
        # Filter to features actually present in the CSV (defensive)
        present = [f for f in available if f in merged.columns]
        if len(present) < len(available):
            missing = set(available) - set(present)
            raise ValueError(
                f"{dataset}: missing features for {display_name}: {missing}"
            )

        top_features = select_top_features(merged, present, n=15)
        out_df = merged[["instance_id"] + top_features].copy()

        output_path = output_dir / filename
        out_df.to_csv(output_path, index=False)
        print(f"  Saved {output_path.relative_to(ROOT)} ({display_name}: {top_features})")
        result[filename] = ABLATION_OUTPUT_BASE / dataset / filename

    return result


def run_ablation_for_dataset(
    dataset: str, k_folds: int, rebuild: bool
) -> Tuple[str, Dict[str, Tuple[float, float]]]:
    """Run the full ablation for one dataset. Returns (display_name, {level: (mean_auc, std_auc)})."""
    display_name = DATASET_DEFAULTS[dataset]["display_name"]
    print(f"\n{'='*60}")
    print(f"Dataset: {display_name}")
    print(f"{'='*60}")

    # Build CSVs if needed
    output_dir = ROOT / ABLATION_OUTPUT_BASE / dataset
    any_missing = any(
        not (output_dir / filename).exists() for filename, _, _ in INFO_LEVELS
    )
    if rebuild or any_missing:
        build_ablation_csvs(dataset)

    # Run CV for each info level
    results = {}
    for filename, _, level_name in INFO_LEVELS:
        csv_path = ABLATION_OUTPUT_BASE / dataset / filename
        config = ExperimentAConfig.for_dataset(
            dataset,
            llm_judge_features_path=csv_path,
            embeddings_path=None,  # Disables embedding + grouped predictors
        )

        cv_output = cross_validate_all_predictors(config, ROOT, k=k_folds)
        cv_results = cv_output["cv_results"]

        llm_result = cv_results.get("llm_judge", {})
        mean_auc = llm_result.get("mean_auc")
        std_auc = llm_result.get("std_auc")

        if mean_auc is not None:
            results[filename] = (mean_auc, std_auc)
            print(f"  {level_name}: {mean_auc:.4f} ± {std_auc:.4f}")
        else:
            results[filename] = (float("nan"), float("nan"))
            print(f"  {level_name}: N/A")

        # Capture oracle and baseline from first run (same across all levels)
        if "oracle" not in results:
            for key in ("oracle", "constant_baseline"):
                r = cv_results.get(key, {})
                if r.get("mean_auc") is not None:
                    results[key] = (r["mean_auc"], r["std_auc"])
                else:
                    results[key] = (float("nan"), float("nan"))

    return display_name, results


def format_results_table(
    all_results: Dict[str, Tuple[str, Dict[str, Tuple[float, float]]]]
) -> str:
    """Format results as a markdown-style table."""
    # Column order matches ALL_DATASETS
    datasets = list(all_results.keys())
    display_names = [all_results[d][0] for d in datasets]

    # Column widths
    level_col_w = 20
    data_col_w = max(len(n) for n in display_names) + 4  # room for "0.xxxx ± 0.xxxx"
    data_col_w = max(data_col_w, 18)

    # Header
    lines = []
    header = f"{'Info Level':<{level_col_w}}"
    for name in display_names:
        header += f" | {name:^{data_col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Helper to format a row
    def _row(label: str, key: str) -> str:
        row = f"{label:<{level_col_w}}"
        for dataset in datasets:
            _, level_results = all_results[dataset]
            mean_auc, std_auc = level_results.get(key, (float("nan"), float("nan")))
            if not np.isnan(mean_auc):
                cell = f"{mean_auc:.4f}"
            else:
                cell = "N/A"
            row += f" | {cell:^{data_col_w}}"
        return row

    # Baseline row
    lines.append(_row("Baseline", "constant_baseline"))
    lines.append("")

    # Info level rows
    for filename, _, level_name in INFO_LEVELS:
        lines.append(_row(level_name, filename))

    lines.append("")

    # Oracle row
    lines.append(_row("Oracle", "oracle"))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run information level ablation study")
    parser.add_argument("--rebuild_csvs", action="store_true", help="Regenerate ablation CSVs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS, help="Datasets to run (default: all)"
    )
    parser.add_argument("--sequential", action="store_true", help="Run datasets sequentially (for debugging)")
    args = parser.parse_args()

    print("=" * 70)
    print("INFORMATION LEVEL ABLATION STUDY (v8 features)")
    print("=" * 70)

    all_results: Dict[str, Tuple[str, Dict[str, Tuple[float, float]]]] = {}

    if args.sequential:
        for dataset in args.datasets:
            display_name, results = run_ablation_for_dataset(
                dataset, args.k_folds, args.rebuild_csvs
            )
            all_results[dataset] = (display_name, results)
    else:
        with ProcessPoolExecutor(max_workers=len(args.datasets)) as executor:
            futures = {
                executor.submit(
                    run_ablation_for_dataset, dataset, args.k_folds, args.rebuild_csvs
                ): dataset
                for dataset in args.datasets
            }
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    display_name, results = future.result()
                    all_results[dataset] = (display_name, results)
                    print(f"\nCompleted: {display_name}")
                except Exception as e:
                    print(f"\nERROR ({dataset}): {e}")
                    raise

    # Print summary table in dataset order
    ordered_results = {d: all_results[d] for d in args.datasets if d in all_results}

    print("\n" + "=" * 70)
    print("RESULTS: LLM Judge AUC by Information Level")
    print("=" * 70)
    print()
    print(format_results_table(ordered_results))
    print()


if __name__ == "__main__":
    main()
