#!/usr/bin/env python3
"""Run Experiment A on all datasets in parallel and produce a summary table.

This script runs experiment_a on all available datasets (SWE-bench Verified,
GSO, TerminalBench, SWE-bench Pro) in parallel, then compiles results into
a compact table format.

Usage:
    python -m experiment_a.run_all_datasets
    python -m experiment_a.run_all_datasets --output results.csv  # Save to CSV
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiment_a.shared.config import DATASET_DEFAULTS


ROOT = Path(__file__).resolve().parents[1]

# All datasets in display order
ALL_DATASETS = ["swebench", "gso", "terminalbench", "swebench_pro"]

# Ablation paths per dataset (only used with --judge_ablation)
JUDGE_ABLATION_PATHS: Dict[str, Dict[str, Path]] = {
    "swebench": {
        "no_solution": Path("chris_output/experiment_a/llm_judge_features/llm_judge_no_solution_plus_auditor.csv"),
        "problem_only": Path("chris_output/llm_judge_features/swebench_unified_problem_only/llm_judge_features.csv"),
    },
    "gso": {
        "no_solution": Path("chris_output/llm_judge_features/gso_unified_no_solution/llm_judge_features.csv"),
        "problem_only": Path("chris_output/llm_judge_features/gso_unified_problem_only/llm_judge_features.csv"),
    },
    "terminalbench": {
        "no_solution": Path("chris_output/llm_judge_features/terminalbench_unified_no_solution/llm_judge_features.csv"),
        "problem_only": Path("chris_output/llm_judge_features/terminalbench_unified_problem_only/llm_judge_features.csv"),
    },
    "swebench_pro": {
        "no_solution": Path("chris_output/llm_judge_features/swebench_pro_unified_no_solution/llm_judge_features.csv"),
        "problem_only": Path("chris_output/llm_judge_features/swebench_pro_unified_problem_only/llm_judge_features.csv"),
    },
}


def run_single_dataset(
    dataset: str,
    output_base: Optional[Path] = None,
    k_folds: int = 5,
    judge_ablation: bool = False,
    extra_embeddings_paths: Optional[List[Tuple[str, Path]]] = None,
    extra_llm_judge_paths: Optional[List[Tuple[str, Path]]] = None,
    coefficients: bool = False,
    predictor_factory=None,
) -> Tuple[str, Dict[str, Any]]:
    """Run experiment_a on a single dataset and return results.

    Args:
        dataset: Dataset short name (e.g., "swebench", "gso").
        output_base: Base directory for outputs.
        k_folds: Number of CV folds.
        judge_ablation: Whether to include no-solution and problem-only LLM judge ablations.
        extra_embeddings_paths: Additional embedding paths for ablation studies.
        extra_llm_judge_paths: Additional LLM judge paths for ablation studies.
        coefficients: Whether to extract LLM Judge Ridge coefficients.
        predictor_factory: Optional callable(source_name, source, config) -> CVPredictor.

    Returns:
        Tuple of (dataset_display_name, results_dict).
    """
    from experiment_a.shared.config import ExperimentAConfig
    from experiment_a.shared.pipeline import cross_validate_all_predictors

    # Build config overrides
    config_kwargs: Dict[str, Any] = {}

    # Add ablation paths if requested
    if judge_ablation and dataset in JUDGE_ABLATION_PATHS:
        ablation_paths = []
        for variant, path in JUDGE_ABLATION_PATHS[dataset].items():
            if (ROOT / path).exists():
                ablation_paths.append((variant, path))
        if ablation_paths:
            if extra_llm_judge_paths:
                extra_llm_judge_paths = list(extra_llm_judge_paths) + ablation_paths
            else:
                extra_llm_judge_paths = ablation_paths

    try:
        config = ExperimentAConfig.for_dataset(dataset, **config_kwargs)
    except Exception as e:
        display_name = DATASET_DEFAULTS[dataset]["display_name"]
        return display_name, {"error": f"Config error: {e}"}

    # Set up coefficient extraction if requested
    diagnostics_extractors = None
    if coefficients:
        from experiment_a.shared.coefficient_analysis import make_llm_coef_extractor
        diagnostics_extractors = make_llm_coef_extractor()

    # Run the experiment
    try:
        results = cross_validate_all_predictors(
            config, ROOT, k_folds,
            extra_embeddings_paths=extra_embeddings_paths,
            extra_llm_judge_paths=extra_llm_judge_paths,
            diagnostics_extractors=diagnostics_extractors,
            predictor_factory=predictor_factory,
        )

        # Print coefficient analysis if requested
        if coefficients:
            from experiment_a.shared.coefficient_analysis import (
                print_coefficient_table,
                save_coefficient_bar_chart,
            )
            cv_results_dict = results.get("cv_results", {})
            llm_result = cv_results_dict.get("llm_judge")
            if llm_result is not None:
                fold_diagnostics = llm_result.get("fold_diagnostics", [])
                coeffs = [d for d in fold_diagnostics if d is not None]
                if coeffs:
                    print(f"\n{'=' * 80}")
                    print(f"LLM JUDGE COEFFICIENT ANALYSIS — {config.display_name}")
                    print(f"{'=' * 80}")
                    print_coefficient_table(coeffs)

                    if output_base:
                        chart_dir = output_base / dataset
                        chart_dir.mkdir(parents=True, exist_ok=True)
                        save_coefficient_bar_chart(
                            coeffs,
                            chart_dir / "coefficient_bar_chart.png",
                            title=f"Mean Coefficient Magnitude ({config.display_name})",
                        )

        return config.display_name, results

    except Exception as e:
        import traceback
        return config.display_name, {"error": f"Execution error: {e}\n{traceback.format_exc()}"}


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract key metrics from experiment results.

    Args:
        results: Raw results dictionary from cross_validate_all_predictors.

    Returns:
        Dictionary mapping method name -> mean AUC.
    """
    if "error" in results:
        return {"error": results["error"]}

    metrics: Dict[str, Optional[float]] = {}

    # Internal name to display name mappings
    name_mappings = {
        "oracle": "Oracle",
        "embedding": "Embedding",
        "llm_judge": "LLM Judge",
        "grouped": "Grouped",
        "constant_baseline": "Baseline",
        # Ablation study predictors
        "llm_judge_no_solution": "LLM (no sol)",
        "llm_judge_problem_only": "LLM (prob only)",
    }

    cv_results = results.get("cv_results", {})

    for internal_name, display_name in name_mappings.items():
        if internal_name in cv_results:
            result = cv_results[internal_name]
            if result.get("mean_auc") is not None:
                metrics[display_name] = result["mean_auc"]

    return metrics


def format_results_table(
    all_results: Dict[str, Dict[str, Optional[float]]],
    methods: Optional[List[str]] = None,
) -> str:
    """Format results as a markdown table with aligned columns.

    Args:
        all_results: Dict mapping dataset name -> {method: auc}.
        methods: List of methods to include (in order).

    Returns:
        Formatted markdown table string with proper column alignment.
    """
    if methods is None:
        methods = ["Oracle", "Grouped", "Embedding", "LLM Judge",
                   "LLM (no sol)", "LLM (prob only)", "Baseline"]

    # Build data rows first to calculate column widths
    data_rows = []
    for dataset_name, metrics in all_results.items():
        if "error" in metrics:
            values = ["ERROR"] * len(methods)
        else:
            values = []
            for method in methods:
                if method in metrics and metrics[method] is not None:
                    values.append(f"{metrics[method]:.4f}")
                else:
                    values.append("-")
        data_rows.append((dataset_name, values))

    # Calculate column widths
    col_widths = [max(len("Dataset"), max(len(row[0]) for row in data_rows))]
    for i, method in enumerate(methods):
        method_width = len(method)
        value_width = max(len(row[1][i]) for row in data_rows)
        col_widths.append(max(method_width, value_width))

    # Build formatted table
    def pad(text: str, width: int) -> str:
        return text.ljust(width)

    header = "| " + " | ".join(pad(col, col_widths[i]) for i, col in enumerate(["Dataset"] + methods)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    rows = [header, separator]
    for dataset_name, values in data_rows:
        row = "| " + pad(dataset_name, col_widths[0]) + " | " + " | ".join(
            pad(v, col_widths[i + 1]) for i, v in enumerate(values)
        ) + " |"
        rows.append(row)

    return "\n".join(rows)


def save_results_csv(
    all_results: Dict[str, Dict[str, Optional[float]]],
    output_path: Path,
    methods: Optional[List[str]] = None,
) -> None:
    """Save results to a CSV file.

    Args:
        all_results: Dict mapping dataset name -> {method: auc}.
        output_path: Path to save CSV.
        methods: List of methods to include.
    """
    import csv

    if methods is None:
        methods = ["Oracle", "Grouped", "Embedding", "LLM Judge",
                   "LLM (no sol)", "LLM (prob only)", "Baseline"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset"] + methods)

        for dataset_name, metrics in all_results.items():
            row = [dataset_name]
            for method in methods:
                if "error" in metrics:
                    row.append("ERROR")
                elif method in metrics and metrics[method] is not None:
                    row.append(f"{metrics[method]:.4f}")
                else:
                    row.append("")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment A on all datasets and produce summary table"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/tmp/experiment_a_all"),
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        help="Specific datasets to run (default: all)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run datasets sequentially instead of in parallel",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum parallel workers for datasets (default: 4)",
    )
    parser.add_argument(
        "--judge_ablation",
        action="store_true",
        help="Include LLM judge ablation variants (no-solution, problem-only) for each dataset.",
    )
    parser.add_argument(
        "--llm_judge_paths",
        type=str,
        default=None,
        help="Comma-separated list of LLM judge paths to compare (ablation study)",
    )
    parser.add_argument(
        "--embeddings_paths",
        type=str,
        default=None,
        help="Comma-separated list of embedding paths to compare (ablation study)",
    )
    parser.add_argument(
        "--coefficients",
        action="store_true",
        help="Extract and display LLM Judge Ridge coefficients (Table 10 / Figure 3).",
    )
    parser.add_argument(
        "--feature_irt",
        action="store_true",
        help="Use Feature-IRT (joint training) instead of Ridge regression.",
    )

    args = parser.parse_args()

    # Filter datasets if specified
    datasets_to_run = args.datasets if args.datasets else ALL_DATASETS

    # Parse extra feature paths for ablation studies
    extra_embeddings_paths: Optional[List[Tuple[str, Path]]] = None
    extra_llm_judge_paths: Optional[List[Tuple[str, Path]]] = None

    if args.embeddings_paths:
        extra_embeddings_paths = []
        for path_str in args.embeddings_paths.split(","):
            path_str = path_str.strip()
            if path_str:
                path = Path(path_str)
                # Extract a short name from the path
                name = path.stem
                if "__" in name:
                    parts = name.split("__")
                    name = parts[-1] if parts[-1] else parts[-2]
                extra_embeddings_paths.append((name, path))

    if args.llm_judge_paths:
        extra_llm_judge_paths = []
        for path_str in args.llm_judge_paths.split(","):
            path_str = path_str.strip()
            if path_str:
                path = Path(path_str)
                # Use parent directory name as the variant name
                name = path.parent.name
                extra_llm_judge_paths.append((name, path))

    # Resolve predictor factory
    predictor_factory = None
    if args.feature_irt:
        from experiment_a.shared.feature_irt import feature_irt_predictor_factory
        predictor_factory = feature_irt_predictor_factory

    training_method = "Feature-IRT (joint training)" if args.feature_irt else "Ridge regression"
    print(f"Running Experiment A on {len(datasets_to_run)} datasets...")
    print(f"Training method: {training_method}")
    print(f"K-folds: {args.k_folds}")
    print(f"Judge ablation: {args.judge_ablation}")
    print(f"Parallelization: datasets={args.max_workers}")
    if extra_embeddings_paths:
        print(f"Extra embedding paths: {[name for name, _ in extra_embeddings_paths]}")
    if extra_llm_judge_paths:
        print(f"Extra LLM judge paths: {[name for name, _ in extra_llm_judge_paths]}")
    print()

    all_results: Dict[str, Dict[str, Optional[float]]] = {}

    if args.sequential:
        # Sequential execution
        for dataset in datasets_to_run:
            display_name = DATASET_DEFAULTS[dataset]["display_name"]
            print(f"Running {display_name}...")
            name, results = run_single_dataset(
                dataset,
                output_base=args.output_dir,
                k_folds=args.k_folds,
                judge_ablation=args.judge_ablation,
                extra_embeddings_paths=extra_embeddings_paths,
                extra_llm_judge_paths=extra_llm_judge_paths,
                coefficients=args.coefficients,
                predictor_factory=predictor_factory,
            )
            metrics = extract_metrics(results)
            all_results[name] = metrics

            if "error" in metrics:
                print(f"  ERROR: {str(metrics['error'])[:100]}...")
            else:
                oracle = metrics.get('Oracle')
                grouped = metrics.get('Grouped')
                oracle_str = f"{oracle:.4f}" if oracle else "N/A"
                grouped_str = f"{grouped:.4f}" if grouped else "N/A"
                print(f"  Done: Oracle={oracle_str}, Grouped={grouped_str}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_dataset,
                    dataset,
                    output_base=args.output_dir,
                    k_folds=args.k_folds,
                    judge_ablation=args.judge_ablation,
                    extra_embeddings_paths=extra_embeddings_paths,
                    extra_llm_judge_paths=extra_llm_judge_paths,
                    coefficients=args.coefficients,
                    predictor_factory=predictor_factory,
                ): DATASET_DEFAULTS[dataset]["display_name"]
                for dataset in datasets_to_run
            }

            for future in as_completed(futures):
                dataset_name = futures[future]
                try:
                    name, results = future.result()
                    metrics = extract_metrics(results)
                    all_results[name] = metrics

                    if "error" in metrics:
                        print(f"{name}: ERROR - {str(metrics['error'])[:80]}...")
                    else:
                        oracle = metrics.get('Oracle')
                        grouped = metrics.get('Grouped')
                        oracle_str = f"{oracle:.4f}" if oracle else "N/A"
                        grouped_str = f"{grouped:.4f}" if grouped else "N/A"
                        print(f"{name}: Oracle={oracle_str}, Grouped={grouped_str}")
                except Exception as e:
                    all_results[dataset_name] = {"error": str(e)}
                    print(f"{dataset_name}: EXCEPTION - {e}")

    # Sort results by original dataset order
    ordered_results: Dict[str, Dict[str, Optional[float]]] = {}
    for dataset in datasets_to_run:
        display_name = DATASET_DEFAULTS[dataset]["display_name"]
        if display_name in all_results:
            ordered_results[display_name] = all_results[display_name]

    print("\n" + "=" * 80)
    print("EXPERIMENT A RESULTS SUMMARY")
    print("=" * 80 + "\n")

    # Print table
    table = format_results_table(ordered_results)
    print(table)

    # Save CSV if requested
    if args.output:
        save_results_csv(ordered_results, args.output)
        print(f"\nResults saved to: {args.output}")

    # Save JSON with full details
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"

    # Convert any non-serializable types
    serializable_results = {}
    for name, metrics in ordered_results.items():
        serializable_results[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) and v is not None else v
            for k, v in metrics.items()
        }

    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Full results saved to: {json_path}")


if __name__ == "__main__":
    main()
