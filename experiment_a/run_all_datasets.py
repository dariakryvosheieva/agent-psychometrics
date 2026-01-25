#!/usr/bin/env python3
"""Run Experiment A on all datasets in parallel and produce a summary table.

This script runs experiment_a on all available datasets (SWE-bench Verified,
GSO, TerminalBench, SWE-bench Pro) in parallel, then compiles results into
a compact table format.

Usage:
    python -m experiment_a.run_all_datasets
    python -m experiment_a.run_all_datasets --unified_judge  # Use unified judge features
    python -m experiment_a.run_all_datasets --output results.csv  # Save to CSV
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExperimentADatasetSpec:
    """Specification for running experiment_a on a single dataset.

    This is different from experiment_b's ExperimentADatasetSpec which includes
    frontier-specific settings (cutoff dates, agent dates, etc.).
    """
    name: str
    short_name: str  # For CLI filtering
    config_module: str  # e.g., "experiment_a.swebench.config"
    config_class_name: str  # e.g., "ExperimentAConfig"
    spec_module: str  # e.g., "experiment_a.swebench.train_evaluate"
    unified_judge_path: Optional[Path]
    extra_kwargs: Dict[str, Any]  # Additional config kwargs like exclude_unsolved=True


# Dataset specifications
DATASETS = [
    ExperimentADatasetSpec(
        name="SWE-bench Verified",
        short_name="swebench",
        config_module="experiment_a.swebench.config",
        config_class_name="ExperimentAConfig",  # Note: SWE-bench uses ExperimentAConfig
        spec_module="experiment_a.swebench.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/swebench_unified/llm_judge_features.csv"),
        extra_kwargs={},
    ),
    ExperimentADatasetSpec(
        name="GSO",
        short_name="gso",
        config_module="experiment_a.gso.config",
        config_class_name="GSOConfig",
        spec_module="experiment_a.gso.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/gso_unified/llm_judge_features.csv"),
        extra_kwargs={"exclude_unsolved": True},  # Match Daria's setup
    ),
    ExperimentADatasetSpec(
        name="TerminalBench",
        short_name="terminalbench",
        config_module="experiment_a.terminalbench.config",
        config_class_name="TerminalBenchConfig",
        spec_module="experiment_a.terminalbench.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/terminalbench_unified/llm_judge_features.csv"),
        extra_kwargs={},
    ),
    ExperimentADatasetSpec(
        name="SWE-bench Pro",
        short_name="swebench_pro",
        config_module="experiment_a.swebench_pro.config",
        config_class_name="SWEBenchProConfig",
        spec_module="experiment_a.swebench_pro.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/swebench_pro_unified/llm_judge_features.csv"),
        extra_kwargs={},
    ),
]


def run_single_dataset(
    dataset_config: ExperimentADatasetSpec,
    use_unified_judge: bool = False,
    output_base: Optional[Path] = None,
    k_folds: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """Run experiment_a on a single dataset and return results.

    Args:
        dataset_config: Dataset configuration.
        use_unified_judge: Whether to use unified judge features.
        output_base: Base directory for outputs.
        k_folds: Number of CV folds.

    Returns:
        Tuple of (dataset_name, results_dict).
    """
    import importlib

    try:
        # Import config class
        config_mod = importlib.import_module(dataset_config.config_module)
        config_class = getattr(config_mod, dataset_config.config_class_name)

        # Import SPEC and ROOT
        spec_mod = importlib.import_module(dataset_config.spec_module)
        spec = getattr(spec_mod, "SPEC")
        root = getattr(spec_mod, "ROOT", Path("."))

        # Import pipeline
        from experiment_a.shared.pipeline import run_cross_validation

    except ImportError as e:
        return dataset_config.name, {"error": f"Import error: {e}"}
    except AttributeError as e:
        return dataset_config.name, {"error": f"Attribute error: {e}"}

    # Build config
    try:
        config_kwargs = dict(dataset_config.extra_kwargs)

        if use_unified_judge and dataset_config.unified_judge_path:
            if dataset_config.unified_judge_path.exists():
                config_kwargs["llm_judge_features_path"] = dataset_config.unified_judge_path
            else:
                return dataset_config.name, {
                    "error": f"Unified judge features not found: {dataset_config.unified_judge_path}"
                }

        if output_base:
            config_kwargs["output_dir"] = output_base / dataset_config.short_name

        config = config_class(**config_kwargs)

    except Exception as e:
        return dataset_config.name, {"error": f"Config error: {e}"}

    # Run the experiment
    try:
        results = run_cross_validation(
            config, spec, root, k_folds,
            metadata_loader=None,
            include_feature_irt=False,
        )
        return dataset_config.name, results

    except Exception as e:
        import traceback
        return dataset_config.name, {"error": f"Execution error: {e}\n{traceback.format_exc()}"}


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract key metrics from experiment results.

    Args:
        results: Raw results dictionary from run_cross_validation.

    Returns:
        Dictionary mapping method name -> mean AUC.
    """
    if "error" in results:
        return {"error": results["error"]}

    metrics: Dict[str, Optional[float]] = {}

    # Internal name to display name mappings
    name_mappings = {
        "oracle": "Oracle",
        "embedding_predictor": "Embedding",
        "llm_judge_predictor": "LLM Judge",
        "grouped_ridge": "Grouped Ridge",
        "constant_baseline": "Baseline",
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
    """Format results as a markdown table.

    Args:
        all_results: Dict mapping dataset name -> {method: auc}.
        methods: List of methods to include (in order).

    Returns:
        Formatted markdown table string.
    """
    if methods is None:
        methods = ["Oracle", "Grouped Ridge", "Embedding", "LLM Judge", "Baseline"]

    # Build header
    header = "| Dataset | " + " | ".join(methods) + " |"
    separator = "|" + "|".join(["---"] * (len(methods) + 1)) + "|"

    rows = [header, separator]

    for dataset_name, metrics in all_results.items():
        if "error" in metrics:
            row = f"| {dataset_name} | " + " | ".join(["ERROR"] * len(methods)) + " |"
        else:
            values = []
            for method in methods:
                if method in metrics and metrics[method] is not None:
                    values.append(f"{metrics[method]:.4f}")
                else:
                    values.append("-")
            row = f"| {dataset_name} | " + " | ".join(values) + " |"
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
        methods = ["Oracle", "Grouped Ridge", "Embedding", "LLM Judge", "Baseline"]

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
        "--unified_judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use unified LLM judge features (default: True). Use --no-unified_judge for dataset-specific features.",
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
        choices=["swebench", "gso", "terminalbench", "swebench_pro"],
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
        help="Maximum parallel workers (default: 4)",
    )

    args = parser.parse_args()

    # Filter datasets if specified
    datasets_to_run = DATASETS
    if args.datasets:
        datasets_to_run = [
            d for d in DATASETS
            if d.short_name in args.datasets
        ]

    print(f"Running Experiment A on {len(datasets_to_run)} datasets...")
    print(f"Unified judge features: {args.unified_judge}")
    print(f"K-folds: {args.k_folds}")
    print()

    all_results: Dict[str, Dict[str, Optional[float]]] = {}

    if args.sequential:
        # Sequential execution
        for config in datasets_to_run:
            print(f"Running {config.name}...")
            name, results = run_single_dataset(
                config,
                use_unified_judge=args.unified_judge,
                output_base=args.output_dir,
                k_folds=args.k_folds,
            )
            metrics = extract_metrics(results)
            all_results[name] = metrics

            if "error" in metrics:
                print(f"  ERROR: {str(metrics['error'])[:100]}...")
            else:
                oracle = metrics.get('Oracle')
                grouped = metrics.get('Grouped Ridge')
                oracle_str = f"{oracle:.4f}" if oracle else "N/A"
                grouped_str = f"{grouped:.4f}" if grouped else "N/A"
                print(f"  Done: Oracle={oracle_str}, Grouped Ridge={grouped_str}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_dataset,
                    config,
                    args.unified_judge,
                    args.output_dir,
                    args.k_folds,
                ): config.name
                for config in datasets_to_run
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
                        grouped = metrics.get('Grouped Ridge')
                        oracle_str = f"{oracle:.4f}" if oracle else "N/A"
                        grouped_str = f"{grouped:.4f}" if grouped else "N/A"
                        print(f"{name}: Oracle={oracle_str}, Grouped Ridge={grouped_str}")
                except Exception as e:
                    all_results[dataset_name] = {"error": str(e)}
                    print(f"{dataset_name}: EXCEPTION - {e}")

    # Sort results by original dataset order
    ordered_results: Dict[str, Dict[str, Optional[float]]] = {}
    for config in datasets_to_run:
        if config.name in all_results:
            ordered_results[config.name] = all_results[config.name]

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
