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
    env_features_path: Optional[Path]  # Path to environment features CSV
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
        env_features_path=Path("chris_output/env_features/swebench_verified/env_features.csv"),
        extra_kwargs={},
    ),
    ExperimentADatasetSpec(
        name="GSO",
        short_name="gso",
        config_module="experiment_a.gso.config",
        config_class_name="GSOConfig",
        spec_module="experiment_a.gso.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/gso_unified/llm_judge_features.csv"),
        env_features_path=None,  # Not yet extracted for GSO
        extra_kwargs={"exclude_unsolved": True},  # Match Daria's setup
    ),
    ExperimentADatasetSpec(
        name="TerminalBench",
        short_name="terminalbench",
        config_module="experiment_a.terminalbench.config",
        config_class_name="TerminalBenchConfig",
        spec_module="experiment_a.terminalbench.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/terminalbench_unified/llm_judge_features.csv"),
        env_features_path=None,  # Not yet extracted for TerminalBench
        extra_kwargs={},
    ),
    ExperimentADatasetSpec(
        name="SWE-bench Pro",
        short_name="swebench_pro",
        config_module="experiment_a.swebench_pro.config",
        config_class_name="SWEBenchProConfig",
        spec_module="experiment_a.swebench_pro.train_evaluate",
        unified_judge_path=Path("chris_output/llm_judge_features/swebench_pro_unified/llm_judge_features.csv"),
        env_features_path=None,  # Not yet extracted for SWE-bench Pro
        extra_kwargs={},
    ),
]


def run_single_dataset(
    dataset_config: ExperimentADatasetSpec,
    use_unified_judge: bool = False,
    unified_judge_suffix: str = "",
    output_base: Optional[Path] = None,
    k_folds: int = 5,
    n_jobs_methods: int = 1,
    n_jobs_folds: int = 1,
    include_mlp: bool = True,
    include_trees: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Run experiment_a on a single dataset and return results.

    Args:
        dataset_config: Dataset configuration.
        use_unified_judge: Whether to use unified judge features.
        unified_judge_suffix: Suffix to append to unified judge directory (e.g., '_core').
        output_base: Base directory for outputs.
        k_folds: Number of CV folds.
        n_jobs_methods: Number of parallel jobs for method execution.
        n_jobs_folds: Number of parallel jobs for fold execution.
        include_mlp: Whether to include MLP predictors (default True).
        include_trees: Whether to include tree-based predictors (default False).

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
            # Apply suffix to the directory name if provided
            judge_path = dataset_config.unified_judge_path
            if unified_judge_suffix:
                # Insert suffix before the filename (e.g., swebench_unified -> swebench_unified_core)
                parent_with_suffix = judge_path.parent.parent / (judge_path.parent.name + unified_judge_suffix)
                judge_path = parent_with_suffix / judge_path.name

            if judge_path.exists():
                config_kwargs["llm_judge_features_path"] = judge_path
            else:
                return dataset_config.name, {
                    "error": f"Unified judge features not found: {judge_path}"
                }

        # Add environment features if available
        if dataset_config.env_features_path and dataset_config.env_features_path.exists():
            config_kwargs["env_features_path"] = dataset_config.env_features_path

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
            include_mlp=include_mlp,
            include_trees=include_trees,
            n_jobs_methods=n_jobs_methods,
            n_jobs_folds=n_jobs_folds,
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
        "env_predictor": "Environment",
        "llm_judge_tree": "LLM Judge (Tree)",
        "llm_judge_rf": "LLM Judge (RF)",
        "grouped_ridge": "Grouped Ridge",
        "grouped_ridge_emb_env": "Emb + Env",
        "grouped_ridge_llm_env": "LLM + Env",
        "grouped_ridge_emb_llm": "Emb + LLM",
        "stacked_residual": "Stacked (Emb → LLM)",
        "mlp_embedding": "MLP (Emb)",
        "mlp_llm_judge": "MLP (Judge)",
        "mlp_grouped": "MLP (Grouped)",
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
    """Format results as a markdown table with aligned columns.

    Args:
        all_results: Dict mapping dataset name -> {method: auc}.
        methods: List of methods to include (in order).

    Returns:
        Formatted markdown table string with proper column alignment.
    """
    if methods is None:
        methods = ["Oracle", "Grouped Ridge", "Emb + LLM", "Emb + Env", "LLM + Env",
                   "Embedding", "LLM Judge", "Environment", "Baseline"]

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
        methods = ["Oracle", "Grouped Ridge", "Emb + LLM", "Emb + Env", "LLM + Env",
                   "Embedding", "LLM Judge", "Environment", "Baseline"]

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
        "--unified_judge_suffix",
        type=str,
        default="",
        help="Suffix to append to unified judge directory names (e.g., '_core' for filtered features).",
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
        help="Maximum parallel workers for datasets (default: 4)",
    )
    parser.add_argument(
        "--n_jobs_methods",
        type=int,
        default=1,
        help="Parallel jobs for methods within each dataset (default: 1 = sequential)",
    )
    parser.add_argument(
        "--n_jobs_folds",
        type=int,
        default=1,
        help="Parallel jobs for folds within each method (default: 1 = sequential)",
    )
    parser.add_argument(
        "--mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include MLP predictors (default: True). Use --no-mlp to skip for faster local runs.",
    )
    parser.add_argument(
        "--trees",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include tree-based predictors (default: False). Use --trees to include Decision Tree and Random Forest.",
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
    print(f"Include MLP: {args.mlp}, Include trees: {args.trees}")
    print(f"Parallelization: datasets={args.max_workers}, methods={args.n_jobs_methods}, folds={args.n_jobs_folds}")
    print()

    all_results: Dict[str, Dict[str, Optional[float]]] = {}

    if args.sequential:
        # Sequential execution
        for config in datasets_to_run:
            print(f"Running {config.name}...")
            name, results = run_single_dataset(
                config,
                use_unified_judge=args.unified_judge,
                unified_judge_suffix=args.unified_judge_suffix,
                output_base=args.output_dir,
                k_folds=args.k_folds,
                n_jobs_methods=args.n_jobs_methods,
                n_jobs_folds=args.n_jobs_folds,
                include_mlp=args.mlp,
                include_trees=args.trees,
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
                    args.unified_judge_suffix,
                    args.output_dir,
                    args.k_folds,
                    args.n_jobs_methods,
                    args.n_jobs_folds,
                    args.mlp,
                    args.trees,
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
