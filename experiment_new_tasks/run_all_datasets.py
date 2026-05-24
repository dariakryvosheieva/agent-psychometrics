#!/usr/bin/env python3
"""Run Experiment A on all datasets in parallel and produce a summary table.

This script runs experiment_new_tasks on all available datasets (SWE-bench Verified,
GSO, TerminalBench, SWE-bench Pro) in parallel, then compiles results into
a compact table format.

Usage:
    python -m experiment_new_tasks.run_all_datasets
    python -m experiment_new_tasks.run_all_datasets --output results.csv  # Save to CSV
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiment_new_tasks.config import DATASET_DEFAULTS


ROOT = Path(__file__).resolve().parents[1]

# All datasets in display order
ALL_DATASETS = ["swebench_verified", "swebench_pro", "gso", "terminalbench"]


def run_single_dataset(
    dataset: str,
    output_base: Optional[Path] = None,
    k_folds: int = 5,
    n_fold_seeds: int = 20,
    fold_seed_start: int = 0,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    predictor_factory=None,
    llm_judge_features_path: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    responses_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
    abilities_path: Optional[str] = None,
    items_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run experiment_new_tasks on a single dataset and return results.

    Args:
        dataset: Dataset short name (e.g., "swebench_verified", "gso").
        output_base: Base directory for outputs.
        k_folds: Number of CV folds.
        n_fold_seeds: Number of different fold seeds to evaluate.
        fold_seed_start: First fold seed; seeds are consecutive from this value.
        n_bootstrap: Number of seed-level bootstrap samples.
        bootstrap_seed: Random seed for bootstrap resampling.
        predictor_factory: Optional callable(source_name, source, config) -> CVPredictor.
        llm_judge_features_path: Optional override for LLM judge features CSV path.
            Supports {dataset} template variable.
        embeddings_path: Optional override for embeddings .npz path.
            Supports {dataset} template variable.
        responses_path: Optional override for the response matrix JSONL path.
            Supports {dataset} template variable. Use this to point at a
            per-attempt file (`{"successes", "trials"}` cells) for binomial-
            likelihood IRT instead of the default binary file.
        output_dir: Optional override for the dataset's output directory (and
            therefore its IRT cache). Use a distinct directory when running
            with non-default responses to avoid mixing fold caches with the
            canonical binary results.
        abilities_path / items_path: Optional overrides for the full-IRT
            abilities/items CSV files used by the Oracle predictor. Supports
            {dataset} template. Required when --responses_path points to a
            file that includes agents not represented in the default full-IRT
            (e.g., when running binomial against a refreshed per-attempt
            scrape with new agents).

    Returns:
        Tuple of (dataset_display_name, results_dict).
    """
    from experiment_new_tasks.config import ExperimentAConfig
    from experiment_new_tasks.pipeline import cross_validate_all_predictors_repeated_seeds

    try:
        overrides = {}
        if llm_judge_features_path is not None:
            expanded = llm_judge_features_path.replace("{dataset}", dataset)
            overrides["llm_judge_features_path"] = Path(expanded)
        if embeddings_path is not None:
            expanded = embeddings_path.replace("{dataset}", dataset)
            overrides["embeddings_path"] = Path(expanded)
        if responses_path is not None:
            expanded = responses_path.replace("{dataset}", dataset)
            overrides["responses_path"] = Path(expanded)
        if output_dir is not None:
            overrides["output_dir"] = Path(output_dir)
        if abilities_path is not None:
            overrides["abilities_path"] = Path(abilities_path.replace("{dataset}", dataset))
        if items_path is not None:
            overrides["items_path"] = Path(items_path.replace("{dataset}", dataset))
        config = ExperimentAConfig.for_dataset(dataset, **overrides)
    except Exception as e:
        display_name = DATASET_DEFAULTS[dataset]["display_name"]
        return display_name, {"error": f"Config error: {e}"}

    # Run the experiment
    try:
        fold_seeds = list(range(fold_seed_start, fold_seed_start + n_fold_seeds))
        results = cross_validate_all_predictors_repeated_seeds(
            config,
            ROOT,
            k=k_folds,
            fold_seeds=fold_seeds,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            predictor_factory=predictor_factory,
        )

        return config.display_name, results

    except Exception as e:
        import traceback
        return config.display_name, {"error": f"Execution error: {e}\n{traceback.format_exc()}"}


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
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
        "llm_judge": "LLM-as-a-Judge",
        "grouped": "Combined",
        "constant_baseline": "Baseline",
    }

    cv_results = results.get("cv_results", {})

    for internal_name, display_name in name_mappings.items():
        if internal_name in cv_results:
            result = cv_results[internal_name]
            if result.get("mean_auc") is not None:
                metrics[display_name] = result["mean_auc"]
                if result.get("std_auc") is not None:
                    metrics[f"{display_name}__std"] = result["std_auc"]
                bootstrap = result.get("bootstrap_difference_vs_baseline")
                if bootstrap is not None:
                    metrics[f"{display_name}__delta"] = result.get(
                        "mean_difference_vs_baseline"
                    )
                    metrics[f"{display_name}__ci_low"] = bootstrap.get("ci_low")
                    metrics[f"{display_name}__ci_high"] = bootstrap.get("ci_high")
                    metrics[f"{display_name}__p_value"] = bootstrap.get("p_value")
                    metrics[f"{display_name}__p_value_is_upper_bound"] = bootstrap.get(
                        "p_value_is_upper_bound"
                    )

    return metrics


def format_results_table(
    all_results: Dict[str, Dict[str, Any]],
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
        methods = ["Baseline", "Embedding", "LLM-as-a-Judge", "Combined", "Oracle"]

    # Build data rows first to calculate column widths
    data_rows = []
    for dataset_name, metrics in all_results.items():
        if "error" in metrics:
            values = ["ERROR"] * len(methods)
        else:
            values = []
            for method in methods:
                if method in metrics and metrics[method] is not None:
                    std = metrics.get(f"{method}__std")
                    if std is not None:
                        values.append(f"{metrics[method]:.4f} +/- {std:.4f}")
                    else:
                        values.append(f"{metrics[method]:.4f}")
                else:
                    values.append("-")
        data_rows.append((dataset_name, values))

    # Calculate column widths
    col_widths = [max(len("Benchmark"), max(len(row[0]) for row in data_rows))]
    for i, method in enumerate(methods):
        method_width = len(method)
        value_width = max(len(row[1][i]) for row in data_rows)
        col_widths.append(max(method_width, value_width))

    # Build formatted table
    def pad(text: str, width: int) -> str:
        return text.ljust(width)

    header = "| " + " | ".join(pad(col, col_widths[i]) for i, col in enumerate(["Benchmark"] + methods)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    rows = [header, separator]
    for dataset_name, values in data_rows:
        row = "| " + pad(dataset_name, col_widths[0]) + " | " + " | ".join(
            pad(v, col_widths[i + 1]) for i, v in enumerate(values)
        ) + " |"
        rows.append(row)

    return "\n".join(rows)


def save_results_csv(
    all_results: Dict[str, Dict[str, Any]],
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
        methods = ["Baseline", "Embedding", "LLM-as-a-Judge", "Combined", "Oracle"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Benchmark"]
        for method in methods:
            header.extend(
                [
                    method,
                    f"{method} SD",
                    f"{method} Delta vs Baseline",
                    f"{method} Delta 95% CI Low",
                    f"{method} Delta 95% CI High",
                    f"{method} Delta p-value",
                ]
            )
        writer.writerow(header)

        for dataset_name, metrics in all_results.items():
            row = [dataset_name]
            for method in methods:
                if "error" in metrics:
                    row.extend(["ERROR", "", "", "", "", ""])
                elif method in metrics and metrics[method] is not None:
                    std = metrics.get(f"{method}__std")
                    delta = metrics.get(f"{method}__delta")
                    ci_low = metrics.get(f"{method}__ci_low")
                    ci_high = metrics.get(f"{method}__ci_high")
                    p_value = metrics.get(f"{method}__p_value")
                    is_upper_bound = metrics.get(f"{method}__p_value_is_upper_bound")
                    if p_value is None:
                        p_value_str = ""
                    elif is_upper_bound:
                        p_value_str = f"<{p_value:.6g}"
                    else:
                        p_value_str = f"{p_value:.6g}"
                    row.extend(
                        [
                            f"{metrics[method]:.4f}",
                            f"{std:.4f}" if std is not None else "",
                            f"{delta:.4f}" if delta is not None else "",
                            f"{ci_low:.4f}" if ci_low is not None else "",
                            f"{ci_high:.4f}" if ci_high is not None else "",
                            p_value_str,
                        ]
                    )
                else:
                    row.extend(["", "", "", "", "", ""])
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
        "--n_fold_seeds",
        type=int,
        default=20,
        help="Number of different fold seeds to run (default: 20)",
    )
    parser.add_argument(
        "--fold_seed_start",
        type=int,
        default=0,
        help="First fold seed; seeds are consecutive from this value (default: 0)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of seed-level bootstrap samples (default: 10000)",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for seed-level bootstrap resampling (default: 0)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum parallel workers for datasets (default: 4)",
    )
    parser.add_argument(
        "--feature_irt",
        action="store_true",
        help="Use Feature-IRT (joint training) instead of Ridge regression.",
    )
    parser.add_argument(
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Override LLM judge features CSV path. Supports {dataset} template "
             "(e.g., 'output/.../v2/{dataset}/features.csv').",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Override embeddings .npz path. Supports {dataset} template "
             "(e.g., 'embeddings/my_embeddings_{dataset}.npz').",
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        default=None,
        help="Override the response matrix JSONL path. Supports {dataset} "
             "template. Use a per-attempt file (`{\"successes\", \"trials\"}` "
             "cells) to train the IRT model with binomial likelihood and to "
             "evaluate AUC over expanded per-attempt observations.",
    )
    parser.add_argument(
        "--per_dataset_output_dir",
        type=str,
        default=None,
        help="Override each dataset's output directory (which also controls the "
             "fold-IRT cache). Supports {dataset} template. Use this together "
             "with --responses_path to isolate caches from the default binary "
             "results.",
    )
    parser.add_argument(
        "--abilities_path",
        type=str,
        default=None,
        help="Override full-IRT abilities.csv path (Oracle predictor). Supports "
             "{dataset} template. Required when --responses_path introduces "
             "agents that aren't in the default full IRT (e.g., new agents in a "
             "refreshed per-attempt scrape).",
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default=None,
        help="Override full-IRT items.csv path (Oracle predictor). Supports "
             "{dataset} template. Pair with --abilities_path.",
    )
    args = parser.parse_args()
    if args.n_fold_seeds < 1:
        parser.error("--n_fold_seeds must be >= 1")
    if args.n_bootstrap < 1:
        parser.error("--n_bootstrap must be >= 1")

    # Filter datasets if specified
    datasets_to_run = args.datasets if args.datasets else ALL_DATASETS

    # Resolve predictor factory
    predictor_factory = None
    if args.feature_irt:
        from experiment_new_tasks.feature_irt import feature_irt_predictor_factory
        predictor_factory = feature_irt_predictor_factory

    training_method = "Feature-IRT (joint training)" if args.feature_irt else "Ridge regression"
    print(f"Running Experiment A on {len(datasets_to_run)} datasets...")
    print(f"Training method: {training_method}")
    print(f"Fold seeds: {args.n_fold_seeds}")
    print(f"K-folds per seed: {args.k_folds}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Parallelization: datasets={args.max_workers}")
    print()

    all_results: Dict[str, Dict[str, Any]] = {}

    def _per_dataset_output_dir(dataset: str) -> Optional[Path]:
        if args.per_dataset_output_dir is None:
            return None
        expanded = args.per_dataset_output_dir.replace("{dataset}", dataset)
        if "{dataset}" not in args.per_dataset_output_dir and len(datasets_to_run) > 1:
            raise SystemExit(
                "--per_dataset_output_dir must contain the {dataset} template when "
                "multiple datasets are selected, to avoid cache collisions."
            )
        return Path(expanded)

    if args.sequential:
        # Sequential execution
        for dataset in datasets_to_run:
            display_name = DATASET_DEFAULTS[dataset]["display_name"]
            print(f"Running {display_name}...")
            name, results = run_single_dataset(
                dataset,
                output_base=args.output_dir,
                k_folds=args.k_folds,
                n_fold_seeds=args.n_fold_seeds,
                fold_seed_start=args.fold_seed_start,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
                predictor_factory=predictor_factory,
                llm_judge_features_path=args.llm_judge_features_path,
                embeddings_path=args.embeddings_path,
                responses_path=args.responses_path,
                output_dir=_per_dataset_output_dir(dataset),
                abilities_path=args.abilities_path,
                items_path=args.items_path,
            )
            metrics = extract_metrics(results)
            all_results[name] = metrics

            if "error" in metrics:
                print(f"  ERROR: {str(metrics['error'])[:100]}...")
            else:
                oracle = metrics.get('Oracle')
                combined = metrics.get('Combined')
                oracle_str = f"{oracle:.4f}" if oracle else "N/A"
                combined_str = f"{combined:.4f}" if combined else "N/A"
                print(f"  Done: Oracle={oracle_str}, Combined={combined_str}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_dataset,
                    dataset,
                    output_base=args.output_dir,
                    k_folds=args.k_folds,
                    n_fold_seeds=args.n_fold_seeds,
                    fold_seed_start=args.fold_seed_start,
                    n_bootstrap=args.n_bootstrap,
                    bootstrap_seed=args.bootstrap_seed,
                    predictor_factory=predictor_factory,
                    llm_judge_features_path=args.llm_judge_features_path,
                    embeddings_path=args.embeddings_path,
                    responses_path=args.responses_path,
                    output_dir=_per_dataset_output_dir(dataset),
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
                        combined = metrics.get('Combined')
                        oracle_str = f"{oracle:.4f}" if oracle else "N/A"
                        combined_str = f"{combined:.4f}" if combined else "N/A"
                        print(f"{name}: Oracle={oracle_str}, Combined={combined_str}")
                except Exception as e:
                    all_results[dataset_name] = {"error": str(e)}
                    print(f"{dataset_name}: EXCEPTION - {e}")

    # Sort results by original dataset order
    ordered_results: Dict[str, Dict[str, Any]] = {}
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

    # Ensure output directory exists before saving any files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results_csv(ordered_results, args.output)
        print(f"\nResults saved to: {args.output}")

    # Save JSON with full details
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
