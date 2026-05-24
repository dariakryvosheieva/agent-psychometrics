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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiment_new_tasks.config import DATASET_DEFAULTS
from experiment_new_tasks.results import (
    extract_metrics as extract_metrics_with_mapping,
    format_results_table,
    save_results_csv,
    save_summary_json,
)


ROOT = Path(__file__).resolve().parents[1]

# All datasets in display order
ALL_DATASETS = ["swebench_verified", "swebench_pro", "gso", "terminalbench"]
METHOD_NAME_MAPPINGS = {
    "oracle": "Oracle",
    "embedding": "Embedding",
    "llm_judge": "LLM-as-a-Judge",
    "grouped": "Combined",
    "constant_baseline": "Baseline",
}
RESULT_METHODS = ["Baseline", "Embedding", "LLM-as-a-Judge", "Combined", "Oracle"]


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
    return extract_metrics_with_mapping(results, METHOD_NAME_MAPPINGS)


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
    table = format_results_table(ordered_results, RESULT_METHODS)
    print(table)

    # Ensure output directory exists before saving any files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results_csv(ordered_results, args.output, RESULT_METHODS)
        print(f"\nResults saved to: {args.output}")

    # Save JSON with full details
    json_path = args.output_dir / "summary.json"

    save_summary_json(ordered_results, json_path)
    print(f"Full results saved to: {json_path}")


if __name__ == "__main__":
    main()
