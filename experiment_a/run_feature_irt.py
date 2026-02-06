#!/usr/bin/env python3
"""Run Experiment A with Feature-IRT (joint training) on all datasets.

This script runs experiment_a on all datasets using Feature-IRT predictors
(joint training of feature weights and agent abilities) instead of Ridge regression.

The key difference from run_all_datasets.py:
- Ridge regression: Train IRT → freeze difficulties → Ridge predicts difficulty from features
- Feature-IRT: Jointly learn feature weights + abilities by maximizing IRT log-likelihood

Usage:
    python -m experiment_a.run_feature_irt
    python -m experiment_a.run_feature_irt --datasets swebench gso
    python -m experiment_a.run_feature_irt --sequential  # Run datasets sequentially
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiment_a.run_all_datasets import (
    DATASETS,
    ExperimentADatasetSpec,
    format_results_table,
    save_results_csv,
)


def run_feature_irt_single_dataset(
    dataset_config: ExperimentADatasetSpec,
    output_base: Optional[Path] = None,
    k_folds: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """Run Feature-IRT experiment on a single dataset.

    Args:
        dataset_config: Dataset configuration.
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

        # Import SPEC
        spec_mod = importlib.import_module(dataset_config.spec_module)
        spec = getattr(spec_mod, "SPEC")
        root = getattr(spec_mod, "ROOT", Path("."))

        # Import necessary modules
        from experiment_a.shared.pipeline import CVPredictorConfig
        from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
        from experiment_a.shared.baselines import (
            OraclePredictor,
            ConstantPredictor,
            FeatureIRTCVPredictor,
        )
        from experiment_ab_shared import load_dataset_for_fold, filter_unsolved_tasks
        from experiment_ab_shared.dataset import _load_binary_responses, _load_binomial_responses
        from experiment_ab_shared.feature_source import (
            GroupedFeatureSource,
            RegularizedFeatureSource,
            build_feature_sources,
        )

    except ImportError as e:
        return dataset_config.name, {"error": f"Import error: {e}"}
    except AttributeError as e:
        return dataset_config.name, {"error": f"Attribute error: {e}"}

    # Build config
    try:
        config_kwargs = dict(dataset_config.extra_kwargs)

        # Always use default unified judge path
        if dataset_config.unified_judge_path and dataset_config.unified_judge_path.exists():
            config_kwargs["llm_judge_features_path"] = dataset_config.unified_judge_path

        if output_base:
            config_kwargs["output_dir"] = output_base / dataset_config.short_name

        config = config_class(**config_kwargs)

    except Exception as e:
        return dataset_config.name, {"error": f"Config error: {e}"}

    # Run the experiment
    try:
        # Resolve paths
        abilities_path = root / config.abilities_path
        items_path = root / config.items_path
        responses_path = root / config.responses_path
        embeddings_path = root / config.embeddings_path if config.embeddings_path else None
        llm_judge_path = root / config.llm_judge_features_path if config.llm_judge_features_path else None

        # Load full items to get all task IDs
        full_items = pd.read_csv(items_path, index_col=0)
        all_task_ids = list(full_items.index)

        # Optionally filter unsolved tasks
        if config.exclude_unsolved:
            if spec.is_binomial:
                responses = _load_binomial_responses(responses_path)
            else:
                responses = _load_binary_responses(responses_path)
            all_task_ids, n_excluded = filter_unsolved_tasks(
                all_task_ids, responses, spec.is_binomial
            )
            print(f"\n{dataset_config.name}: Excluded {n_excluded} unsolved tasks ({len(all_task_ids)} remaining)")

        print(f"\n{dataset_config.name}: {len(all_task_ids)} tasks, {k_folds}-fold CV")

        # Generate k folds
        folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

        # Create fold data loader
        def load_fold_data(train_tasks: List[str], test_tasks: List[str], fold_idx: int):
            return load_dataset_for_fold(
                abilities_path=abilities_path,
                items_path=items_path,
                responses_path=responses_path,
                train_tasks=train_tasks,
                test_tasks=test_tasks,
                fold_idx=fold_idx,
                k_folds=k_folds,
                split_seed=config.split_seed,
                is_binomial=spec.is_binomial,
                irt_cache_dir=spec.irt_cache_dir,
                exclude_unsolved=config.exclude_unsolved,
            )

        # Build feature sources
        feature_source_list = build_feature_sources(
            embeddings_path=embeddings_path,
            llm_judge_path=llm_judge_path,
            verbose=False,
        )
        source_by_name = {name: source for name, source in feature_source_list}

        # Build predictor configs with Feature-IRT
        predictor_configs: List[CVPredictorConfig] = []

        # Oracle (unchanged)
        predictor_configs.append(
            CVPredictorConfig(
                predictor=OraclePredictor(),
                name="oracle",
                display_name="Oracle",
            )
        )

        # Feature-IRT with Grouped (Emb + LLM) - uses per-source regularization
        if "Embedding" in source_by_name and "LLM Judge" in source_by_name:
            grouped_source = GroupedFeatureSource([
                RegularizedFeatureSource(source_by_name["Embedding"]),
                RegularizedFeatureSource(source_by_name["LLM Judge"]),
            ])
            predictor_configs.append(
                CVPredictorConfig(
                    predictor=FeatureIRTCVPredictor(grouped_source, verbose=False),
                    name="feature_irt_grouped",
                    display_name="Feature-IRT (Emb+LLM)",
                )
            )

        # Feature-IRT with LLM Judge
        if "LLM Judge" in source_by_name:
            predictor_configs.append(
                CVPredictorConfig(
                    predictor=FeatureIRTCVPredictor(source_by_name["LLM Judge"], verbose=False),
                    name="feature_irt_llm",
                    display_name="Feature-IRT (LLM)",
                )
            )

        # Feature-IRT with Embedding
        if "Embedding" in source_by_name:
            predictor_configs.append(
                CVPredictorConfig(
                    predictor=FeatureIRTCVPredictor(source_by_name["Embedding"], verbose=False),
                    name="feature_irt_embedding",
                    display_name="Feature-IRT (Emb)",
                )
            )

        # Baseline (unchanged)
        predictor_configs.append(
            CVPredictorConfig(
                predictor=ConstantPredictor(),
                name="constant_baseline",
                display_name="Baseline",
            )
        )

        # Run CV for each predictor
        cv_results = {}
        for pc in predictor_configs:
            print(f"  {pc.display_name}...", end="", flush=True)
            result = run_cv(
                pc.predictor,
                folds,
                load_fold_data,
                verbose=False,
                compute_pass_rate_mse=spec.is_binomial,
            )
            cv_results[pc.name] = result
            print(f" AUC={result.mean_auc:.4f}")

        # Format results
        results = {
            "config": config.to_dict(),
            "k_folds": k_folds,
            "cv_results": {
                name: {
                    "mean_auc": r.mean_auc,
                    "std_auc": r.std_auc,
                    "fold_aucs": r.fold_aucs,
                }
                for name, r in cv_results.items()
            },
        }

        return dataset_config.name, results

    except Exception as e:
        import traceback
        return dataset_config.name, {"error": f"Execution error: {e}\n{traceback.format_exc()}"}


def extract_metrics(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract key metrics from experiment results."""
    if "error" in results:
        return {"error": results["error"]}

    metrics: Dict[str, Optional[float]] = {}
    name_mappings = {
        "oracle": "Oracle",
        "feature_irt_grouped": "Feature-IRT (Emb+LLM)",
        "feature_irt_llm": "Feature-IRT (LLM)",
        "feature_irt_embedding": "Feature-IRT (Emb)",
        "constant_baseline": "Baseline",
    }

    cv_results = results.get("cv_results", {})
    for internal_name, display_name in name_mappings.items():
        if internal_name in cv_results:
            result = cv_results[internal_name]
            if result.get("mean_auc") is not None:
                metrics[display_name] = result["mean_auc"]

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment A with Feature-IRT on all datasets"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/tmp/experiment_a_feature_irt"),
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
        help="Run datasets sequentially instead of in parallel (default: parallel)",
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

    args = parser.parse_args()

    # Filter datasets if specified
    datasets_to_run = DATASETS
    if args.datasets:
        datasets_to_run = [d for d in DATASETS if d.short_name in args.datasets]

    print(f"Running Feature-IRT on {len(datasets_to_run)} datasets...")
    print(f"K-folds: {args.k_folds}")
    print()

    all_results: Dict[str, Dict[str, Optional[float]]] = {}

    if args.sequential:
        for config in datasets_to_run:
            name, results = run_feature_irt_single_dataset(
                config,
                output_base=args.output_dir,
                k_folds=args.k_folds,
            )
            metrics = extract_metrics(results)
            all_results[name] = metrics

            if "error" in metrics:
                print(f"  ERROR: {str(metrics['error'])[:100]}...")
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_feature_irt_single_dataset,
                    config,
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
                except Exception as e:
                    all_results[dataset_name] = {"error": str(e)}
                    print(f"{dataset_name}: EXCEPTION - {e}")

    # Sort results by original dataset order
    ordered_results: Dict[str, Dict[str, Optional[float]]] = {}
    for config in datasets_to_run:
        if config.name in all_results:
            ordered_results[config.name] = all_results[config.name]

    print("\n" + "=" * 80)
    print("FEATURE-IRT RESULTS SUMMARY")
    print("=" * 80 + "\n")

    # Define methods to display
    methods = ["Oracle", "Feature-IRT (Emb+LLM)", "Feature-IRT (LLM)", "Feature-IRT (Emb)", "Baseline"]

    # Print table
    table = format_results_table(ordered_results, methods=methods)
    print(table)

    # Save CSV if requested
    if args.output:
        save_results_csv(ordered_results, args.output, methods=methods)
        print(f"\nResults saved to: {args.output}")

    # Save JSON with full details
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "feature_irt_summary.json"

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
