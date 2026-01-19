"""Shared pipeline for running Experiment A across different datasets.

This module provides the common evaluation pipeline that both SWE-bench and
TerminalBench experiments use. The experiments differ only in:
- Dataset name
- Response type (binary vs binomial)
- IRT cache directory
- LLM judge features (dataset-specific semantic features)
- Metadata loading (TerminalBench loads task data from repo)
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from experiment_a.difficulty_predictor import (
    EmbeddingPredictor,
    ConstantPredictor,
    GroundTruthPredictor,
    LLMJudgePredictor,
)
from experiment_a_common import (
    load_dataset,
    load_dataset_for_fold,
    compute_auc,
    run_evaluation_pipeline,
    PredictorConfig,
    agent_only_baseline,
    convert_numpy,
    filter_unsolved_tasks,
)
from experiment_a_common.dataset import (
    _load_binary_responses,
    _load_binomial_responses,
)
from experiment_a_common.cross_validation import (
    k_fold_split_tasks,
    run_cv_for_predictor,
    run_cv_for_baseline,
    CrossValidationResult,
)


@dataclass
class ExperimentSpec:
    """Specification for an experiment.

    Attributes:
        name: Human-readable experiment name (e.g., "SWE-bench", "TerminalBench")
        is_binomial: Whether responses are binomial (True) or binary (False)
        irt_cache_dir: Directory for caching fold-specific IRT models
        llm_judge_features: List of LLM judge feature columns to use.
            If None, uses the LLMJudgePredictor class defaults.
    """

    name: str
    is_binomial: bool
    irt_cache_dir: Path
    llm_judge_features: Optional[List[str]] = None


def build_predictor_configs(
    config: Any,
    root: Path,
    llm_judge_features: Optional[List[str]] = None,
) -> List[PredictorConfig]:
    """Build list of predictor configurations from experiment config.

    Args:
        config: Experiment configuration (ExperimentAConfig or TerminalBenchConfig)
        root: Root directory for resolving relative paths
        llm_judge_features: Optional list of feature columns for LLM Judge.
            If provided, passed to LLMJudgePredictor to override defaults.

    Returns:
        List of PredictorConfig objects for enabled predictors.
    """
    configs = []

    # Constant baseline (mean difficulty)
    configs.append(
        PredictorConfig(
            predictor_class=ConstantPredictor,
            name="constant_baseline",
            display_name="Constant (mean b)",
        )
    )

    # Embedding predictor
    if config.embeddings_path is not None:
        embeddings_path = root / config.embeddings_path
        if embeddings_path.exists():
            configs.append(
                PredictorConfig(
                    predictor_class=EmbeddingPredictor,
                    name="embedding_predictor",
                    display_name="Embedding",
                    kwargs={
                        "embeddings_path": embeddings_path,
                        "ridge_alphas": list(config.ridge_alphas),
                    },
                )
            )

    # LLM Judge predictor
    if config.llm_judge_features_path is not None:
        llm_judge_path = root / config.llm_judge_features_path
        if llm_judge_path.exists():
            kwargs = {
                "features_path": llm_judge_path,
                "ridge_alphas": list(config.llm_judge_ridge_alphas),
                "max_features": config.llm_judge_max_features,
            }
            # Only pass feature_cols if explicitly specified
            if llm_judge_features is not None:
                kwargs["feature_cols"] = llm_judge_features

            configs.append(
                PredictorConfig(
                    predictor_class=LLMJudgePredictor,
                    name="llm_judge_predictor",
                    display_name="LLM Judge",
                    kwargs=kwargs,
                )
            )

    return configs


def run_single_holdout(
    config: Any,
    spec: ExperimentSpec,
    root: Path,
    metadata_loader: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline with a single holdout split.

    Args:
        config: Experiment configuration
        spec: Experiment specification
        root: Root directory for resolving relative paths
        metadata_loader: Optional callable to load task metadata

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print(f"EXPERIMENT A: PRIOR VALIDATION (IRT AUC) - {spec.name}")
    print("=" * 60)

    # Resolve paths relative to root
    abilities_path = root / config.abilities_path
    items_path = root / config.items_path
    responses_path = root / config.responses_path
    output_dir = root / config.output_dir

    # 1. Load data using common loader
    print("\n1. Loading data...")
    data = load_dataset(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        test_fraction=config.test_fraction,
        split_seed=config.split_seed,
        is_binomial=spec.is_binomial,
        irt_cache_dir=spec.irt_cache_dir,
        metadata_loader=metadata_loader,
        exclude_unsolved=config.exclude_unsolved,
    )
    print(f"   Agents: {data.n_agents}")
    print(f"   Tasks: {data.n_tasks}")
    print(f"   Train tasks: {data.n_train_tasks}")
    print(f"   Test tasks: {data.n_test_tasks}")
    if metadata_loader and data.metadata:
        task_data = data.metadata.get("task_data", {})
        if task_data:
            print(f"   Tasks with metadata: {len(task_data)}")

    # 2. Initialize results dict
    results: Dict[str, Any] = {
        "config": config.to_dict(),
        "data_summary": {
            "n_agents": data.n_agents,
            "n_tasks_total": data.n_tasks,
            "n_train_tasks": data.n_train_tasks,
            "n_test_tasks": data.n_test_tasks,
        },
    }

    # Determine if we should compute binomial metrics
    compute_binomial = spec.is_binomial

    # 3. Oracle baseline (handled specially - needs full_items)
    print("\n2. Computing oracle baseline (ground truth b from full IRT)...")
    oracle_predictor = GroundTruthPredictor(data.full_items)
    oracle_preds = oracle_predictor.predict(data.test_tasks)
    oracle_result = compute_auc(data, oracle_preds, use_full_abilities=True)
    print(f"   Oracle AUC: {oracle_result.get('auc', 'N/A'):.4f}")

    # Add binomial metrics to oracle if applicable
    if compute_binomial:
        from experiment_a_common.binomial_metrics import compute_binomial_metrics
        from experiment_a_common.dataset import BinomialExperimentData
        if isinstance(data, BinomialExperimentData):
            binom_result = compute_binomial_metrics(data, oracle_preds, use_full_abilities=True)
            oracle_result["binomial_metrics"] = binom_result.to_dict()
            print(f"   Oracle Pass Rate MSE: {binom_result.pass5_mse:.4f}")

    results["oracle"] = oracle_result

    # 4. Build predictor configs and run evaluation pipeline
    predictor_configs = build_predictor_configs(
        config, root, llm_judge_features=spec.llm_judge_features
    )
    pipeline_results = run_evaluation_pipeline(
        data, predictor_configs, verbose=True, compute_binomial=compute_binomial
    )
    results.update(pipeline_results)

    # 5. Agent-only baseline (uses common implementation)
    print("\nComputing agent-only baseline...")
    agent_result = agent_only_baseline(data, compute_binomial=compute_binomial)
    print(f"   Agent-only AUC: {agent_result.get('auc', 'N/A'):.4f}")
    if compute_binomial and "binomial_metrics" in agent_result:
        bm = agent_result["binomial_metrics"]
        print(f"   Agent-only Pass Rate MSE: {bm['pass5_mse']:.4f}")
    results["agent_only_baseline"] = agent_result

    # 6. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTest set: {data.n_test_tasks} tasks")

    # Define display order
    display_order = [
        ("Oracle (true b)", "oracle"),
        ("Embedding", "embedding_predictor"),
        ("LLM Judge", "llm_judge_predictor"),
        ("Constant (mean b)", "constant_baseline"),
        ("Agent-only", "agent_only_baseline"),
    ]

    if compute_binomial:
        print(f"\n{'Method':<25} {'AUC':>10} {'Pass Rate MSE':>14}")
        print("-" * 52)

        for name, key in display_order:
            result = results.get(key, {})
            if "auc_result" in result:
                auc = result["auc_result"].get("auc")
            else:
                auc = result.get("auc")

            bm = result.get("binomial_metrics", {})
            mse = bm.get("pass5_mse")

            if auc is not None:
                mse_str = f"{mse:.4f}" if mse is not None else "N/A"
                print(f"{name:<25} {auc:>10.4f} {mse_str:>14}")
            elif "error" in result:
                print(f"{name:<25} {'ERROR':>10} {'N/A':>14}")
            elif "skipped" in result:
                continue
            elif key not in results:
                continue
            else:
                print(f"{name:<25} {'N/A':>10} {'N/A':>14}")
    else:
        print(f"\n{'Method':<30} {'AUC':>10}")
        print("-" * 42)

        for name, key in display_order:
            result = results.get(key, {})
            if "auc_result" in result:
                auc = result["auc_result"].get("auc")
            else:
                auc = result.get("auc")

            if auc is not None:
                print(f"{name:<30} {auc:>10.4f}")
            elif "error" in result:
                print(f"{name:<30} {'ERROR':>10}")
            elif "skipped" in result:
                continue
            elif key not in results:
                continue
            else:
                print(f"{name:<30} {'N/A':>10}")

    return results


def run_cross_validation(
    config: Any,
    spec: ExperimentSpec,
    root: Path,
    k: int = 5,
    metadata_loader: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline with k-fold cross-validation.

    Args:
        config: Experiment configuration
        spec: Experiment specification
        root: Root directory for resolving relative paths
        k: Number of folds
        metadata_loader: Optional callable to load task metadata

    Returns:
        Dict with CV results for each method
    """
    print("=" * 60)
    print(f"EXPERIMENT A: {k}-FOLD CROSS-VALIDATION - {spec.name}")
    print("=" * 60)

    # Resolve paths relative to root
    abilities_path = root / config.abilities_path
    items_path = root / config.items_path
    responses_path = root / config.responses_path

    # Load full items to get all task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    # Optionally filter unsolved tasks before generating folds
    n_excluded = 0
    if config.exclude_unsolved:
        if spec.is_binomial:
            responses = _load_binomial_responses(responses_path)
        else:
            responses = _load_binary_responses(responses_path)
        all_task_ids, n_excluded = filter_unsolved_tasks(
            all_task_ids, responses, spec.is_binomial
        )
        print(f"\nExcluded {n_excluded} unsolved tasks ({len(all_task_ids)} remaining)")

    print(f"\nTotal tasks: {len(all_task_ids)}")
    print(f"Tasks per fold (test): ~{len(all_task_ids) // k}")

    # Generate k folds
    folds = k_fold_split_tasks(all_task_ids, k=k, seed=config.split_seed)

    # Create a fold data loader function
    def load_fold_data(train_tasks: List[str], test_tasks: List[str], fold_idx: int):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k,
            split_seed=config.split_seed,
            is_binomial=spec.is_binomial,
            irt_cache_dir=spec.irt_cache_dir,
            metadata_loader=metadata_loader,
            exclude_unsolved=config.exclude_unsolved,
        )

    # Build predictor configs
    predictor_configs = build_predictor_configs(
        config, root, llm_judge_features=spec.llm_judge_features
    )

    # Results dict
    cv_results: Dict[str, CrossValidationResult] = {}

    # Add oracle as a predictor config
    oracle_config = PredictorConfig(
        predictor_class=GroundTruthPredictor,
        name="oracle",
        display_name="Oracle (true b)",
        kwargs={"items_df": full_items},
        use_full_abilities=True,
    )

    # Determine if we should compute binomial metrics
    compute_binomial = spec.is_binomial

    # Run CV for oracle
    print("\n1. Oracle (ground truth b from full IRT):")
    cv_results["oracle"] = run_cv_for_predictor(
        oracle_config, folds, load_fold_data, verbose=True, compute_binomial=compute_binomial
    )
    print(
        f"   Mean AUC: {cv_results['oracle'].mean_auc:.4f} ± {cv_results['oracle'].std_auc:.4f}"
    )

    # Run CV for each predictor
    for i, pc in enumerate(predictor_configs, 2):
        print(f"\n{i}. {pc.display_name}:")
        cv_results[pc.name] = run_cv_for_predictor(
            pc, folds, load_fold_data, verbose=True, compute_binomial=compute_binomial
        )
        result = cv_results[pc.name]
        if result.mean_auc is not None:
            print(f"   Mean AUC: {result.mean_auc:.4f} ± {result.std_auc:.4f}")
        else:
            print("   Mean AUC: N/A")

    # Run CV for agent-only baseline with binomial support
    print(f"\n{len(predictor_configs) + 2}. Agent-only baseline:")

    def agent_only_with_binomial(data):
        return agent_only_baseline(data, compute_binomial=compute_binomial)

    cv_results["agent_only_baseline"] = run_cv_for_baseline(
        agent_only_with_binomial, folds, load_fold_data, verbose=True, compute_binomial=compute_binomial
    )
    agent_result = cv_results["agent_only_baseline"]
    if agent_result.mean_auc is not None:
        print(f"   Mean AUC: {agent_result.mean_auc:.4f} ± {agent_result.std_auc:.4f}")
    else:
        print("   Mean AUC: N/A")

    # Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({k}-FOLD CROSS-VALIDATION)")
    print("=" * 60)

    # Define display order
    display_order = [
        ("Oracle (true b)", "oracle"),
        ("Embedding", "embedding_predictor"),
        ("LLM Judge", "llm_judge_predictor"),
        ("Constant (mean b)", "constant_baseline"),
        ("Agent-only", "agent_only_baseline"),
    ]

    if compute_binomial:
        print(f"\n{'Method':<25} {'Mean AUC':>10} {'Std':>8} {'Pass Rate MSE':>14}")
        print("-" * 62)

        for name, key in display_order:
            if key in cv_results:
                result = cv_results[key]
                if result.mean_auc is not None:
                    mse_str = f"{result.mean_pass5_mse:.4f}" if result.mean_pass5_mse is not None else "N/A"
                    print(f"{name:<25} {result.mean_auc:>10.4f} {result.std_auc:>8.4f} {mse_str:>14}")
                else:
                    print(f"{name:<25} {'N/A':>10} {'N/A':>8} {'N/A':>14}")
    else:
        print(f"\n{'Method':<30} {'Mean AUC':>10} {'Std':>8}")
        print("-" * 50)

        for name, key in display_order:
            if key in cv_results:
                result = cv_results[key]
                if result.mean_auc is not None:
                    print(f"{name:<30} {result.mean_auc:>10.4f} {result.std_auc:>8.4f}")
                else:
                    print(f"{name:<30} {'N/A':>10} {'N/A':>8}")

    # Return results as dict
    return {
        "config": config.to_dict(),
        "k_folds": k,
        "cv_results": {name: asdict(result) for name, result in cv_results.items()},
    }


def create_main_parser(experiment_name: str, default_output_dir: str) -> argparse.ArgumentParser:
    """Create the common argument parser for experiment main functions.

    Args:
        experiment_name: Name of the experiment for help text
        default_output_dir: Default output directory path

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=f"Run Experiment A: Prior Validation (IRT AUC) on {experiment_name}"
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--single_holdout",
        action="store_true",
        help="Use single 20%% holdout instead of cross-validation (legacy behavior)",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of tasks to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed for train/test split (default: 0)",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Path to pre-computed embeddings .npz file",
    )
    parser.add_argument(
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Path to LLM judge features CSV file",
    )
    parser.add_argument(
        "--llm_judge_max_features",
        type=int,
        default=None,
        help="Max features to select for LLM Judge (default: None = all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help=f"Output directory (default: {default_output_dir})",
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default=None,
        help="Path to IRT items.csv (overrides config default)",
    )
    parser.add_argument(
        "--abilities_path",
        type=str,
        default=None,
        help="Path to IRT abilities.csv (overrides config default)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running",
    )
    parser.add_argument(
        "--exclude_unsolved",
        action="store_true",
        help="Exclude tasks that no agent has solved from both train and test sets",
    )
    return parser


def run_experiment_main(
    config_class: Type,
    spec: ExperimentSpec,
    root: Path,
    metadata_loader_factory: Optional[Callable[[Any], Callable[[List[str]], Dict[str, Any]]]] = None,
) -> None:
    """Shared main entry point for experiments.

    Args:
        config_class: The config class (ExperimentAConfig or TerminalBenchConfig)
        spec: Experiment specification
        root: Root directory for the project
        metadata_loader_factory: Optional factory that takes config and returns a metadata loader.
            This is used by TerminalBench to create a loader that uses config.repo_path.
    """
    parser = create_main_parser(spec.name, str(config_class().output_dir))
    args = parser.parse_args()

    # Build config kwargs from args
    config_kwargs = dict(
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        embeddings_path=Path(args.embeddings_path) if args.embeddings_path else None,
        output_dir=Path(args.output_dir),
        llm_judge_features_path=(
            Path(args.llm_judge_features_path) if args.llm_judge_features_path else None
        ),
        llm_judge_max_features=args.llm_judge_max_features,
        exclude_unsolved=args.exclude_unsolved,
    )
    if args.items_path:
        config_kwargs["items_path"] = Path(args.items_path)
    if args.abilities_path:
        config_kwargs["abilities_path"] = Path(args.abilities_path)

    config = config_class(**config_kwargs)

    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    # Create metadata loader if factory provided
    metadata_loader = None
    if metadata_loader_factory is not None:
        metadata_loader = metadata_loader_factory(config)

    # Run experiment - CV is the default, single holdout is legacy
    if args.single_holdout:
        results = run_single_holdout(config, spec, root, metadata_loader)
        output_filename = "experiment_a_results.json"
    else:
        results = run_cross_validation(config, spec, root, args.k_folds, metadata_loader)
        output_filename = f"experiment_a_cv{args.k_folds}_results.json"

    # Save results
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    results = convert_numpy(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")