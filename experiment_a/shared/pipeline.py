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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd

from experiment_ab_shared.feature_source import (
    EmbeddingFeatureSource,
    CSVFeatureSource,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
)
from experiment_ab_shared import (
    load_dataset,
    load_dataset_for_fold,
    convert_numpy,
    filter_unsolved_tasks,
)
from experiment_ab_shared.dataset import (
    _load_binary_responses,
    _load_binomial_responses,
)

from experiment_a.shared.cross_validation import (
    k_fold_split_tasks,
    run_cv,
    CrossValidationResult,
)
from experiment_a.shared.baselines import (
    AgentOnlyPredictor,
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
    FeatureIRTCVPredictor,
)

# Default SWE-bench LLM Judge features (all 9 semantic features)
SWEBENCH_LLM_JUDGE_FEATURES = [
    "fix_in_description",
    "problem_clarity",
    "error_message_provided",
    "reproduction_steps",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
]


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


# Import CVPredictor protocol for type hints
from experiment_a.shared.cross_validation import CVPredictor


@dataclass
class CVPredictorConfig:
    """Configuration for a predictor in cross-validation.

    Attributes:
        predictor: Any predictor implementing the CVPredictor protocol
        name: Key for storing results
        display_name: Human-readable name for display
    """

    predictor: CVPredictor
    name: str
    display_name: str


def build_cv_predictors(
    config: Any,
    root: Path,
    llm_judge_features: Optional[List[str]] = None,
    include_feature_irt: bool = False,
) -> List[CVPredictorConfig]:
    """Build list of CVPredictor configurations for cross-validation.

    All predictors implement the CVPredictor protocol (fit/predict_probability).

    Args:
        config: Experiment configuration (ExperimentAConfig or TerminalBenchConfig)
        root: Root directory for resolving relative paths
        llm_judge_features: Optional list of feature columns for LLM Judge.
        include_feature_irt: Whether to include Feature-IRT joint learning methods.
            Defaults to False since they provide minimal improvement over Ridge.

    Returns:
        List of CVPredictorConfig objects with pre-instantiated predictors.
    """
    configs: List[CVPredictorConfig] = []

    # Oracle (upper bound) - uses full IRT model
    configs.append(
        CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true b)",
        )
    )

    # Embedding predictor (Ridge regression)
    if config.embeddings_path is not None:
        embeddings_path = root / config.embeddings_path
        if embeddings_path.exists():
            source = EmbeddingFeatureSource(embeddings_path)
            difficulty_predictor = FeatureBasedPredictor(
                source,
                ridge_alphas=list(config.ridge_alphas),
            )
            configs.append(
                CVPredictorConfig(
                    predictor=DifficultyPredictorAdapter(difficulty_predictor),
                    name="embedding_predictor",
                    display_name="Embedding",
                )
            )

            # Feature-IRT with embeddings (joint learning)
            if include_feature_irt:
                configs.append(
                    CVPredictorConfig(
                        predictor=FeatureIRTCVPredictor(source, verbose=False),
                        name="feature_irt_embedding",
                        display_name="Feature-IRT (Embedding)",
                    )
                )

    # LLM Judge predictor (Ridge regression)
    if config.llm_judge_features_path is not None:
        llm_judge_path = root / config.llm_judge_features_path
        if llm_judge_path.exists():
            feature_cols = llm_judge_features or SWEBENCH_LLM_JUDGE_FEATURES
            source = CSVFeatureSource(llm_judge_path, feature_cols, name="LLM Judge")
            difficulty_predictor = FeatureBasedPredictor(
                source,
                ridge_alphas=list(config.ridge_alphas),
            )
            configs.append(
                CVPredictorConfig(
                    predictor=DifficultyPredictorAdapter(difficulty_predictor),
                    name="llm_judge_predictor",
                    display_name="LLM Judge",
                )
            )

            # Feature-IRT with LLM Judge features (joint learning)
            if include_feature_irt:
                configs.append(
                    CVPredictorConfig(
                        predictor=FeatureIRTCVPredictor(source, verbose=False),
                        name="feature_irt_llm_judge",
                        display_name="Feature-IRT (LLM Judge)",
                    )
                )

    # Constant baseline (mean difficulty)
    configs.append(
        CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant_baseline",
            display_name="Constant (mean b)",
        )
    )

    # Agent-only baseline
    configs.append(
        CVPredictorConfig(
            predictor=AgentOnlyPredictor(),
            name="agent_only_baseline",
            display_name="Agent-only",
        )
    )

    return configs


def run_cross_validation(
    config: Any,
    spec: ExperimentSpec,
    root: Path,
    k: int = 5,
    metadata_loader: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    include_feature_irt: bool = False,
) -> Dict[str, Any]:
    """Run the evaluation pipeline with k-fold cross-validation.

    Uses the unified run_cv function for ALL predictors including baselines.

    Args:
        config: Experiment configuration
        spec: Experiment specification
        root: Root directory for resolving relative paths
        k: Number of folds
        metadata_loader: Optional callable to load task metadata
        include_feature_irt: Whether to include Feature-IRT joint learning methods.
            Defaults to False since they provide minimal improvement over Ridge.

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

    # Build ALL predictor configs (oracle, feature-based, baselines)
    predictor_configs = build_cv_predictors(
        config, root, llm_judge_features=spec.llm_judge_features,
        include_feature_irt=include_feature_irt,
    )

    # Determine if we should compute binomial metrics
    compute_pass_rate_mse = spec.is_binomial

    # Results dict
    cv_results: Dict[str, CrossValidationResult] = {}

    # Run CV for each predictor using the unified framework
    for i, pc in enumerate(predictor_configs, 1):
        print(f"\n{i}. {pc.display_name}:")
        cv_results[pc.name] = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            compute_pass_rate_mse=compute_pass_rate_mse,
        )
        result = cv_results[pc.name]
        if result.mean_auc is not None:
            print(f"   Mean AUC: {result.mean_auc:.4f} ± {result.std_auc:.4f}")
        else:
            print("   Mean AUC: N/A")

    # Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {spec.name} ({k}-FOLD CROSS-VALIDATION)")
    print("=" * 60)

    # Sort by mean AUC descending
    display_order = [
        (pc.display_name, pc.name, cv_results[pc.name].mean_auc or 0.0)
        for pc in predictor_configs
        if pc.name in cv_results
    ]
    display_order.sort(key=lambda x: x[2], reverse=True)

    if compute_pass_rate_mse:
        print(f"\n{'Method':<25} {'Mean AUC':>10} {'Std':>8} {'Pass Rate MSE':>14}")
        print("-" * 62)

        for name, key, _ in display_order:
            result = cv_results[key]
            if result.mean_auc is not None:
                mse_str = f"{result.mean_pass_rate_mse:.4f}" if result.mean_pass_rate_mse is not None else "N/A"
                print(f"{name:<25} {result.mean_auc:>10.4f} {result.std_auc:>8.4f} {mse_str:>14}")
            else:
                print(f"{name:<25} {'N/A':>10} {'N/A':>8} {'N/A':>14}")
    else:
        print(f"\n{'Method':<30} {'Mean AUC':>10} {'Std':>8}")
        print("-" * 50)

        for name, key, _ in display_order:
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
    parser.add_argument(
        "--include_feature_irt",
        action="store_true",
        help="Include Feature-IRT joint learning methods (slower, minimal improvement over Ridge)",
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

    # Build config kwargs from args - only override if CLI arg is provided
    config_kwargs: Dict[str, Any] = {
        "split_seed": args.split_seed,
        "output_dir": Path(args.output_dir),
        "exclude_unsolved": args.exclude_unsolved,
    }
    # Only override paths if explicitly provided via CLI
    if args.embeddings_path is not None:
        config_kwargs["embeddings_path"] = Path(args.embeddings_path)
    if args.llm_judge_features_path is not None:
        config_kwargs["llm_judge_features_path"] = Path(args.llm_judge_features_path)
    if args.llm_judge_max_features is not None:
        config_kwargs["llm_judge_max_features"] = args.llm_judge_max_features
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

    # Run cross-validation
    results = run_cross_validation(
        config, spec, root, args.k_folds, metadata_loader,
        include_feature_irt=args.include_feature_irt,
    )
    output_filename = f"experiment_a_cv{args.k_folds}_results.json"

    # Save results
    output_dir = root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    results = convert_numpy(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")