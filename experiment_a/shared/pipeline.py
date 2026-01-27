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
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd

from experiment_ab_shared.feature_source import (
    EmbeddingFeatureSource,
    CSVFeatureSource,
    RegularizedFeatureSource,
    GroupedFeatureSource,
    build_feature_sources,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
    GroupedRidgePredictor,
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
    FullFeatureIRTAdapter,
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
    full_firt_l2_weight: float = 0.001,
    full_firt_l2_residual: float = 0.0001,
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

    # Resolve paths relative to root
    embeddings_path = (
        root / config.embeddings_path if config.embeddings_path is not None else None
    )
    llm_judge_path = (
        root / config.llm_judge_features_path
        if config.llm_judge_features_path is not None
        else None
    )
    trajectory_path = (
        root / config.trajectory_features_path
        if getattr(config, "trajectory_features_path", None) is not None
        else None
    )

    # Build feature sources once using shared utility (verbose=False to avoid extra output)
    feature_source_list = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        llm_judge_feature_cols=llm_judge_features,
        trajectory_features_path=trajectory_path,
        verbose=False,
    )

    # Build a dict for easy lookup by source name
    source_by_name = {name: source for name, source in feature_source_list}

    # Embedding predictor (Ridge regression)
    if "Embedding" in source_by_name:
        source = source_by_name["Embedding"]
        difficulty_predictor = FeatureBasedPredictor(
            source,
            alphas=list(config.ridge_alphas),
        )
        configs.append(
            CVPredictorConfig(
                predictor=DifficultyPredictorAdapter(difficulty_predictor),
                name="embedding_predictor",
                display_name="Embedding",
            )
        )

        # Full Feature-IRT with embeddings (trains on ALL tasks like Oracle)
        # Tests whether features + IRT can improve on Oracle IRT alone
        if include_feature_irt:
            configs.append(
                CVPredictorConfig(
                    predictor=FullFeatureIRTAdapter(
                        source,
                        l2_weight=full_firt_l2_weight,
                        l2_residual=full_firt_l2_residual,
                        verbose=True,
                    ),
                    name="full_feature_irt_embedding",
                    display_name="Full Feature-IRT (Embedding)",
                )
            )

    # LLM Judge predictor (Ridge regression)
    if "LLM Judge" in source_by_name:
        source = source_by_name["LLM Judge"]
        difficulty_predictor = FeatureBasedPredictor(
            source,
            alphas=list(config.ridge_alphas),
        )
        configs.append(
            CVPredictorConfig(
                predictor=DifficultyPredictorAdapter(difficulty_predictor),
                name="llm_judge_predictor",
                display_name="LLM Judge",
            )
        )

        # Full Feature-IRT with LLM Judge (trains on ALL tasks like Oracle)
        if include_feature_irt:
            configs.append(
                CVPredictorConfig(
                    predictor=FullFeatureIRTAdapter(
                        source,
                        l2_weight=full_firt_l2_weight,
                        l2_residual=full_firt_l2_residual,
                        verbose=True,
                    ),
                    name="full_feature_irt_llm_judge",
                    display_name="Full Feature-IRT (LLM Judge)",
                )
            )

    # Trajectory features predictor (Ridge regression)
    if "Trajectory" in source_by_name:
        source = source_by_name["Trajectory"]
        difficulty_predictor = FeatureBasedPredictor(
            source,
            alphas=list(config.ridge_alphas),
        )
        configs.append(
            CVPredictorConfig(
                predictor=DifficultyPredictorAdapter(difficulty_predictor),
                name="trajectory_predictor",
                display_name="Trajectory",
            )
        )

        # Full Feature-IRT with Trajectory (trains on ALL tasks like Oracle)
        if include_feature_irt:
            configs.append(
                CVPredictorConfig(
                    predictor=FullFeatureIRTAdapter(
                        source,
                        l2_weight=full_firt_l2_weight,
                        l2_residual=full_firt_l2_residual,
                        verbose=True,
                    ),
                    name="full_feature_irt_trajectory",
                    display_name="Full Feature-IRT (Trajectory)",
                )
            )

    # Grouped Ridge predictor (combines all available sources with per-source regularization)
    if len(feature_source_list) >= 2:
        # Extract just the sources (without names) for GroupedFeatureSource
        feature_sources = [source for _, source in feature_source_list]
        grouped_source = GroupedFeatureSource([
            RegularizedFeatureSource(src) for src in feature_sources
        ])

        # Check if we should expand to multiple configs for AUC-based alpha selection
        expand_grouped_ridge = getattr(config, "expand_grouped_ridge", False)
        if expand_grouped_ridge:
            # Expand to multiple fixed-alpha configs (evaluated by AUC in outer CV loop)
            configs.extend(expand_grouped_ridge_configs(grouped_source))
        else:
            # Original behavior: single GroupedRidgePredictor with MSE-based grid search
            grouped_predictor = GroupedRidgePredictor(grouped_source)
            configs.append(
                CVPredictorConfig(
                    predictor=DifficultyPredictorAdapter(grouped_predictor),
                    name="grouped_ridge",
                    display_name=f"Grouped Ridge ({grouped_source.name})",
                )
            )

        # Full Feature-IRT with grouped features (trains on ALL tasks like Oracle)
        if include_feature_irt:
            configs.append(
                CVPredictorConfig(
                    predictor=FullFeatureIRTAdapter(
                        grouped_source,
                        l2_weight=full_firt_l2_weight,
                        l2_residual=full_firt_l2_residual,
                        verbose=True,
                    ),
                    name="full_feature_irt_grouped",
                    display_name=f"Full Feature-IRT ({grouped_source.name})",
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


def expand_grouped_ridge_configs(
    grouped_source: GroupedFeatureSource,
    alpha_grids: Optional[Dict[str, List[float]]] = None,
) -> List[CVPredictorConfig]:
    """Expand grouped ridge into multiple fixed-alpha configs for AUC-based selection.

    Instead of using internal grid search (which optimizes MSE), this creates
    one predictor config per alpha combination. The outer CV loop evaluates
    each by AUC, allowing AUC-based alpha selection.

    Args:
        grouped_source: GroupedFeatureSource with 2+ underlying sources.
        alpha_grids: Per-source alpha grids. Defaults to SOURCE_ALPHA_GRIDS.

    Returns:
        List of CVPredictorConfig, one per alpha combination.
    """
    # Get per-source alpha grids
    source_grids = []
    source_names = [s.name for s in grouped_source.sources]
    for name in source_names:
        if alpha_grids and name in alpha_grids:
            source_grids.append(alpha_grids[name])
        elif name in GroupedRidgePredictor.SOURCE_ALPHA_GRIDS:
            source_grids.append(GroupedRidgePredictor.SOURCE_ALPHA_GRIDS[name])
        else:
            raise ValueError(
                f"No alpha grid for source '{name}'. "
                f"Provide alpha_grids or add to SOURCE_ALPHA_GRIDS."
            )

    configs = []
    for alpha_combo in itertools.product(*source_grids):
        fixed_alphas = dict(zip(source_names, alpha_combo))

        # Create predictor with fixed alphas (no internal grid search)
        predictor = GroupedRidgePredictor(grouped_source, fixed_alphas=fixed_alphas)

        # Create unique name for this combination
        alpha_str = "_".join(f"{v}" for v in alpha_combo)
        configs.append(
            CVPredictorConfig(
                predictor=DifficultyPredictorAdapter(predictor),
                name=f"grouped_ridge_alpha_{alpha_str}",
                display_name=predictor.name,
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
    full_firt_l2_weight: float = 0.001,
    full_firt_l2_residual: float = 0.0001,
    expansion_mode: Optional[str] = None,
    binomial_responses: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
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
        expansion_mode: Override AUC expansion method ("binary", "expand", or None)
        binomial_responses: Original binomial responses, required for expansion_mode="expand"
            when data is binary (trained on sampled data)
        diagnostics_extractors: Optional dict mapping predictor name -> extractor function.
            Each extractor is called as extractor(predictor, fold_idx) after each fold.
            Results are stored in CrossValidationResult.fold_diagnostics.

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
    # LLM judge features are auto-detected from the CSV
    predictor_configs = build_cv_predictors(
        config, root, llm_judge_features=None,  # Auto-detect from CSV
        include_feature_irt=include_feature_irt,
        full_firt_l2_weight=full_firt_l2_weight,
        full_firt_l2_residual=full_firt_l2_residual,
    )

    # Determine if we should compute binomial metrics
    compute_pass_rate_mse = spec.is_binomial

    # Results dict
    cv_results: Dict[str, CrossValidationResult] = {}

    # Run CV for each predictor using the unified framework
    for i, pc in enumerate(predictor_configs, 1):
        print(f"\n{i}. {pc.display_name}:")

        # Get diagnostics extractor for this predictor if provided
        extractor = None
        if diagnostics_extractors and pc.name in diagnostics_extractors:
            extractor = diagnostics_extractors[pc.name]

        cv_results[pc.name] = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            compute_pass_rate_mse=compute_pass_rate_mse,
            expansion_mode=expansion_mode,
            binomial_responses=binomial_responses,
            diagnostics_extractor=extractor,
        )
        result = cv_results[pc.name]
        if result.mean_auc is not None:
            print(f"   Mean AUC: {result.mean_auc:.4f} ± {result.std_auc:.4f}")
        else:
            print("   Mean AUC: N/A")

    # Post-process grouped ridge results if expanded
    expand_grouped_ridge = getattr(config, "expand_grouped_ridge", False)
    if expand_grouped_ridge:
        # Find all grouped ridge alpha variants
        grouped_results = {
            k: v for k, v in cv_results.items()
            if k.startswith("grouped_ridge_alpha_")
        }
        if grouped_results:
            # Select best by mean AUC
            best_name = max(
                grouped_results,
                key=lambda n: grouped_results[n].mean_auc or 0.0
            )
            best_result = grouped_results[best_name]

            # Remove all alpha variants from results
            for name in list(cv_results.keys()):
                if name.startswith("grouped_ridge_alpha_"):
                    del cv_results[name]

            # Add the best one back with a clear name
            cv_results["grouped_ridge_best_auc"] = best_result

            # Also update predictor_configs for summary display
            # Find the matching config for display name
            best_display_name = None
            for pc in predictor_configs:
                if pc.name == best_name:
                    best_display_name = pc.display_name
                    break

            print(f"\n=> Best Grouped Ridge by AUC: {best_display_name}")
            print(f"   Mean AUC: {best_result.mean_auc:.4f} ± {best_result.std_auc:.4f}")

            # Update predictor_configs to only include the best grouped ridge for summary
            predictor_configs = [
                pc for pc in predictor_configs
                if not pc.name.startswith("grouped_ridge_alpha_")
            ]
            # Simplify the display name - extract just the alphas part
            # e.g., "Grouped Ridge (Embedding=3000.0, LLM Judge=300.0)" -> keep as is
            predictor_configs.append(
                CVPredictorConfig(
                    predictor=None,  # Not needed for summary
                    name="grouped_ridge_best_auc",
                    display_name=best_display_name,  # Already includes "Grouped Ridge (...)"
                )
            )

    # Print summary
    print("\n" + "=" * 75)
    print(f"SUMMARY: {spec.name} ({k}-FOLD CROSS-VALIDATION)")
    print("=" * 75)

    # Sort by mean AUC descending
    display_order = [
        (pc.display_name, pc.name, cv_results[pc.name].mean_auc or 0.0)
        for pc in predictor_configs
        if pc.name in cv_results
    ]
    display_order.sort(key=lambda x: x[2], reverse=True)

    if compute_pass_rate_mse:
        print(f"\n{'Method':<55} {'Mean AUC':>10} {'Std':>8} {'Pass Rate MSE':>14}")
        print("-" * 92)

        for name, key, _ in display_order:
            result = cv_results[key]
            if result.mean_auc is not None:
                mse_str = f"{result.mean_pass_rate_mse:.4f}" if result.mean_pass_rate_mse is not None else "N/A"
                print(f"{name:<55} {result.mean_auc:>10.4f} {result.std_auc:>8.4f} {mse_str:>14}")
            else:
                print(f"{name:<55} {'N/A':>10} {'N/A':>8} {'N/A':>14}")
    else:
        print(f"\n{'Method':<55} {'Mean AUC':>10} {'Std':>8}")
        print("-" * 75)

        for name, key, _ in display_order:
            result = cv_results[key]
            if result.mean_auc is not None:
                print(f"{name:<55} {result.mean_auc:>10.4f} {result.std_auc:>8.4f}")
            else:
                print(f"{name:<55} {'N/A':>10} {'N/A':>8}")

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
        help="Include Full Feature-IRT predictors (trains on ALL tasks like Oracle)",
    )
    parser.add_argument(
        "--full_firt_l2_weight",
        type=float,
        default=0.001,
        help="L2 regularization on feature weights for Full Feature-IRT (default: 0.001)",
    )
    parser.add_argument(
        "--full_firt_l2_residual",
        type=float,
        default=0.0001,
        help="L2 regularization on residuals for Full Feature-IRT (default: 0.0001)",
    )
    parser.add_argument(
        "--expand_grouped_ridge",
        action="store_true",
        help="Use AUC-based alpha selection for grouped ridge (instead of MSE-based grid search)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use collapsed binary data (any success = 1) instead of binomial (k/n successes). "
             "Only applies to TerminalBench experiments.",
    )
    parser.add_argument(
        "--include_trajectory",
        action="store_true",
        help="Include trajectory features in evaluation (default: excluded)",
    )
    return parser


def run_experiment_main(
    config_class: Type,
    spec: ExperimentSpec,
    root: Path,
    metadata_loader_factory: Optional[Callable[[Any], Callable[[List[str]], Dict[str, Any]]]] = None,
    spec_factory: Optional[Callable[[bool], ExperimentSpec]] = None,
) -> None:
    """Shared main entry point for experiments.

    Args:
        config_class: The config class (ExperimentAConfig or TerminalBenchConfig)
        spec: Experiment specification (used if spec_factory is None)
        root: Root directory for the project
        metadata_loader_factory: Optional factory that takes config and returns a metadata loader.
            This is used by TerminalBench to create a loader that uses config.repo_path.
        spec_factory: Optional factory that takes use_binary flag and returns ExperimentSpec.
            If provided, this is used instead of the static spec parameter. This allows
            TerminalBench to dynamically choose between binomial and binary modes.
    """
    parser = create_main_parser(spec.name, str(config_class().output_dir))
    args = parser.parse_args()

    # Build config kwargs from args - only override if CLI arg is provided
    config_kwargs: Dict[str, Any] = {
        "split_seed": args.split_seed,
        "output_dir": Path(args.output_dir),
        "exclude_unsolved": args.exclude_unsolved,
        "expand_grouped_ridge": args.expand_grouped_ridge,
    }

    # Handle --binary flag for TerminalBench (default is binomial)
    if hasattr(args, "binary") and args.binary:
        config_kwargs["use_binary"] = True

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

    # Handle --include_trajectory flag (default is excluded)
    if args.include_trajectory:
        config_kwargs["trajectory_features_path"] = Path(
            "chris_output/trajectory_features/aggregated_features.csv"
        )

    config = config_class(**config_kwargs)

    # Get the appropriate spec (dynamic if factory provided, static otherwise)
    # Default is binomial (use_binary=False), --binary flag switches to binary mode
    if spec_factory is not None and hasattr(args, "binary"):
        use_binary = args.binary
        spec = spec_factory(use_binary)

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
        full_firt_l2_weight=args.full_firt_l2_weight,
        full_firt_l2_residual=args.full_firt_l2_residual,
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