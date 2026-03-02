"""Shared pipeline for running Experiment A across different datasets.

This module provides the common evaluation pipeline that all datasets use.
The experiments differ only in:
- Dataset name
- Response type (binary vs binomial)
- IRT cache directory
- Feature paths (embeddings, LLM judge CSVs)
- Metadata loading (TerminalBench loads task data from repo)
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

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
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)
@dataclass
class ExperimentSpec:
    """Specification for an experiment.

    Attributes:
        name: Human-readable experiment name (e.g., "SWE-bench", "TerminalBench")
        is_binomial: Whether responses are binomial (True) or binary (False)
        irt_cache_dir: Directory for caching fold-specific IRT models
    """

    name: str
    is_binomial: bool
    irt_cache_dir: Path


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


def _default_predictor_factory(source_name: str, source: Any, config: Any) -> CVPredictor:
    """Default predictor factory: Ridge regression with DifficultyPredictorAdapter.

    Args:
        source_name: One of "Embedding", "LLM Judge", or "Grouped".
        source: Feature source object.
        config: Experiment config (used for ridge_alphas).

    Returns:
        CVPredictor wrapping Ridge regression.
    """
    if source_name == "Grouped":
        return DifficultyPredictorAdapter(GroupedRidgePredictor(source))
    return DifficultyPredictorAdapter(
        FeatureBasedPredictor(source, alphas=list(config.ridge_alphas))
    )


def build_cv_predictors(
    config: Any,
    root: Path,
    extra_embeddings_paths: Optional[List[Tuple[str, Path]]] = None,
    extra_llm_judge_paths: Optional[List[Tuple[str, Path]]] = None,
    predictor_factory: Optional[Callable[[str, Any, Any], CVPredictor]] = None,
) -> List[CVPredictorConfig]:
    """Build list of CVPredictor configurations for cross-validation.

    All predictors implement the CVPredictor protocol (fit/predict_probability).
    LLM judge feature columns are auto-detected from the CSV.

    Args:
        config: Experiment configuration (ExperimentAConfig or TerminalBenchConfig)
        root: Root directory for resolving relative paths
        extra_embeddings_paths: List of (name, path) tuples for additional embedding
            sources to compare (ablation study).
        extra_llm_judge_paths: List of (name, path) tuples for additional LLM judge
            sources to compare (ablation study).
        predictor_factory: Optional callable(source_name, source, config) -> CVPredictor.
            Controls how feature sources are wrapped into predictors. If None, uses
            Ridge regression (the default). source_name is one of "Embedding",
            "LLM Judge", or "Grouped". Naming is handled by this function, not the factory.

    Returns:
        List of CVPredictorConfig objects with pre-instantiated predictors.
    """
    if predictor_factory is None:
        predictor_factory = _default_predictor_factory

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

    # Build feature sources (Embedding + LLM Judge only)
    feature_source_list = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        verbose=False,
    )

    # Build a dict for easy lookup by source name
    source_by_name = {name: source for name, source in feature_source_list}

    # Add extra embedding sources (for ablation studies)
    if extra_embeddings_paths:
        for name, path in extra_embeddings_paths:
            emb_source = EmbeddingFeatureSource(path)
            predictor = predictor_factory("Embedding", emb_source, config)
            configs.append(
                CVPredictorConfig(
                    predictor=predictor,
                    name=f"embedding_{name}",
                    display_name=f"Embedding ({name})",
                )
            )

    # Add extra LLM judge sources (for ablation studies)
    if extra_llm_judge_paths:
        for name, path in extra_llm_judge_paths:
            judge_source = CSVFeatureSource(path, feature_cols=None)  # Auto-detect cols
            predictor = predictor_factory("LLM Judge", judge_source, config)
            configs.append(
                CVPredictorConfig(
                    predictor=predictor,
                    name=f"llm_judge_{name}",
                    display_name=f"LLM Judge ({name})",
                )
            )

    # Individual feature source predictors
    if "Embedding" in source_by_name:
        predictor = predictor_factory("Embedding", source_by_name["Embedding"], config)
        configs.append(
            CVPredictorConfig(predictor=predictor, name="embedding", display_name="Embedding")
        )

    if "LLM Judge" in source_by_name:
        predictor = predictor_factory("LLM Judge", source_by_name["LLM Judge"], config)
        configs.append(
            CVPredictorConfig(predictor=predictor, name="llm_judge", display_name="LLM Judge")
        )

    # Grouped predictor (Embedding + LLM Judge with per-source regularization)
    if "Embedding" in source_by_name and "LLM Judge" in source_by_name:
        grouped_source = GroupedFeatureSource([
            RegularizedFeatureSource(source_by_name["Embedding"]),
            RegularizedFeatureSource(source_by_name["LLM Judge"]),
        ])
        predictor = predictor_factory("Grouped", grouped_source, config)
        configs.append(
            CVPredictorConfig(
                predictor=predictor,
                name="grouped",
                display_name=f"Grouped ({grouped_source.name})",
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

    return configs


def _run_single_predictor(
    pc: CVPredictorConfig,
    folds: List,
    load_fold_data: Callable,
    compute_pass_rate_mse: bool,
    expansion_mode: Optional[str],
    binomial_responses: Optional[Dict[str, Dict[str, Dict[str, int]]]],
    diagnostics_extractor: Optional[Callable],
    n_jobs_folds: int,
) -> Tuple[str, CrossValidationResult]:
    """Run CV for a single predictor. Helper for parallel execution."""
    import copy
    # Deep copy predictor to avoid state conflicts in parallel execution
    predictor_copy = copy.deepcopy(pc.predictor)

    result = run_cv(
        predictor_copy,
        folds,
        load_fold_data,
        verbose=False,  # Disable verbose in parallel mode
        compute_pass_rate_mse=compute_pass_rate_mse,
        expansion_mode=expansion_mode,
        binomial_responses=binomial_responses,
        diagnostics_extractor=diagnostics_extractor,
        n_jobs=n_jobs_folds,
    )
    return pc.name, result


def run_cross_validation(
    config: Any,
    spec: ExperimentSpec,
    root: Path,
    k: int = 5,
    metadata_loader: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
    expansion_mode: Optional[str] = None,
    binomial_responses: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
    n_jobs_methods: int = 1,
    n_jobs_folds: int = 1,
    extra_embeddings_paths: Optional[List[Tuple[str, Path]]] = None,
    extra_llm_judge_paths: Optional[List[Tuple[str, Path]]] = None,
    predictor_factory: Optional[Callable[[str, Any, Any], CVPredictor]] = None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline with k-fold cross-validation.

    Uses the unified run_cv function for ALL predictors including baselines.
    Supports parallelization at two levels: across methods and across folds.

    Args:
        config: Experiment configuration
        spec: Experiment specification
        root: Root directory for resolving relative paths
        k: Number of folds
        metadata_loader: Optional callable to load task metadata
        expansion_mode: Override AUC expansion method ("binary", "expand", or None)
        binomial_responses: Original binomial responses, required for expansion_mode="expand"
            when data is binary (trained on sampled data)
        diagnostics_extractors: Optional dict mapping predictor name -> extractor function.
            Each extractor is called as extractor(predictor, fold_idx) after each fold.
            Results are stored in CrossValidationResult.fold_diagnostics.
        n_jobs_methods: Number of parallel jobs for method execution. 1 = sequential.
        n_jobs_folds: Number of parallel jobs for fold execution within each method.
        extra_embeddings_paths: List of (name, path) tuples for additional embedding
            sources to compare (ablation study).
        extra_llm_judge_paths: List of (name, path) tuples for additional LLM judge
            sources to compare (ablation study).
        predictor_factory: Optional callable(source_name, source, config) -> CVPredictor.
            Controls how feature sources become predictors. If None, uses Ridge regression.
            Passed through to build_cv_predictors().

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

    # Build predictor configs
    predictor_configs = build_cv_predictors(
        config, root,
        extra_embeddings_paths=extra_embeddings_paths,
        extra_llm_judge_paths=extra_llm_judge_paths,
        predictor_factory=predictor_factory,
    )

    # Determine if we should compute binomial metrics
    compute_pass_rate_mse = spec.is_binomial

    # Results dict
    cv_results: Dict[str, CrossValidationResult] = {}

    # Run CV for each predictor using the unified framework
    if n_jobs_methods == 1:
        # Sequential execution (original behavior)
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
                n_jobs=n_jobs_folds,
            )
            result = cv_results[pc.name]
            if result.mean_auc is not None:
                print(f"   Mean AUC: {result.mean_auc:.4f} ± {result.std_auc:.4f}")
            else:
                print("   Mean AUC: N/A")
    else:
        # Parallel execution across methods
        from joblib import Parallel, delayed

        print(f"\nRunning {len(predictor_configs)} predictors in parallel (n_jobs={n_jobs_methods})...")

        # Build list of (config, extractor) tuples for parallel execution
        predictor_tasks = []
        for pc in predictor_configs:
            extractor = None
            if diagnostics_extractors and pc.name in diagnostics_extractors:
                extractor = diagnostics_extractors[pc.name]
            predictor_tasks.append((pc, extractor))

        results = Parallel(n_jobs=n_jobs_methods)(
            delayed(_run_single_predictor)(
                pc, folds, load_fold_data, compute_pass_rate_mse,
                expansion_mode, binomial_responses, extractor, n_jobs_folds
            )
            for pc, extractor in predictor_tasks
        )

        # Collect results
        for name, result in results:
            cv_results[name] = result

        # Print results after parallel execution
        for pc in predictor_configs:
            result = cv_results[pc.name]
            if result.mean_auc is not None:
                print(f"   {pc.display_name}: AUC = {result.mean_auc:.4f} ± {result.std_auc:.4f}")
            else:
                print(f"   {pc.display_name}: AUC = N/A")

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
        "--embeddings_paths",
        type=str,
        default=None,
        help="Comma-separated list of embedding paths to compare (ablation study)",
    )
    parser.add_argument(
        "--llm_judge_features_path",
        type=str,
        default=None,
        help="Path to LLM judge features CSV file",
    )
    parser.add_argument(
        "--llm_judge_paths",
        type=str,
        default=None,
        help="Comma-separated list of LLM judge paths to compare (ablation study)",
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
        "--binary",
        action="store_true",
        help="Use collapsed binary data (any success = 1) instead of binomial (k/n successes). "
             "Only applies to TerminalBench experiments.",
    )
    parser.add_argument(
        "--coefficients",
        action="store_true",
        help="Extract and display LLM Judge Ridge coefficients (Table 10 / Figure 3).",
    )
    return parser


def run_experiment_main(
    dataset: str,
    spec: ExperimentSpec,
    root: Path,
    config_class: Optional[Type] = None,
    metadata_loader_factory: Optional[Callable[[Any], Callable[[List[str]], Dict[str, Any]]]] = None,
    spec_factory: Optional[Callable[[bool], ExperimentSpec]] = None,
) -> None:
    """Shared main entry point for experiments.

    Args:
        dataset: Dataset short name (e.g., "swebench", "gso", "terminalbench").
            Used to look up default config values via ExperimentAConfig.for_dataset().
        spec: Experiment specification (used if spec_factory is None)
        root: Root directory for the project
        config_class: Config class to use. If None, uses ExperimentAConfig.
            TerminalBench passes TerminalBenchConfig for binary mode support.
        metadata_loader_factory: Optional factory that takes config and returns a metadata loader.
            This is used by TerminalBench to create a loader that uses config.repo_path.
        spec_factory: Optional factory that takes use_binary flag and returns ExperimentSpec.
            If provided, this is used instead of the static spec parameter. This allows
            TerminalBench to dynamically choose between binomial and binary modes.
    """
    from experiment_a.shared.config import ExperimentAConfig as _DefaultConfig

    if config_class is None:
        config_class = _DefaultConfig

    default_config = config_class.for_dataset(dataset)
    parser = create_main_parser(spec.name, str(default_config.output_dir))
    args = parser.parse_args()

    # Build config kwargs from args - only override if CLI arg is provided
    config_kwargs: Dict[str, Any] = {
        "split_seed": args.split_seed,
        "output_dir": Path(args.output_dir),
        "exclude_unsolved": args.exclude_unsolved,
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

    config = config_class.for_dataset(dataset, **config_kwargs)

    # Get the appropriate spec (dynamic if factory provided, static otherwise)
    # Default is binomial (use_binary=False), --binary flag switches to binary mode
    if spec_factory is not None and hasattr(args, "binary"):
        use_binary = args.binary
        spec = spec_factory(use_binary)

    # Parse extra feature paths for ablation studies
    extra_embeddings_paths: Optional[List[Tuple[str, Path]]] = None
    extra_llm_judge_paths: Optional[List[Tuple[str, Path]]] = None

    if args.embeddings_paths:
        extra_embeddings_paths = []
        for path_str in args.embeddings_paths.split(","):
            path_str = path_str.strip()
            if path_str:
                path = root / Path(path_str)
                # Extract a short name from the path
                name = Path(path_str).stem
                # If it has double underscores, use the last part as name
                if "__" in name:
                    parts = name.split("__")
                    # Try to find a meaningful part (e.g., "no_solution")
                    name = parts[-1] if parts[-1] else parts[-2]
                extra_embeddings_paths.append((name, path))

    if args.llm_judge_paths:
        extra_llm_judge_paths = []
        for path_str in args.llm_judge_paths.split(","):
            path_str = path_str.strip()
            if path_str:
                path = root / Path(path_str)
                # Use parent directory name as the variant name
                name = Path(path_str).parent.name
                extra_llm_judge_paths.append((name, path))

    if args.dry_run:
        print("DRY RUN - Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        if extra_embeddings_paths:
            print(f"\nExtra embedding paths: {extra_embeddings_paths}")
        if extra_llm_judge_paths:
            print(f"Extra LLM judge paths: {extra_llm_judge_paths}")
        return

    # Create metadata loader if factory provided
    metadata_loader = None
    if metadata_loader_factory is not None:
        metadata_loader = metadata_loader_factory(config)

    # Set up coefficient extraction if requested
    diagnostics_extractors = None
    if args.coefficients:
        from experiment_a.shared.coefficient_analysis import make_llm_coef_extractor
        diagnostics_extractors = make_llm_coef_extractor()

    # Run cross-validation
    results = run_cross_validation(
        config, spec, root, args.k_folds, metadata_loader,
        extra_embeddings_paths=extra_embeddings_paths,
        extra_llm_judge_paths=extra_llm_judge_paths,
        diagnostics_extractors=diagnostics_extractors,
    )

    # Print coefficient analysis if requested
    if args.coefficients:
        from experiment_a.shared.coefficient_analysis import (
            print_coefficient_table,
            save_coefficient_bar_chart,
        )
        cv_results = results.get("cv_results", {})
        llm_result = cv_results.get("llm_judge")
        if llm_result is not None:
            fold_diagnostics = llm_result.get("fold_diagnostics", [])
            coeffs = [d for d in fold_diagnostics if d is not None]
            if coeffs:
                print(f"\n{'=' * 80}")
                print("LLM JUDGE COEFFICIENT ANALYSIS")
                print(f"{'=' * 80}")
                print_coefficient_table(coeffs)

                output_dir = root / config.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                save_coefficient_bar_chart(
                    coeffs,
                    output_dir / "coefficient_bar_chart.png",
                    title=f"Mean Coefficient Magnitude by Feature Source ({spec.name})",
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
