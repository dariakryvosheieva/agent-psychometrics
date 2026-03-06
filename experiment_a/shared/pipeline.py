"""Shared pipeline for running Experiment A across different datasets.

This module provides the common evaluation pipeline that all datasets use.
The experiments differ only in:
- Dataset name
- IRT cache directory
- Feature paths (embeddings, LLM judge CSVs)
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from experiment_ab_shared.feature_source import (
    RegularizedFeatureSource,
    GroupedFeatureSource,
    build_feature_sources,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
    GroupedRidgePredictor,
)
from experiment_ab_shared import (
    load_dataset_for_fold,
    convert_numpy,
    filter_unsolved_tasks,
)
from experiment_ab_shared.dataset import (
    _load_binary_responses,
)

from experiment_a.shared.cross_validation import (
    k_fold_split_tasks,
    evaluate_predictor_cv,
    CrossValidationResult,
)
from experiment_a.shared.difficulty_predictors import (
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)
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
    predictor_factory: Optional[Callable[[str, Any, Any], CVPredictor]] = None,
) -> List[CVPredictorConfig]:
    """Build list of CVPredictor configurations for cross-validation.

    All predictors implement the CVPredictor protocol (fit/predict_probability).
    LLM judge feature columns are auto-detected from the CSV.

    Args:
        config: Experiment configuration (ExperimentAConfig)
        root: Root directory for resolving relative paths
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


def cross_validate_all_predictors(
    config: Any,
    root: Path,
    k: int = 5,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
    predictor_factory: Optional[Callable[[str, Any, Any], CVPredictor]] = None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline with k-fold cross-validation.

    Uses the unified run_cv function for ALL predictors including baselines.

    Args:
        config: Experiment configuration (ExperimentAConfig with display_name,
            irt_cache_dir, and all data paths)
        root: Root directory for resolving relative paths
        k: Number of folds
        diagnostics_extractors: Optional dict mapping predictor name -> extractor function.
            Each extractor is called as extractor(predictor, fold_idx) after each fold.
            Results are stored in CrossValidationResult.fold_diagnostics.
        predictor_factory: Optional callable(source_name, source, config) -> CVPredictor.
            Controls how feature sources become predictors. If None, uses Ridge regression.
            Passed through to build_cv_predictors().

    Returns:
        Dict with CV results for each method
    """
    print("=" * 60)
    print(f"EXPERIMENT A: {k}-FOLD CROSS-VALIDATION - {config.display_name}")
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
        responses = _load_binary_responses(responses_path)
        all_task_ids, n_excluded = filter_unsolved_tasks(all_task_ids, responses)
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
            irt_cache_dir=root / config.irt_cache_dir,
            exclude_unsolved=config.exclude_unsolved,
        )

    # Build predictor configs
    predictor_configs = build_cv_predictors(
        config, root,
        predictor_factory=predictor_factory,
    )

    # Results dict
    cv_results: Dict[str, CrossValidationResult] = {}

    # Run CV for each predictor using the unified framework
    for i, pc in enumerate(predictor_configs, 1):
        print(f"\n{i}. {pc.display_name}:")

        # Get diagnostics extractor for this predictor if provided
        extractor = None
        if diagnostics_extractors and pc.name in diagnostics_extractors:
            extractor = diagnostics_extractors[pc.name]

        cv_results[pc.name] = evaluate_predictor_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            diagnostics_extractor=extractor,
        )
        result = cv_results[pc.name]
        if result.mean_auc is not None:
            print(f"   Mean AUC: {result.mean_auc:.4f} ± {result.std_auc:.4f}")
        else:
            print("   Mean AUC: N/A")

    # Print summary
    print("\n" + "=" * 75)
    print(f"SUMMARY: {config.display_name} ({k}-FOLD CROSS-VALIDATION)")
    print("=" * 75)

    # Sort by mean AUC descending
    display_order = [
        (pc.display_name, pc.name, cv_results[pc.name].mean_auc or 0.0)
        for pc in predictor_configs
        if pc.name in cv_results
    ]
    display_order.sort(key=lambda x: x[2], reverse=True)

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