"""Root-level shared utilities for experiments A and B.

This module provides the core abstractions for difficulty prediction:
- TaskFeatureSource: Plug-and-play interface for any feature type
- FeatureBasedPredictor: Source-agnostic Ridge regression predictor
- DifficultyPredictorBase: Abstract base for all predictors
"""

from experiment_ab_shared.dataset import (
    ExperimentData,
    BinaryExperimentData,
    BinomialExperimentData,
    load_dataset,
    load_dataset_for_fold,
    stable_split_tasks,
    filter_unsolved_tasks,
    expand_response_for_auc,
)
from experiment_ab_shared.evaluator import (
    compute_auc,
    compute_irt_probability,
    convert_numpy,
    evaluate_single_predictor,
    run_evaluation_pipeline,
    PredictorConfig,
    PredictorResult,
)
from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    ConcatenatedFeatureSource,
)
from experiment_ab_shared.feature_predictor import (
    FeatureBasedPredictor,
)
from experiment_ab_shared.predictor_base import (
    DifficultyPredictorBase,
    ConstantPredictor,
    GroundTruthPredictor,
)
from experiment_ab_shared.train_irt_split import (
    get_or_train_split_irt,
    get_split_cache_dir,
    check_cached_irt,
)

__all__ = [
    # Dataset
    "ExperimentData",
    "BinaryExperimentData",
    "BinomialExperimentData",
    "load_dataset",
    "load_dataset_for_fold",
    "stable_split_tasks",
    "filter_unsolved_tasks",
    "expand_response_for_auc",
    # Evaluator
    "compute_auc",
    "compute_irt_probability",
    "convert_numpy",
    "evaluate_single_predictor",
    "run_evaluation_pipeline",
    "PredictorConfig",
    "PredictorResult",
    # Feature sources
    "TaskFeatureSource",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "ConcatenatedFeatureSource",
    # Feature-based predictor
    "FeatureBasedPredictor",
    # Predictor base classes
    "DifficultyPredictorBase",
    "ConstantPredictor",
    "GroundTruthPredictor",
    # IRT training
    "get_or_train_split_irt",
    "get_split_cache_dir",
    "check_cached_irt",
]
