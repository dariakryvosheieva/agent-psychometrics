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
    compute_irt_probability,
    convert_numpy,
)
from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    RegularizedFeatureSource,
    GroupedFeatureSource,
)
from experiment_ab_shared.feature_predictor import (
    DifficultyPredictorBase,
    FeatureBasedPredictor,
    GroupedRidgePredictor,
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
    "compute_irt_probability",
    "convert_numpy",
    # Feature sources
    "TaskFeatureSource",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "RegularizedFeatureSource",
    "GroupedFeatureSource",
    # Feature-based predictors
    "FeatureBasedPredictor",
    "GroupedRidgePredictor",
    # Predictor base classes
    "DifficultyPredictorBase",
    # IRT training
    "get_or_train_split_irt",
    "get_split_cache_dir",
    "check_cached_irt",
]
