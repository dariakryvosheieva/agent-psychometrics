"""Shared utilities for Experiment A (SWE-bench and TerminalBench).

This module provides the pipeline orchestration specific to Experiment A.
"""

from experiment_a.shared.pipeline import (
    CVPredictorConfig,
    build_cv_predictors,
    cross_validate_all_predictors,
)
from experiment_a.shared.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    k_fold_split_tasks,
    evaluate_predictor_cv,
)
from experiment_a.shared.difficulty_predictors import (
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.shared.feature_irt import (
    JointTrainingCVPredictor,
    feature_irt_predictor_factory,
)

__all__ = [
    # Pipeline
    "CVPredictorConfig",
    "build_cv_predictors",
    "cross_validate_all_predictors",
    # Cross-validation
    "CVPredictor",
    "CrossValidationResult",
    "k_fold_split_tasks",
    "evaluate_predictor_cv",
    # Difficulty predictors
    "ConstantPredictor",
    "OraclePredictor",
    "DifficultyPredictorAdapter",
    # Feature-IRT
    "JointTrainingCVPredictor",
    "feature_irt_predictor_factory",
]
