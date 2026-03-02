"""Shared utilities for Experiment A (SWE-bench and TerminalBench).

This module provides the pipeline orchestration specific to Experiment A.
"""

from experiment_a.shared.pipeline import (
    ExperimentSpec,
    CVPredictorConfig,
    build_cv_predictors,
    run_cross_validation,
)
from experiment_a.shared.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    k_fold_split_tasks,
    run_cv,
)
from experiment_a.shared.baselines import (
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
    "ExperimentSpec",
    "CVPredictorConfig",
    "build_cv_predictors",
    "run_cross_validation",
    # Cross-validation
    "CVPredictor",
    "CrossValidationResult",
    "k_fold_split_tasks",
    "run_cv",
    # Baseline predictors
    "ConstantPredictor",
    "OraclePredictor",
    "DifficultyPredictorAdapter",
    # Feature-IRT
    "JointTrainingCVPredictor",
    "feature_irt_predictor_factory",
]
