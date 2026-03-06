"""Experiment A: Prior Validation (IRT AUC).

Evaluates how well a difficulty predictor can predict agent success on held-out
tasks using the 1PL IRT model.

Entry point:
- python -m experiment_a.run_all_datasets
"""

# Re-export core classes for convenience
from experiment_ab_shared import (
    DifficultyPredictorBase,
    FeatureBasedPredictor,
    EmbeddingFeatureSource,
    CSVFeatureSource,
)

from experiment_a.pipeline import (
    CVPredictorConfig,
    build_cv_predictors,
    cross_validate_all_predictors,
)
from experiment_a.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    k_fold_split_tasks,
    evaluate_predictor_cv,
)
from experiment_a.difficulty_predictors import (
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.feature_irt import (
    JointTrainingCVPredictor,
    feature_irt_predictor_factory,
)

__all__ = [
    # From experiment_ab_shared
    "DifficultyPredictorBase",
    "FeatureBasedPredictor",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    # Pipeline
    "CVPredictorConfig",
    "build_cv_predictors",
    "cross_validate_all_predictors",
    # Cross-validation
    "CVPredictor",
    "CrossValidationResult",
    "ConstantPredictor",
    "OraclePredictor",
    "DifficultyPredictorAdapter",
    # Feature-IRT
    "JointTrainingCVPredictor",
    "feature_irt_predictor_factory",
    "evaluate_predictor_cv",
    "k_fold_split_tasks",
]
