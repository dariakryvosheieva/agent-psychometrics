"""Experiment A: Prior Validation (IRT AUC).

Evaluates how well a difficulty predictor can predict agent success on held-out
tasks using the 1PL IRT model.

Entry point:
- python -m experiment_new_tasks.run_all_datasets
"""

# Re-export core classes for convenience
from experiment_new_tasks.feature_predictor import (
    DifficultyPredictorBase,
    FeatureBasedPredictor,
)
from experiment_new_tasks.feature_source import (
    EmbeddingFeatureSource,
    CSVFeatureSource,
)

from experiment_new_tasks.pipeline import (
    CVPredictorConfig,
    build_cv_predictors,
    cross_validate_all_predictors,
)
from experiment_new_tasks.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    k_fold_split_tasks,
    evaluate_predictor_cv,
)
from experiment_new_tasks.difficulty_predictors import (
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)
from experiment_new_tasks.feature_irt import (
    JointTrainingCVPredictor,
    feature_irt_predictor_factory,
)

__all__ = [
    # Core classes
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
