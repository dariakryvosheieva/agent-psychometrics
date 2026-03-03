"""Experiment A: Prior Validation (IRT AUC).

Evaluates how well a difficulty predictor can predict agent success on held-out
tasks using the 1PL IRT model.

Entry point:
- python -m experiment_a.run_all_datasets
"""

# Re-export core classes from shared modules for convenience
from experiment_ab_shared import (
    DifficultyPredictorBase,
    FeatureBasedPredictor,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    stable_split_tasks,
)

from experiment_a.shared import (
    CVPredictor,
    CrossValidationResult,
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
    JointTrainingCVPredictor,
    evaluate_predictor_cv,
    k_fold_split_tasks,
)

__all__ = [
    # From experiment_ab_shared
    "DifficultyPredictorBase",
    "FeatureBasedPredictor",
    "EmbeddingFeatureSource",
    "CSVFeatureSource",
    "stable_split_tasks",
    # From experiment_a.shared
    "CVPredictor",
    "CrossValidationResult",
    "ConstantPredictor",
    "OraclePredictor",
    "DifficultyPredictorAdapter",
    "JointTrainingCVPredictor",
    "evaluate_predictor_cv",
    "k_fold_split_tasks",
]
