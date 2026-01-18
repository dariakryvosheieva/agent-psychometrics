"""Common framework for Experiment A across different datasets.

This module provides:
- Abstract dataset interface for binary (SWE-bench) and binomial (TerminalBench) data
- Unified evaluator that loops through difficulty predictors
- Common AUC computation with proper IRT formulas
- Generic baselines that work with any dataset type
- Cross-validation utilities for k-fold CV
"""

from experiment_a_common.dataset import (
    ExperimentData,
    BinaryExperimentData,
    BinomialExperimentData,
    load_dataset,
    load_dataset_for_fold,
    stable_split_tasks,
)
from experiment_a_common.evaluator import (
    compute_auc,
    compute_irt_probability,
    convert_numpy,
    evaluate_single_predictor,
    run_evaluation_pipeline,
    PredictorConfig,
    PredictorResult,
)
from experiment_a_common.baselines import (
    agent_only_baseline,
    random_baseline,
    verify_random_baseline_sanity,
)
from experiment_a_common.cross_validation import (
    CrossValidationResult,
    k_fold_split_tasks,
    run_cv_for_predictor,
    run_cv_for_baseline,
)

__all__ = [
    # Dataset
    "ExperimentData",
    "BinaryExperimentData",
    "BinomialExperimentData",
    "load_dataset",
    "load_dataset_for_fold",
    "stable_split_tasks",
    # Evaluator
    "compute_auc",
    "compute_irt_probability",
    "convert_numpy",
    "evaluate_single_predictor",
    "run_evaluation_pipeline",
    "PredictorConfig",
    "PredictorResult",
    # Baselines
    "agent_only_baseline",
    "random_baseline",
    "verify_random_baseline_sanity",
    # Cross-validation
    "CrossValidationResult",
    "k_fold_split_tasks",
    "run_cv_for_predictor",
    "run_cv_for_baseline",
]
