"""Shared utilities for Experiment A (SWE-bench and TerminalBench).

This module provides the pipeline orchestration specific to Experiment A.
"""

from experiment_a.shared.pipeline import (
    ExperimentSpec,
    CVPredictorConfig,
    build_cv_predictors,
    run_cross_validation,
    create_main_parser,
    run_experiment_main,
    SWEBENCH_LLM_JUDGE_FEATURES,
)
from experiment_a.shared.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    k_fold_split_tasks,
    run_cv,
)
from experiment_a.shared.baselines import (
    AgentOnlyPredictor,
    ConstantPredictor,
    OraclePredictor,
    DifficultyPredictorAdapter,
)

__all__ = [
    # Pipeline
    "ExperimentSpec",
    "CVPredictorConfig",
    "build_cv_predictors",
    "run_cross_validation",
    "create_main_parser",
    "run_experiment_main",
    "SWEBENCH_LLM_JUDGE_FEATURES",
    # Cross-validation
    "CVPredictor",
    "CrossValidationResult",
    "k_fold_split_tasks",
    "run_cv",
    # Baseline predictors
    "AgentOnlyPredictor",
    "ConstantPredictor",
    "OraclePredictor",
    "DifficultyPredictorAdapter",
]
