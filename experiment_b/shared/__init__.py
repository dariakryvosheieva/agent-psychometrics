"""Shared utilities for Experiment B."""

from experiment_b.shared.baseline_irt import (
    compute_baseline_irt_cache_key,
    get_or_train_baseline_irt,
)
from experiment_b.shared.config_base import DatasetConfig
from experiment_b.shared.data_splits import (
    get_all_agents_from_responses,
    identify_frontier_tasks,
    identify_nontrivial_tasks,
    split_agents_by_dates,
)
from experiment_b.shared.evaluate import (
    compute_frontier_auc,
    compute_frontier_difficulty_metrics,
    compute_scale_offset,
    load_responses_dict,
    shift_to_oracle_scale,
)
from experiment_b.shared.feature_irt_predictor import FeatureIRTPredictor

__all__ = [
    # baseline_irt
    "compute_baseline_irt_cache_key",
    "get_or_train_baseline_irt",
    # config_base
    "DatasetConfig",
    # data_splits
    "get_all_agents_from_responses",
    "identify_frontier_tasks",
    "identify_nontrivial_tasks",
    "split_agents_by_dates",
    # evaluate
    "compute_frontier_auc",
    "compute_frontier_difficulty_metrics",
    "compute_scale_offset",
    "load_responses_dict",
    "shift_to_oracle_scale",
    # feature_irt_predictor
    "FeatureIRTPredictor",
]
