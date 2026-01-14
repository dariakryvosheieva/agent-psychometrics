"""Aggregate test progression features across multiple agents for a task.

This module provides functions to:
- Aggregate features across agents (mean of feature vectors)
- Compute cross-agent features (variance, agreement metrics)
"""

from typing import Dict, List, Set

import numpy as np

from .features import (
    TEST_PROGRESSION_FEATURE_NAMES,
    to_feature_vector,
    to_raw_vector,
)
from .types import TestProgressionFeatures


def aggregate_test_progression_features(
    agent_features: Dict[str, TestProgressionFeatures],
) -> np.ndarray:
    """Aggregate features across agents (mean of feature vectors).

    Args:
        agent_features: Dict mapping agent -> TestProgressionFeatures

    Returns:
        Aggregated feature vector of length len(TEST_PROGRESSION_FEATURE_NAMES)
    """
    if not agent_features:
        return np.zeros(len(TEST_PROGRESSION_FEATURE_NAMES))

    vectors = [to_feature_vector(f) for f in agent_features.values()]
    return np.mean(vectors, axis=0)


def aggregate_raw_features(
    agent_features: Dict[str, TestProgressionFeatures],
) -> np.ndarray:
    """Aggregate raw (unnormalized) features across agents.

    Args:
        agent_features: Dict mapping agent -> TestProgressionFeatures

    Returns:
        Aggregated raw feature vector
    """
    if not agent_features:
        return np.zeros(len(TEST_PROGRESSION_FEATURE_NAMES))

    vectors = [to_raw_vector(f) for f in agent_features.values()]
    return np.mean(vectors, axis=0)


def compute_cross_agent_test_features(
    agent_features: Dict[str, TestProgressionFeatures],
) -> Dict[str, float]:
    """Compute features that look across agents on same task.

    These features capture agreement/variance across agents:
    - Do different agents converge to similar pass rates?
    - Do they agree on which tests are blocking?
    - How consistent are improvement slopes?

    Args:
        agent_features: Dict mapping agent -> TestProgressionFeatures

    Returns:
        Dict of cross-agent feature values
    """
    if len(agent_features) < 2:
        return {
            "final_pass_rate_std": 0.0,
            "improvement_slope_std": 0.0,
            "blocking_test_agreement": 0.0,
            "num_runs_std": 0.0,
            "max_pass_rate_std": 0.0,
        }

    features_list = list(agent_features.values())

    # Standard deviations of key metrics
    final_rates = [f.final_pass_rate for f in features_list]
    slopes = [f.improvement_slope for f in features_list]
    num_runs = [f.num_test_runs for f in features_list]
    max_rates = [f.max_pass_rate for f in features_list]

    # Blocking test agreement (Jaccard similarity of blocking tests)
    blocking_sets = [
        set(f.blocking_test_ids) for f in features_list if f.blocking_test_ids
    ]
    if len(blocking_sets) >= 2:
        union = set.union(*blocking_sets)
        intersection = set.intersection(*blocking_sets)
        agreement = len(intersection) / len(union) if union else 0.0
    else:
        agreement = 0.0

    return {
        "final_pass_rate_std": float(np.std(final_rates)),
        "improvement_slope_std": float(np.std(slopes)),
        "blocking_test_agreement": agreement,
        "num_runs_std": float(np.std(num_runs)),
        "max_pass_rate_std": float(np.std(max_rates)),
    }


def filter_features_with_test_data(
    agent_features: Dict[str, TestProgressionFeatures],
    min_test_runs: int = 1,
    require_granular: bool = False,
) -> Dict[str, TestProgressionFeatures]:
    """Filter to only agents with sufficient test data.

    Args:
        agent_features: Dict mapping agent -> TestProgressionFeatures
        min_test_runs: Minimum number of test runs required
        require_granular: If True, require has_granular_data=True

    Returns:
        Filtered dict with only qualifying agents
    """
    filtered = {}
    for agent, features in agent_features.items():
        if features.num_test_runs < min_test_runs:
            continue
        if require_granular and not features.has_granular_data:
            continue
        filtered[agent] = features
    return filtered


def compute_coverage_stats(
    agent_features: Dict[str, TestProgressionFeatures],
) -> Dict[str, float]:
    """Compute coverage statistics for a set of features.

    Args:
        agent_features: Dict mapping agent -> TestProgressionFeatures

    Returns:
        Dict with coverage statistics
    """
    if not agent_features:
        return {
            "total_agents": 0,
            "has_test_output_rate": 0.0,
            "has_granular_rate": 0.0,
            "avg_test_runs": 0.0,
            "multi_run_rate": 0.0,  # % with >= 2 runs
        }

    features_list = list(agent_features.values())
    n = len(features_list)

    has_test = sum(1 for f in features_list if f.has_test_output)
    has_granular = sum(1 for f in features_list if f.has_granular_data)
    multi_run = sum(1 for f in features_list if f.num_test_runs >= 2)
    total_runs = sum(f.num_test_runs for f in features_list)

    return {
        "total_agents": n,
        "has_test_output_rate": has_test / n,
        "has_granular_rate": has_granular / n,
        "avg_test_runs": total_runs / n,
        "multi_run_rate": multi_run / n,
    }
