"""Test progression feature extraction module.

Extracts features from test execution patterns within agent trajectories:
- Number of test runs
- Pass rate progression over time
- Blocking tests identification
- Test stability/churn metrics

Usage:
    from experiment_b.test_progression import (
        extract_test_progression,
        compute_test_progression_features,
        aggregate_test_progression_features,
        TEST_PROGRESSION_FEATURE_NAMES,
    )

Example:
    >>> import json
    >>> with open("trajectory.json") as f:
    ...     trajectory = json.load(f)
    >>> progression = extract_test_progression(trajectory)
    >>> features = compute_test_progression_features(progression)
    >>> print(f"Test runs: {features.num_test_runs}")
    >>> print(f"Pass rate improvement: {features.pass_rate_improvement}")
"""

from .types import (
    TestStatus,
    TestRun,
    TestProgression,
    TestProgressionFeatures,
)
from .parsers import (
    detect_framework,
    parse_test_output,
    extract_all_test_runs,
)
from .features import (
    extract_test_progression,
    compute_test_progression_features,
    to_feature_vector,
    to_raw_vector,
    features_to_dict,
    features_from_dict,
    TEST_PROGRESSION_FEATURE_NAMES,
)
from .aggregator import (
    aggregate_test_progression_features,
    aggregate_raw_features,
    compute_cross_agent_test_features,
    filter_features_with_test_data,
    compute_coverage_stats,
)

__all__ = [
    # Types
    "TestStatus",
    "TestRun",
    "TestProgression",
    "TestProgressionFeatures",
    # Parsers
    "detect_framework",
    "parse_test_output",
    "extract_all_test_runs",
    # Features
    "extract_test_progression",
    "compute_test_progression_features",
    "to_feature_vector",
    "to_raw_vector",
    "features_to_dict",
    "features_from_dict",
    "TEST_PROGRESSION_FEATURE_NAMES",
    # Aggregation
    "aggregate_test_progression_features",
    "aggregate_raw_features",
    "compute_cross_agent_test_features",
    "filter_features_with_test_data",
    "compute_coverage_stats",
]
