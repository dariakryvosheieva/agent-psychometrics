"""Extract task-level features from test progression.

This module computes features that capture how agents iterate on test failures:
- Pass rate progression over time
- Blocking tests identification
- Test stability metrics
"""

from dataclasses import asdict
from typing import Dict, List, Set

import numpy as np

from .parsers import extract_all_test_runs
from .types import TestProgression, TestProgressionFeatures, TestRun, TestStatus


# Feature names for the feature vector
TEST_PROGRESSION_FEATURE_NAMES = [
    "num_test_runs",
    "initial_pass_rate",
    "final_pass_rate",
    "pass_rate_improvement",
    "max_pass_rate",
    "runs_until_first_improvement",
    "runs_until_max_pass_rate",
    "improvement_slope",
    "num_blocking_tests",
    "test_churn_rate",
    "avg_messages_between_runs",
    "has_granular_data",
]


def extract_test_progression(trajectory: dict) -> TestProgression:
    """Extract complete test progression from a trajectory.

    Args:
        trajectory: Loaded trajectory JSON dict

    Returns:
        TestProgression containing all test runs
    """
    task_id = trajectory.get("task_id", trajectory.get("instance_id", "unknown"))
    agent = trajectory.get("agent", "unknown")
    resolved = trajectory.get("resolved", False)
    messages = trajectory.get("messages", [])

    runs = extract_all_test_runs(messages)

    # Determine if we have granular results
    has_granular = any(len(r.individual_results) > 0 for r in runs)

    # Detect primary framework
    if runs:
        framework_counts: Dict[str, int] = {}
        for r in runs:
            framework_counts[r.framework] = framework_counts.get(r.framework, 0) + 1
        framework = max(framework_counts, key=lambda k: framework_counts[k])
    else:
        framework = "unknown"

    return TestProgression(
        task_id=task_id,
        agent=agent,
        resolved=resolved,
        runs=runs,
        has_test_output=len(runs) > 0,
        has_granular_results=has_granular,
        framework_detected=framework,
    )


def compute_test_progression_features(
    progression: TestProgression,
) -> TestProgressionFeatures:
    """Compute task-level features from test progression.

    Args:
        progression: TestProgression extracted from trajectory

    Returns:
        TestProgressionFeatures with computed values
    """
    features = TestProgressionFeatures()
    features.has_test_output = progression.has_test_output
    features.has_granular_data = progression.has_granular_results
    features.framework = progression.framework_detected

    if not progression.runs:
        return features

    runs = progression.runs
    features.num_test_runs = len(runs)

    # Pass rate progression
    pass_rates = [r.pass_rate for r in runs]
    features.initial_pass_rate = pass_rates[0]
    features.final_pass_rate = pass_rates[-1]
    features.pass_rate_improvement = features.final_pass_rate - features.initial_pass_rate
    features.max_pass_rate = max(pass_rates)

    # Runs until milestones
    features.runs_until_max_pass_rate = pass_rates.index(features.max_pass_rate) + 1

    # Find first improvement
    for i, rate in enumerate(pass_rates):
        if i > 0 and rate > pass_rates[0]:
            features.runs_until_first_improvement = i + 1
            break

    # Improvement slope (linear regression)
    if len(pass_rates) >= 2:
        x = np.arange(len(pass_rates))
        try:
            slope, _ = np.polyfit(x, pass_rates, 1)
            features.improvement_slope = float(slope)
        except (np.linalg.LinAlgError, ValueError):
            features.improvement_slope = 0.0
    else:
        features.improvement_slope = 0.0

    # Blocking tests (tests that fail on ALL runs with granular data)
    if progression.has_granular_results:
        all_test_ids: Set[str] = set()
        for run in runs:
            all_test_ids.update(run.individual_results.keys())

        blocking_tests = []
        for test_id in all_test_ids:
            # Get all appearances of this test
            appearances = [
                (r.run_index, r.individual_results.get(test_id))
                for r in runs
                if test_id in r.individual_results
            ]
            # Check if test fails on all runs where it appears
            if appearances and all(
                status in (TestStatus.FAILED, TestStatus.ERROR)
                for _, status in appearances
            ):
                blocking_tests.append(test_id)

        features.num_blocking_tests = len(blocking_tests)
        features.blocking_test_ids = blocking_tests[:10]  # Limit for storage

    # Test churn (fraction of tests that change status between consecutive runs)
    if len(runs) >= 2 and progression.has_granular_results:
        total_flips = 0
        total_comparisons = 0

        for i in range(1, len(runs)):
            prev_results = runs[i - 1].individual_results
            curr_results = runs[i].individual_results
            common_tests = set(prev_results.keys()) & set(curr_results.keys())

            for test_id in common_tests:
                total_comparisons += 1
                if prev_results[test_id] != curr_results[test_id]:
                    total_flips += 1

        if total_comparisons > 0:
            features.test_churn_rate = total_flips / total_comparisons

    # Average messages between test runs
    if len(runs) >= 2:
        gaps = [
            runs[i + 1].message_index - runs[i].message_index
            for i in range(len(runs) - 1)
        ]
        features.avg_messages_between_runs = float(np.mean(gaps))

    return features


def to_feature_vector(features: TestProgressionFeatures) -> np.ndarray:
    """Convert features to normalized vector (0-1 range).

    Args:
        features: TestProgressionFeatures to convert

    Returns:
        numpy array of normalized feature values
    """
    return np.array(
        [
            min(1.0, features.num_test_runs / 20),  # Normalize by 20
            features.initial_pass_rate,
            features.final_pass_rate,
            (features.pass_rate_improvement + 1) / 2,  # Shift from [-1,1] to [0,1]
            features.max_pass_rate,
            min(1.0, features.runs_until_first_improvement / 10),
            min(1.0, features.runs_until_max_pass_rate / 10),
            (features.improvement_slope + 0.5) / 1.0,  # Assume slope in [-0.5, 0.5]
            min(1.0, features.num_blocking_tests / 10),
            features.test_churn_rate,
            min(1.0, features.avg_messages_between_runs / 20),
            1.0 if features.has_granular_data else 0.0,
        ]
    )


def to_raw_vector(features: TestProgressionFeatures) -> np.ndarray:
    """Convert features to raw vector (unnormalized).

    Args:
        features: TestProgressionFeatures to convert

    Returns:
        numpy array of raw feature values
    """
    return np.array(
        [
            float(features.num_test_runs),
            features.initial_pass_rate,
            features.final_pass_rate,
            features.pass_rate_improvement,
            features.max_pass_rate,
            float(features.runs_until_first_improvement),
            float(features.runs_until_max_pass_rate),
            features.improvement_slope,
            float(features.num_blocking_tests),
            features.test_churn_rate,
            features.avg_messages_between_runs,
            1.0 if features.has_granular_data else 0.0,
        ]
    )


def features_to_dict(features: TestProgressionFeatures) -> Dict:
    """Convert features to dict for JSON serialization.

    Args:
        features: TestProgressionFeatures to convert

    Returns:
        Dict with all feature values
    """
    return {
        "num_test_runs": features.num_test_runs,
        "initial_pass_rate": features.initial_pass_rate,
        "final_pass_rate": features.final_pass_rate,
        "pass_rate_improvement": features.pass_rate_improvement,
        "max_pass_rate": features.max_pass_rate,
        "runs_until_first_improvement": features.runs_until_first_improvement,
        "runs_until_max_pass_rate": features.runs_until_max_pass_rate,
        "improvement_slope": features.improvement_slope,
        "num_blocking_tests": features.num_blocking_tests,
        "blocking_test_ids": features.blocking_test_ids,
        "test_churn_rate": features.test_churn_rate,
        "avg_messages_between_runs": features.avg_messages_between_runs,
        "framework": features.framework,
        "has_granular_data": features.has_granular_data,
        "has_test_output": features.has_test_output,
    }


def features_from_dict(d: Dict) -> TestProgressionFeatures:
    """Create features from dict (JSON load).

    Args:
        d: Dict with feature values

    Returns:
        TestProgressionFeatures
    """
    return TestProgressionFeatures(
        num_test_runs=d.get("num_test_runs", 0),
        initial_pass_rate=d.get("initial_pass_rate", 0.0),
        final_pass_rate=d.get("final_pass_rate", 0.0),
        pass_rate_improvement=d.get("pass_rate_improvement", 0.0),
        max_pass_rate=d.get("max_pass_rate", 0.0),
        runs_until_first_improvement=d.get("runs_until_first_improvement", 0),
        runs_until_max_pass_rate=d.get("runs_until_max_pass_rate", 0),
        improvement_slope=d.get("improvement_slope", 0.0),
        num_blocking_tests=d.get("num_blocking_tests", 0),
        blocking_test_ids=d.get("blocking_test_ids", []),
        test_churn_rate=d.get("test_churn_rate", 0.0),
        avg_messages_between_runs=d.get("avg_messages_between_runs", 0.0),
        framework=d.get("framework", "unknown"),
        has_granular_data=d.get("has_granular_data", False),
        has_test_output=d.get("has_test_output", False),
    )
