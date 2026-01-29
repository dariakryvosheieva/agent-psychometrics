"""Environment feature extraction via Inspect AI sandbox execution.

This module extracts deterministic features from SWE-bench task environments
by running bash commands inside Docker containers.

Usage:
    # Run consistency test on 2 tasks
    inspect eval experiment_a/env_features/inspect_task.py --limit 2 \\
        --log-dir chris_output/env_features/validation_run1/

    # Run full extraction with batching
    python -m experiment_a.env_features.run_extraction

See README.md for full documentation.
"""

from experiment_a.env_features.feature_definitions import (
    FEATURE_DEFINITIONS,
    FEATURE_BY_NAME,
    FeatureDefinition,
    get_feature_names,
)

__all__ = [
    "FEATURE_DEFINITIONS",
    "FEATURE_BY_NAME",
    "FeatureDefinition",
    "get_feature_names",
]
