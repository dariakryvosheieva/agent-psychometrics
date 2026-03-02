"""Core evaluation utilities for IRT experiments.

This module provides:
- compute_irt_probability: 1PL IRT probability formula
- convert_numpy: Recursive numpy-to-native conversion for JSON serialization
"""

from typing import Any

import numpy as np
from scipy.special import expit as sigmoid


def compute_irt_probability(theta: float, beta: float) -> float:
    """Compute IRT probability using 1PL formula.

    P(success) = sigmoid(theta - beta)

    Args:
        theta: Agent ability
        beta: Task difficulty

    Returns:
        Probability of success
    """
    return float(sigmoid(theta - beta))


def convert_numpy(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj
