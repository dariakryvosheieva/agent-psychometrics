"""Base class for difficulty predictors.

This module defines the abstract base class that difficulty predictors
implement. The interface is: fit on tasks with known difficulties,
then predict difficulties for new tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class DifficultyPredictorBase(ABC):
    """Abstract base class for all difficulty predictors.

    All predictors must implement:
    - fit(): Train on tasks with known difficulties
    - predict(): Predict difficulties for new tasks
    - name: Human-readable predictor name
    """

    @abstractmethod
    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Train on tasks with known IRT difficulties.

        Args:
            task_ids: List of task identifiers
            ground_truth_b: Array of ground truth difficulty values (b parameters)
        """
        ...

    @abstractmethod
    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable predictor name."""
        ...
