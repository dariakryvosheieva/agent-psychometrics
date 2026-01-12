"""Difficulty predictor protocol and implementations."""

from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DifficultyPredictor(Protocol):
    """Protocol for difficulty predictors.

    Any class implementing this protocol can be used to predict task difficulties
    in the Experiment A pipeline.
    """

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Train on tasks with known IRT difficulties.

        Args:
            task_ids: List of task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        ...

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        ...


class EmbeddingPredictor:
    """Difficulty predictor using pre-computed embeddings + Ridge regression.

    Based on Daria's predict_question_difficulty.py pipeline.
    Requires a pre-computed embeddings .npz file.
    """

    def __init__(
        self,
        embeddings_path: Path,
        ridge_alpha: float = 10000.0,
    ):
        """Initialize embedding predictor.

        Args:
            embeddings_path: Path to pre-computed embeddings .npz file
            ridge_alpha: Ridge regression regularization parameter
        """
        self.embeddings_path = embeddings_path
        self.ridge_alpha = ridge_alpha
        self._model: Optional[Pipeline] = None
        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._embedding_dim: Optional[int] = None

        # Load embeddings immediately
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load embeddings from .npz file."""
        data = np.load(self.embeddings_path, allow_pickle=True)

        # Extract task IDs and embedding matrix
        task_ids = [str(x) for x in data["task_ids"].tolist()]
        X = data["X"].astype(np.float32)

        self._embedding_dim = int(X.shape[1])
        self._embeddings = {task_id: X[i] for i, task_id in enumerate(task_ids)}

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Ridge regression on task embeddings.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for training tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]
        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from embeddings")

        # Build training matrix
        X = np.stack([self._embeddings[t] for t in available_tasks])
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Fit StandardScaler + Ridge
        self._model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=self.ridge_alpha)),
        ])
        self._model.fit(X, y)

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for prediction tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]

        if not available_tasks:
            return {}

        X = np.stack([self._embeddings[t] for t in available_tasks])
        preds = self._model.predict(X)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    @property
    def n_embeddings(self) -> int:
        """Return number of loaded embeddings."""
        return len(self._embeddings) if self._embeddings else 0


class ConstantPredictor:
    """Baseline: predict mean difficulty for all tasks."""

    def __init__(self):
        self._mean_b: Optional[float] = None

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Compute mean difficulty from training data.

        Args:
            task_ids: List of training task identifiers (unused)
            ground_truth_b: Array of ground truth difficulty values
        """
        self._mean_b = float(np.mean(ground_truth_b))

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict mean difficulty for all tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to mean difficulty
        """
        if self._mean_b is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return {t: self._mean_b for t in task_ids}


class GroundTruthPredictor:
    """Oracle: use actual IRT difficulties (upper bound baseline)."""

    def __init__(self, items_df: pd.DataFrame):
        """Initialize with ground truth items.

        Args:
            items_df: DataFrame with index=task_id, column 'b' for difficulty
        """
        self._items = items_df

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """No training needed for oracle.

        Args:
            task_ids: Unused
            ground_truth_b: Unused
        """
        pass  # No training needed

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Return actual IRT difficulties.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to ground truth difficulty
        """
        predictions = {}
        for t in task_ids:
            if t in self._items.index:
                predictions[t] = float(self._items.loc[t, "b"])
        return predictions
