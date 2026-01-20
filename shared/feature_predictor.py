"""Unified feature-based difficulty predictor.

This module provides a single predictor class that works with any TaskFeatureSource.
The predictor uses StandardScaler + RidgeCV regression, which is sufficient for
both high-dimensional embeddings and low-dimensional semantic features.

Example usage:
    from shared.feature_source import EmbeddingFeatureSource
    from shared.feature_predictor import FeatureBasedPredictor

    # Create feature source
    source = EmbeddingFeatureSource(Path("embeddings.npz"))

    # Create predictor
    predictor = FeatureBasedPredictor(source)

    # Fit on training data
    predictor.fit(train_task_ids, train_difficulties)

    # Predict on test data
    predictions = predictor.predict(test_task_ids)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
)


class FeatureBasedPredictor:
    """Difficulty predictor using any TaskFeatureSource + Ridge regression.

    Pipeline: features -> StandardScaler -> RidgeCV -> predict

    This predictor is source-agnostic: it works identically for embeddings,
    LLM judge features, or any other feature type. The only requirements are:
    1. A TaskFeatureSource that provides features for tasks
    2. Ground truth difficulty values for training

    Attributes:
        source: The TaskFeatureSource providing features.
        name: Human-readable name (from source.name).
        ridge_alphas: List of alpha values for RidgeCV.
    """

    # Default Ridge alphas spanning 5 orders of magnitude
    # Note: Starting from 0.1 (not 0.01) to avoid ill-conditioned matrices
    # with high-dimensional embeddings. With 5120-dim embeddings and 500 samples,
    # alpha=0.01 results in rcond~2e-8 which triggers scipy warnings.
    DEFAULT_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    def __init__(
        self,
        source: TaskFeatureSource,
        ridge_alphas: Optional[List[float]] = None,
    ):
        """Initialize the predictor with a feature source.

        Args:
            source: TaskFeatureSource that provides features for tasks.
            ridge_alphas: List of alpha values for RidgeCV cross-validation.
                Defaults to [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0].
        """
        self.source = source
        self.ridge_alphas = ridge_alphas or self.DEFAULT_RIDGE_ALPHAS

        # Model state (set after fit())
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[RidgeCV] = None
        self._best_alpha: Optional[float] = None
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable predictor name (from feature source)."""
        return self.source.name

    def fit(self, task_ids: List[str], ground_truth_b: Union[np.ndarray, List[float]]) -> None:
        """Fit the predictor on training data.

        Args:
            task_ids: List of training task identifiers.
            ground_truth_b: Ground truth difficulty values (IRT b parameters).
                Must be same length as task_ids.

        Raises:
            ValueError: If task_ids and ground_truth_b have different lengths.
            ValueError: If any task is missing from the feature source.
        """
        # Convert to numpy array if needed
        y = np.asarray(ground_truth_b, dtype=np.float32)

        if len(task_ids) != len(y):
            raise ValueError(
                f"task_ids ({len(task_ids)}) and ground_truth_b ({len(y)}) must have same length"
            )

        # Get features (will raise if any task is missing)
        X = self.source.get_features(task_ids)

        # Fit scaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit Ridge with cross-validation
        self._model = RidgeCV(alphas=self.ridge_alphas, cv=5)
        self._model.fit(X_scaled, y)
        self._best_alpha = float(self._model.alpha_)

        self._is_fitted = True

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for the given tasks.

        Args:
            task_ids: List of task identifiers to predict for.

        Returns:
            Dictionary mapping task_id -> predicted difficulty.

        Raises:
            RuntimeError: If predict() is called before fit().
            ValueError: If any task is missing from the feature source.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict()")

        # Get features (will raise if any task is missing)
        X = self.source.get_features(task_ids)

        # Scale and predict
        X_scaled = self._scaler.transform(X)
        predictions = self._model.predict(X_scaled)

        return {task_id: float(pred) for task_id, pred in zip(task_ids, predictions)}

    def get_coefficients(self) -> Optional[Dict[str, float]]:
        """Get feature coefficients if feature names are available.

        Returns:
            Dictionary mapping feature_name -> coefficient, or None if
            the feature source doesn't provide feature names.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before getting coefficients")

        feature_names = self.source.feature_names
        if feature_names is None:
            return None

        coeffs = self._model.coef_
        return {name: float(c) for name, c in zip(feature_names, coeffs)}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model.

        Returns:
            Dictionary with model information including:
            - best_alpha: Selected Ridge regularization parameter
            - n_features: Number of features
            - feature_names: List of feature names (if available)
            - feature_coefficients: Dict of feature -> coefficient (if available)
        """
        if not self._is_fitted:
            return {"is_fitted": False}

        info = {
            "is_fitted": True,
            "best_alpha": self._best_alpha,
            "n_features": self.source.feature_dim,
            "feature_source": self.source.name,
        }

        # Add feature names and coefficients if available
        feature_names = self.source.feature_names
        if feature_names is not None:
            info["feature_names"] = feature_names
            info["feature_coefficients"] = self.get_coefficients()

        return info

    def print_model_summary(self) -> None:
        """Print a summary of the fitted model."""
        info = self.get_model_info()

        if not info["is_fitted"]:
            print("Model not fitted yet")
            return

        print(f"  Feature source: {info['feature_source']}")
        print(f"  Number of features: {info['n_features']}")
        print(f"  Best Ridge alpha: {info['best_alpha']:.2e}")

        if "feature_coefficients" in info:
            print("  Feature coefficients:")
            # Sort by absolute coefficient value
            sorted_coeffs = sorted(
                info["feature_coefficients"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for name, coef in sorted_coeffs:
                print(f"    {name}: {coef:+.4f}")