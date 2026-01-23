"""Unified feature-based difficulty predictor.

This module provides a single predictor class that works with any TaskFeatureSource.
The predictor uses StandardScaler + RidgeCV regression, which is sufficient for
both high-dimensional embeddings and low-dimensional semantic features.

Example usage:
    from experiment_ab_shared.feature_source import EmbeddingFeatureSource
    from experiment_ab_shared.feature_predictor import FeatureBasedPredictor

    # Create feature source
    source = EmbeddingFeatureSource(Path("embeddings.npz"))

    # Create predictor
    predictor = FeatureBasedPredictor(source)

    # Fit on training data
    predictor.fit(train_task_ids, train_difficulties)

    # Predict on test data
    predictions = predictor.predict(test_task_ids)
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    GroupedFeatureSource,
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


class GroupedRidgePredictor:
    """Ridge regression with per-group L2 penalties via feature pre-scaling.

    This predictor enables combining multiple feature sources with different
    regularization strengths for each source. This is useful when sources have
    different dimensionalities (e.g., 5120-dim embeddings vs 9-dim LLM features).

    The approach uses feature pre-scaling: scale features from group g by
    1/sqrt(alpha_g) before fitting standard ridge with alpha=1. This is
    mathematically equivalent to applying different per-group L2 penalties.

    Example:
        from experiment_ab_shared.feature_source import (
            GroupedFeatureSource, RegularizedFeatureSource,
            EmbeddingFeatureSource, CSVFeatureSource,
        )

        # Create grouped source
        grouped = GroupedFeatureSource([
            RegularizedFeatureSource(EmbeddingFeatureSource(path1), alpha=1000.0),
            RegularizedFeatureSource(CSVFeatureSource(path2), alpha=1.0),
        ])

        # Create predictor (will grid search over per-source alphas)
        predictor = GroupedRidgePredictor(grouped)
        predictor.fit(train_task_ids, train_difficulties)
        predictions = predictor.predict(test_task_ids)
    """

    DEFAULT_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

    def __init__(
        self,
        source: GroupedFeatureSource,
        alpha_grids: Optional[Dict[str, List[float]]] = None,
        default_alpha_grid: Optional[List[float]] = None,
    ):
        """Initialize the grouped ridge predictor.

        Args:
            source: GroupedFeatureSource with multiple underlying feature sources.
            alpha_grids: Optional per-source alpha grids for grid search.
                Keys are source names, values are lists of alphas to try.
                If a source is not in this dict, uses default_alpha_grid.
            default_alpha_grid: Default alpha grid for sources not in alpha_grids.
                Defaults to [0.1, 1.0, 10.0, 100.0, 1000.0].

        Raises:
            TypeError: If source is not a GroupedFeatureSource.
        """
        if not isinstance(source, GroupedFeatureSource):
            raise TypeError(
                f"GroupedRidgePredictor requires GroupedFeatureSource, got {type(source).__name__}. "
                "Use FeatureBasedPredictor for single sources."
            )

        self.source = source
        self._alpha_grids = alpha_grids or {}
        self._default_alpha_grid = default_alpha_grid or self.DEFAULT_ALPHA_GRID

        # Model state (set after fit())
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[Ridge] = None
        self._best_alphas: Optional[Dict[str, float]] = None
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return f"Grouped Ridge ({self.source.name})"

    def _get_alpha_grid_for_source(self, source_name: str) -> List[float]:
        """Get alpha grid for a specific source."""
        return self._alpha_grids.get(source_name, self._default_alpha_grid)

    def _apply_group_scaling(
        self, X: np.ndarray, alphas: Tuple[float, ...]
    ) -> np.ndarray:
        """Scale each feature group by 1/sqrt(alpha).

        This transforms the problem so that standard ridge with alpha=1 is
        equivalent to grouped ridge with per-group alphas.
        """
        X_out = X.copy()
        for slice_i, alpha_i in zip(self.source.group_slices, alphas):
            X_out[:, slice_i] = X_out[:, slice_i] / np.sqrt(alpha_i)
        return X_out

    def fit(self, task_ids: List[str], ground_truth_b: Union[np.ndarray, List[float]]) -> None:
        """Fit the predictor on training data with grid search over per-source alphas.

        Args:
            task_ids: List of training task identifiers.
            ground_truth_b: Ground truth difficulty values (IRT b parameters).

        Raises:
            ValueError: If task_ids and ground_truth_b have different lengths.
            ValueError: If any task is missing from the feature source.
        """
        y = np.asarray(ground_truth_b, dtype=np.float32)

        if len(task_ids) != len(y):
            raise ValueError(
                f"task_ids ({len(task_ids)}) and ground_truth_b ({len(y)}) must have same length"
            )

        # Get concatenated features
        X = self.source.get_features(task_ids)

        # Build alpha grid for each source
        source_grids = []
        for s in self.source.sources:
            grid = self._get_alpha_grid_for_source(s.name)
            source_grids.append(grid)

        # Grid search over all combinations of per-source alphas
        best_score = float("inf")
        best_alphas: Optional[Tuple[float, ...]] = None

        for alpha_combo in itertools.product(*source_grids):
            # Apply per-group scaling
            X_scaled = self._apply_group_scaling(X, alpha_combo)

            # Standardize (important: fit scaler on scaled data)
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_scaled)

            # Cross-validate with standard ridge (alpha=1 after scaling)
            scores = cross_val_score(
                Ridge(alpha=1.0), X_std, y, cv=5, scoring="neg_mean_squared_error"
            )
            mean_score = -scores.mean()  # Convert to positive MSE

            if mean_score < best_score:
                best_score = mean_score
                best_alphas = alpha_combo

        # Refit with best alphas
        self._best_alphas = {
            s.name: alpha for s, alpha in zip(self.source.sources, best_alphas)
        }

        X_scaled = self._apply_group_scaling(X, best_alphas)
        self._scaler = StandardScaler()
        X_std = self._scaler.fit_transform(X_scaled)
        self._model = Ridge(alpha=1.0)
        self._model.fit(X_std, y)

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

        X = self.source.get_features(task_ids)

        # Apply same scaling used during training
        alphas = tuple(self._best_alphas[s.name] for s in self.source.sources)
        X_scaled = self._apply_group_scaling(X, alphas)
        X_std = self._scaler.transform(X_scaled)

        predictions = self._model.predict(X_std)
        return {task_id: float(pred) for task_id, pred in zip(task_ids, predictions)}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model.

        Returns:
            Dictionary with model information including per-source alphas.
        """
        if not self._is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "best_alphas": self._best_alphas,
            "sources": [
                {"name": s.name, "dim": s.feature_dim, "best_alpha": self._best_alphas[s.name]}
                for s in self.source.sources
            ],
            "n_features_total": self.source.feature_dim,
        }