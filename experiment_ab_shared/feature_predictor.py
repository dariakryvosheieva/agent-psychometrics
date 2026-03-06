"""Feature-based difficulty predictors and base class.

This module provides:
- DifficultyPredictorBase: Abstract base class for difficulty predictors
- FeatureBasedPredictor: Source-agnostic RidgeCV predictor
- GroupedRidgePredictor: Per-group L2 penalties via feature pre-scaling
"""

import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    GroupedFeatureSource,
)


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


class FeatureBasedPredictor:
    """Difficulty predictor using any TaskFeatureSource + RidgeCV regression.

    Pipeline: features -> StandardScaler -> RidgeCV -> predict

    This predictor is source-agnostic: it works identically for embeddings,
    LLM judge features, or any other feature type. The only requirements are:
    1. A TaskFeatureSource that provides features for tasks
    2. Ground truth difficulty values for training

    Attributes:
        source: The TaskFeatureSource providing features.
        name: Human-readable name (from source.name).
        alphas: List of alpha values for cross-validation.
    """

    # Default Ridge alphas spanning 5 orders of magnitude
    # Note: Starting from 0.1 (not 0.01) to avoid ill-conditioned matrices
    # with high-dimensional embeddings. With 5120-dim embeddings and 500 samples,
    # alpha=0.01 results in rcond~2e-8 which triggers scipy warnings.
    DEFAULT_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    def __init__(
        self,
        source: TaskFeatureSource,
        alphas: Optional[List[float]] = None,
    ):
        """Initialize the predictor with a feature source.

        Args:
            source: TaskFeatureSource that provides features for tasks.
            alphas: List of alpha values for cross-validation.
        """
        self.source = source
        self.alphas = alphas if alphas is not None else self.DEFAULT_RIDGE_ALPHAS

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
        y = np.asarray(ground_truth_b, dtype=np.float32)

        if len(task_ids) != len(y):
            raise ValueError(
                f"task_ids ({len(task_ids)}) and ground_truth_b ({len(y)}) must have same length"
            )

        # Get features (will raise if any task is missing)
        X = self.source.get_features(task_ids)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = RidgeCV(alphas=self.alphas, cv=5)
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
            - best_alpha: Selected regularization parameter
            - n_features: Number of features
            - feature_names: List of feature names (if available)
            - feature_coefficients: Dict of feature -> coefficient (if available)
        """
        if not self._is_fitted:
            return {"is_fitted": False}

        coeffs = self._model.coef_
        n_nonzero = int(np.sum(np.abs(coeffs) > 1e-10))

        info = {
            "is_fitted": True,
            "best_alpha": self._best_alpha,
            "n_features": self.source.feature_dim,
            "n_nonzero": n_nonzero,
            "sparsity": 1 - (n_nonzero / self.source.feature_dim),
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
            GroupedFeatureSource, EmbeddingFeatureSource, CSVFeatureSource,
        )

        # Create grouped source
        grouped = GroupedFeatureSource([
            EmbeddingFeatureSource(path1),
            CSVFeatureSource(path2),
        ])

        # Create predictor (will grid search over per-source alphas)
        predictor = GroupedRidgePredictor(grouped)
        predictor.fit(train_task_ids, train_difficulties)
        predictions = predictor.predict(test_task_ids)
    """

    # Per-source alpha grids for common feature types.
    # Grids are tailored to feature dimensionality:
    # - High-dim (Embedding ~5120): needs stronger regularization (higher alphas)
    # - Low-dim (LLM Judge ~9): needs weaker regularization (lower alphas)
    SOURCE_ALPHA_GRIDS = {
        "Embedding": [100.0, 1000.0, 10000.0],  # High-dim: strong regularization
        "LLM Judge": [0.01, 0.1, 1.0, 10.0],    # Low-dim: weak regularization
    }

    def __init__(
        self,
        source: GroupedFeatureSource,
        alpha_grids: Optional[Dict[str, List[float]]] = None,
        fixed_alphas: Optional[Dict[str, float]] = None,
    ):
        """Initialize the grouped ridge predictor.

        Args:
            source: GroupedFeatureSource with multiple underlying feature sources.
            alpha_grids: Optional per-source alpha grids for grid search.
                Keys are source names, values are lists of alphas to try.
                If a source is not in this dict, uses SOURCE_ALPHA_GRIDS.
                Raises error if source is in neither.
            fixed_alphas: If provided, use these exact alphas per source (no grid search).
                Keys are source names, values are alpha values.
                Mutually exclusive with alpha_grids.

        Raises:
            TypeError: If source is not a GroupedFeatureSource.
            ValueError: If fixed_alphas is provided but missing entries for some sources.
        """
        if not isinstance(source, GroupedFeatureSource):
            raise TypeError(
                f"GroupedRidgePredictor requires GroupedFeatureSource, got {type(source).__name__}. "
                "Use FeatureBasedPredictor for single sources."
            )

        # Validate fixed_alphas if provided
        if fixed_alphas is not None:
            source_names = {s.name for s in source.sources}
            missing = source_names - set(fixed_alphas.keys())
            if missing:
                raise ValueError(
                    f"fixed_alphas missing entries for sources: {missing}"
                )

        self.source = source
        self._alpha_grids = alpha_grids or {}
        self._fixed_alphas = fixed_alphas

        # Model state (set after fit())
        self._per_source_scalers: Optional[Dict[str, StandardScaler]] = None
        self._model: Optional[Ridge] = None
        self._best_alphas: Optional[Dict[str, float]] = None
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        if self._fixed_alphas is not None:
            alpha_str = ", ".join(f"{k}={v}" for k, v in self._fixed_alphas.items())
            return f"Grouped Ridge ({alpha_str})"
        return f"Grouped Ridge ({self.source.name})"

    def _get_alpha_grid_for_source(self, source_name: str) -> List[float]:
        """Get alpha grid for a specific source.

        Raises:
            ValueError: If source is not in alpha_grids or SOURCE_ALPHA_GRIDS.
        """
        if source_name in self._alpha_grids:
            return self._alpha_grids[source_name]
        if source_name in self.SOURCE_ALPHA_GRIDS:
            return self.SOURCE_ALPHA_GRIDS[source_name]
        raise ValueError(
            f"No alpha grid defined for source '{source_name}'. "
            f"Either add it to SOURCE_ALPHA_GRIDS or provide alpha_grids explicitly. "
            f"Known sources: {list(self.SOURCE_ALPHA_GRIDS.keys())}"
        )

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
        """Fit the predictor on training data.

        If fixed_alphas was provided at construction, uses those alphas directly.
        Otherwise, performs grid search over per-source alphas using MSE.

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

        # IMPORTANT: StandardScaler must be applied BEFORE group scaling.
        # If applied after, it normalizes each feature to unit variance,
        # completely negating the differential regularization from group scaling.
        # Correct order: StandardScaler -> Group Scaling -> Ridge(alpha=1)
        #
        # We fit separate scalers per source, ensuring each block has mean=0, std=1
        # independently. This is important when combining high-dim (embeddings) and
        # low-dim (LLM) sources, as a single scaler would have statistics dominated
        # by the high-dim source.
        self._per_source_scalers, X_std = self.source.fit_scalers(X)

        if self._fixed_alphas is not None:
            # Use fixed alphas directly (no grid search)
            best_alphas = tuple(self._fixed_alphas[s.name] for s in self.source.sources)
        else:
            # Grid search over all combinations of per-source alphas
            source_grids = [self._get_alpha_grid_for_source(s.name) for s in self.source.sources]

            best_score = float("inf")
            best_alphas = None

            # Use manual CV to re-fit scalers per fold (avoids data leakage)
            from sklearn.model_selection import KFold
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

            for alpha_combo in itertools.product(*source_grids):
                fold_mses = []
                for train_idx, val_idx in inner_cv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    fold_scalers, X_train_std = self.source.fit_scalers(X_train)
                    X_val_std = self.source.apply_scalers(X_val, fold_scalers)

                    X_train_scaled = self._apply_group_scaling(X_train_std, alpha_combo)
                    X_val_scaled = self._apply_group_scaling(X_val_std, alpha_combo)

                    model = Ridge(alpha=1.0)
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    fold_mses.append(float(np.mean((y_val - pred) ** 2)))

                if np.mean(fold_mses) < best_score:
                    best_score = np.mean(fold_mses)
                    best_alphas = alpha_combo

            assert best_alphas is not None  # grid always has at least one combination

        # Fit with selected alphas
        self._best_alphas = {
            s.name: alpha for s, alpha in zip(self.source.sources, best_alphas)
        }

        X_scaled = self._apply_group_scaling(X_std, best_alphas)
        self._model = Ridge(alpha=1.0)
        self._model.fit(X_scaled, y)

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

        X_std = self.source.apply_scalers(X, self._per_source_scalers)
        alphas = tuple(self._best_alphas[s.name] for s in self.source.sources)
        X_scaled = self._apply_group_scaling(X_std, alphas)

        predictions = self._model.predict(X_scaled)
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

    def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics about the fitted model for debugging.

        Returns:
            Dictionary with detailed coefficient analysis including:
            - selected_alphas: Per-source regularization strengths
            - coef_by_source: Coefficient statistics for each source
            - intercept: Model intercept
            - feature_names: Names of LLM judge features (if available)
        """
        if not self._is_fitted:
            return {"is_fitted": False}

        coefs = self._model.coef_

        # Slice coefficients by source
        coef_by_source = {}
        for source, slice_obj in zip(self.source.sources, self.source.group_slices):
            source_coefs = coefs[slice_obj]
            coef_by_source[source.name] = {
                "n_features": len(source_coefs),
                "l2_norm": float(np.linalg.norm(source_coefs)),
                "mean_abs": float(np.mean(np.abs(source_coefs))),
                "max_abs": float(np.max(np.abs(source_coefs))),
                "nonzero_frac": float(np.mean(np.abs(source_coefs) > 1e-6)),
                "coefficients": source_coefs.tolist(),
            }

            # Add feature names if available (typically for LLM Judge)
            if source.feature_names is not None:
                coef_by_source[source.name]["feature_names"] = source.feature_names
                coef_by_source[source.name]["named_coefficients"] = {
                    name: float(c) for name, c in zip(source.feature_names, source_coefs)
                }

        # Compute effective contribution (sum of squared coefficients)
        total_l2_sq = sum(d["l2_norm"] ** 2 for d in coef_by_source.values())
        if total_l2_sq > 0:
            for name, d in coef_by_source.items():
                d["contribution_pct"] = 100 * (d["l2_norm"] ** 2) / total_l2_sq
        else:
            for name, d in coef_by_source.items():
                d["contribution_pct"] = 0.0

        return {
            "is_fitted": True,
            "selected_alphas": self._best_alphas,
            "coef_by_source": coef_by_source,
            "intercept": float(self._model.intercept_),
        }
