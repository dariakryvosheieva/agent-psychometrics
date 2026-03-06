"""Feature-based difficulty predictors and base class.

This module provides:
- DifficultyPredictorBase: Abstract base class for difficulty predictors
- FeatureBasedPredictor: Source-agnostic Ridge/Lasso regression predictor
- GroupedRidgePredictor: Per-group L2 penalties via feature pre-scaling

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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
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
    """Difficulty predictor using any TaskFeatureSource + regularized regression.

    Pipeline: features -> StandardScaler -> RidgeCV/LassoCV -> predict

    This predictor is source-agnostic: it works identically for embeddings,
    LLM judge features, or any other feature type. The only requirements are:
    1. A TaskFeatureSource that provides features for tasks
    2. Ground truth difficulty values for training

    Supports two regularization methods:
    - Ridge (L2): Shrinks all coefficients toward zero, good for correlated features
    - Lasso (L1): Performs feature selection by driving some coefficients to exactly zero

    Attributes:
        source: The TaskFeatureSource providing features.
        name: Human-readable name (from source.name).
        method: Regularization method ("ridge" or "lasso").
        alphas: List of alpha values for cross-validation.
    """

    # Default Ridge alphas spanning 5 orders of magnitude
    # Note: Starting from 0.1 (not 0.01) to avoid ill-conditioned matrices
    # with high-dimensional embeddings. With 5120-dim embeddings and 500 samples,
    # alpha=0.01 results in rcond~2e-8 which triggers scipy warnings.
    DEFAULT_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # Lasso alphas for feature selection
    # Lower values = less regularization (more features kept)
    # Higher values = more regularization (fewer features kept)
    # Note: Very low alphas (0.0001, 0.001) cause convergence issues with high-dim features
    DEFAULT_LASSO_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

    def __init__(
        self,
        source: TaskFeatureSource,
        alphas: Optional[List[float]] = None,
        method: str = "ridge",
    ):
        """Initialize the predictor with a feature source.

        Args:
            source: TaskFeatureSource that provides features for tasks.
            alphas: List of alpha values for cross-validation.
                Defaults depend on method: Ridge uses higher alphas, Lasso uses wider range.
            method: Regularization method, either "ridge" (default) or "lasso".
                - ridge: L2 regularization, shrinks all coefficients
                - lasso: L1 regularization, performs feature selection
        """
        if method not in ("ridge", "lasso"):
            raise ValueError(f"method must be 'ridge' or 'lasso', got '{method}'")

        self.source = source
        self.method = method

        # Set default alphas based on method
        if alphas is not None:
            self.alphas = alphas
        elif method == "ridge":
            self.alphas = self.DEFAULT_RIDGE_ALPHAS
        else:  # lasso
            self.alphas = self.DEFAULT_LASSO_ALPHAS

        # Model state (set after fit())
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[Union[RidgeCV, LassoCV]] = None
        self._best_alpha: Optional[float] = None
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        """Human-readable predictor name (from feature source)."""
        suffix = " (Lasso)" if self.method == "lasso" else ""
        return f"{self.source.name}{suffix}"

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

        # Fit with cross-validation (within training data only)
        if self.method == "ridge":
            self._model = RidgeCV(alphas=self.alphas, cv=5)
        else:  # lasso
            # max_iter increased for convergence with high-dim features
            self._model = LassoCV(alphas=self.alphas, cv=5, max_iter=50000)

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
            - method: Regularization method ("ridge" or "lasso")
            - best_alpha: Selected regularization parameter
            - n_features: Number of features
            - n_nonzero: Number of non-zero coefficients (for Lasso)
            - sparsity: Fraction of zero coefficients (for Lasso)
            - feature_names: List of feature names (if available)
            - feature_coefficients: Dict of feature -> coefficient (if available)
        """
        if not self._is_fitted:
            return {"is_fitted": False}

        coeffs = self._model.coef_
        n_nonzero = int(np.sum(np.abs(coeffs) > 1e-10))

        info = {
            "is_fitted": True,
            "method": self.method,
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

        method_name = info["method"].capitalize()
        print(f"  Feature source: {info['feature_source']}")
        print(f"  Method: {method_name}")
        print(f"  Number of features: {info['n_features']}")
        print(f"  Best {method_name} alpha: {info['best_alpha']:.2e}")

        # Show sparsity info for Lasso
        if info["method"] == "lasso":
            pct_nonzero = 100 * (1 - info["sparsity"])
            print(f"  Non-zero features: {info['n_nonzero']} ({pct_nonzero:.1f}%)")

        if "feature_coefficients" in info:
            # For Lasso, only show non-zero coefficients
            if info["method"] == "lasso":
                print("  Non-zero feature coefficients:")
            else:
                print("  Feature coefficients:")

            # Sort by absolute coefficient value
            sorted_coeffs = sorted(
                info["feature_coefficients"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for name, coef in sorted_coeffs:
                # For Lasso, skip zero coefficients
                if info["method"] == "lasso" and abs(coef) < 1e-10:
                    continue
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
        self._per_source_scalers = {}
        X_std = np.empty_like(X)
        for source, slice_obj in zip(self.source.sources, self.source.group_slices):
            scaler = StandardScaler()
            X_std[:, slice_obj] = scaler.fit_transform(X[:, slice_obj])
            self._per_source_scalers[source.name] = scaler

        if self._fixed_alphas is not None:
            # Use fixed alphas directly (no grid search)
            best_alphas = tuple(self._fixed_alphas[s.name] for s in self.source.sources)
        else:
            # Grid search over all combinations of per-source alphas
            source_grids = []
            for s in self.source.sources:
                grid = self._get_alpha_grid_for_source(s.name)
                source_grids.append(grid)

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

                    # Fit per-source scalers on TRAINING fold only
                    X_train_std = np.empty_like(X_train)
                    X_val_std = np.empty_like(X_val)
                    for source, slice_obj in zip(self.source.sources, self.source.group_slices):
                        scaler = StandardScaler()
                        X_train_std[:, slice_obj] = scaler.fit_transform(X_train[:, slice_obj])
                        X_val_std[:, slice_obj] = scaler.transform(X_val[:, slice_obj])

                    # Apply per-group alpha scaling
                    X_train_scaled = self._apply_group_scaling(X_train_std, alpha_combo)
                    X_val_scaled = self._apply_group_scaling(X_val_std, alpha_combo)

                    # Fit and evaluate
                    model = Ridge(alpha=1.0)
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_val_scaled)
                    mse = float(np.mean((y_val - pred) ** 2))
                    fold_mses.append(mse)

                mean_score = np.mean(fold_mses)

                if mean_score < best_score:
                    best_score = mean_score
                    best_alphas = alpha_combo

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

        # Apply same transformations as training: per-source StandardScaler -> Group Scaling
        X_std = np.empty_like(X)
        for source, slice_obj in zip(self.source.sources, self.source.group_slices):
            scaler = self._per_source_scalers[source.name]
            X_std[:, slice_obj] = scaler.transform(X[:, slice_obj])
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
