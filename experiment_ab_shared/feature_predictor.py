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
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
    GroupedFeatureSource,
)


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

    # Per-source alpha grids for common feature types.
    # Wide range spanning 1e-6 to 1e4, matching Daria's predict_question_difficulty.py defaults.
    # Sources not in this dict must have alpha_grids explicitly provided.
    SOURCE_ALPHA_GRIDS = {
        "Embedding": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "LLM Judge": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "Trajectory": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
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

            for alpha_combo in itertools.product(*source_grids):
                # Apply per-group scaling AFTER standardization
                X_scaled = self._apply_group_scaling(X_std, alpha_combo)

                # Cross-validate with standard ridge (alpha=1 after scaling)
                scores = cross_val_score(
                    Ridge(alpha=1.0), X_scaled, y, cv=5, scoring="neg_mean_squared_error"
                )
                mean_score = -scores.mean()  # Convert to positive MSE

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


class StackedResidualPredictor:
    """Two-stage stacked predictor: base model + residual correction.

    Instead of concatenating features and fitting jointly (like GroupedRidgePredictor),
    this uses a two-stage approach:

    1. Stage 1 (Base): Fit Ridge on base_source to predict β
    2. Stage 2 (Residual): Fit Ridge on residual_source to predict (β_true - β̂_base)
    3. Final prediction: β̂ = β̂_base + β̂_residual

    This allows the residual source to specifically correct errors from the base model,
    rather than competing in the same feature space.

    Example:
        # Embedding as base, LLM Judge corrects residuals
        predictor = StackedResidualPredictor(
            base_source=embedding_source,
            residual_source=llm_judge_source,
        )
        predictor.fit(train_task_ids, train_difficulties)
        predictions = predictor.predict(test_task_ids)
    """

    # Default alpha grids by source type
    # High-dim embeddings need stronger regularization
    HIGH_DIM_ALPHAS = [100.0, 1000.0, 10000.0, 30000.0]
    # Low-dim features (like LLM judge) need less regularization
    LOW_DIM_ALPHAS = [0.1, 1.0, 10.0, 100.0]

    def __init__(
        self,
        base_source: TaskFeatureSource,
        residual_source: TaskFeatureSource,
        base_alphas: Optional[List[float]] = None,
        residual_alphas: Optional[List[float]] = None,
    ):
        """Initialize the stacked residual predictor.

        Args:
            base_source: Feature source for the base model (Stage 1).
            residual_source: Feature source for the residual model (Stage 2).
            base_alphas: Optional list of alpha values for base model RidgeCV.
                Defaults based on feature dimensionality.
            residual_alphas: Optional list of alpha values for residual model RidgeCV.
                Defaults based on feature dimensionality.
        """
        self.base_source = base_source
        self.residual_source = residual_source

        # Set default alphas based on feature dimensionality
        if base_alphas is not None:
            self.base_alphas = base_alphas
        else:
            self.base_alphas = (
                self.HIGH_DIM_ALPHAS if base_source.feature_dim > 100
                else self.LOW_DIM_ALPHAS
            )

        if residual_alphas is not None:
            self.residual_alphas = residual_alphas
        else:
            self.residual_alphas = (
                self.HIGH_DIM_ALPHAS if residual_source.feature_dim > 100
                else self.LOW_DIM_ALPHAS
            )

        # Model state (set after fit())
        self._base_scaler: Optional[StandardScaler] = None
        self._residual_scaler: Optional[StandardScaler] = None
        self._base_model: Optional[RidgeCV] = None
        self._residual_model: Optional[RidgeCV] = None
        self._base_best_alpha: Optional[float] = None
        self._residual_best_alpha: Optional[float] = None
        self._is_fitted: bool = False

        # Diagnostics
        self._train_residual_std: Optional[float] = None  # Std of residuals from base model

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return f"Stacked ({self.base_source.name} → {self.residual_source.name})"

    def fit(self, task_ids: List[str], ground_truth_b: Union[np.ndarray, List[float]]) -> None:
        """Fit the two-stage stacked predictor.

        Stage 1: Fit base model to predict β from base_source features
        Stage 2: Fit residual model to predict (β_true - β̂_base) from residual_source features

        Args:
            task_ids: List of training task identifiers.
            ground_truth_b: Ground truth difficulty values (IRT b parameters).

        Raises:
            ValueError: If task_ids and ground_truth_b have different lengths.
            ValueError: If any task is missing from either feature source.
        """
        y = np.asarray(ground_truth_b, dtype=np.float32)

        if len(task_ids) != len(y):
            raise ValueError(
                f"task_ids ({len(task_ids)}) and ground_truth_b ({len(y)}) must have same length"
            )

        # === Stage 1: Base Model ===
        # Get base features (will raise if any task is missing)
        X_base = self.base_source.get_features(task_ids)

        # Fit scaler for base features
        self._base_scaler = StandardScaler()
        X_base_scaled = self._base_scaler.fit_transform(X_base)

        # Fit base model with cross-validation
        self._base_model = RidgeCV(alphas=self.base_alphas, cv=5)
        self._base_model.fit(X_base_scaled, y)
        self._base_best_alpha = float(self._base_model.alpha_)

        # Get base predictions on training data
        base_predictions = self._base_model.predict(X_base_scaled)

        # === Stage 2: Residual Model ===
        # Compute residuals: what the base model got wrong
        residuals = y - base_predictions
        self._train_residual_std = float(np.std(residuals))

        # Get residual features (will raise if any task is missing)
        X_residual = self.residual_source.get_features(task_ids)

        # Fit scaler for residual features
        self._residual_scaler = StandardScaler()
        X_residual_scaled = self._residual_scaler.fit_transform(X_residual)

        # Fit residual model with cross-validation
        self._residual_model = RidgeCV(alphas=self.residual_alphas, cv=5)
        self._residual_model.fit(X_residual_scaled, residuals)
        self._residual_best_alpha = float(self._residual_model.alpha_)

        self._is_fitted = True

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty using the stacked ensemble.

        Final prediction: β̂ = β̂_base + β̂_residual

        Args:
            task_ids: List of task identifiers to predict for.

        Returns:
            Dictionary mapping task_id -> predicted difficulty.

        Raises:
            RuntimeError: If predict() is called before fit().
            ValueError: If any task is missing from either feature source.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict()")

        # Get base predictions
        X_base = self.base_source.get_features(task_ids)
        X_base_scaled = self._base_scaler.transform(X_base)
        base_predictions = self._base_model.predict(X_base_scaled)

        # Get residual predictions
        X_residual = self.residual_source.get_features(task_ids)
        X_residual_scaled = self._residual_scaler.transform(X_residual)
        residual_predictions = self._residual_model.predict(X_residual_scaled)

        # Combine: β̂ = β̂_base + β̂_residual
        final_predictions = base_predictions + residual_predictions

        return {task_id: float(pred) for task_id, pred in zip(task_ids, final_predictions)}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model.

        Returns:
            Dictionary with model information including:
            - base_model: Info about the base model (source, best_alpha, n_features)
            - residual_model: Info about the residual model
            - train_residual_std: Standard deviation of residuals from base model
        """
        if not self._is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "base_model": {
                "source": self.base_source.name,
                "n_features": self.base_source.feature_dim,
                "best_alpha": self._base_best_alpha,
            },
            "residual_model": {
                "source": self.residual_source.name,
                "n_features": self.residual_source.feature_dim,
                "best_alpha": self._residual_best_alpha,
            },
            "train_residual_std": self._train_residual_std,
        }

    def print_model_summary(self) -> None:
        """Print a summary of the fitted model."""
        info = self.get_model_info()

        if not info["is_fitted"]:
            print("Model not fitted yet")
            return

        print(f"  Stacked Residual Predictor")
        print(f"  Stage 1 (Base): {info['base_model']['source']}")
        print(f"    Features: {info['base_model']['n_features']}")
        print(f"    Best alpha: {info['base_model']['best_alpha']:.2e}")
        print(f"  Stage 2 (Residual): {info['residual_model']['source']}")
        print(f"    Features: {info['residual_model']['n_features']}")
        print(f"    Best alpha: {info['residual_model']['best_alpha']:.2e}")
        print(f"  Training residual std: {info['train_residual_std']:.4f}")