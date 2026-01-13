"""Difficulty predictor base class and implementations.

All difficulty predictors inherit from DifficultyPredictorBase and implement
the fit() and predict() methods for use in Experiment A.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DifficultyPredictorBase(ABC):
    """Abstract base class for all difficulty predictors.

    Provides the common interface that all predictors must implement.
    """

    @abstractmethod
    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Train on tasks with known IRT difficulties.

        Args:
            task_ids: List of task identifiers
            ground_truth_b: Array of ground truth difficulty values
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


class EmbeddingPredictor(DifficultyPredictorBase):
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

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Embedding"

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


class ConstantPredictor(DifficultyPredictorBase):
    """Baseline: predict mean difficulty for all tasks."""

    def __init__(self):
        self._mean_b: Optional[float] = None

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Constant"

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


class GroundTruthPredictor(DifficultyPredictorBase):
    """Oracle: use actual IRT difficulties (upper bound baseline)."""

    def __init__(self, items_df: pd.DataFrame):
        """Initialize with ground truth items.

        Args:
            items_df: DataFrame with index=task_id, column 'b' for difficulty
        """
        self._items = items_df

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Oracle"

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


class LunettePredictor(DifficultyPredictorBase):
    """Difficulty predictor using Lunette-extracted features + Ridge regression.

    Includes automatic feature selection using LassoCV for sparse selection.
    """

    # Default feature columns to use (exclude metadata and reasoning)
    DEFAULT_FEATURE_COLS = [
        "repo_file_count",
        "repo_line_count",
        "patch_file_count",
        "patch_line_count",
        "test_file_count",
        "related_file_count",
        "import_count",
        "class_count_in_file",
        "function_count_in_file",
        "test_count_fail_to_pass",
        "test_count_pass_to_pass",
        "git_commit_count",
        "directory_depth",
        "has_conftest",
        "has_init",
        "fix_in_description",
        "problem_clarity",
        "error_message_provided",
        "reproduction_steps",
        "fix_locality",
        "domain_knowledge_required",
        "fix_complexity",
        "logical_reasoning_required",
        "atypicality",
    ]

    def __init__(
        self,
        features_path: Path,
        ridge_alpha: float = 1.0,
        feature_selection: str = "lasso_cv",
        max_features: Optional[int] = 10,
        feature_cols: Optional[List[str]] = None,
    ):
        """Initialize Lunette predictor.

        Args:
            features_path: Path to CSV file with Lunette features
            ridge_alpha: Ridge regression regularization parameter
            feature_selection: Method for feature selection ("lasso_cv" or "select_k_best")
            max_features: Maximum number of features to select (None = no limit)
            feature_cols: List of feature columns to use (None = use defaults)
        """
        self.features_path = Path(features_path)
        self.ridge_alpha = ridge_alpha
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS

        self._model: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        self._feature_coefficients: Optional[Dict[str, float]] = None

        # Load features immediately
        self._load_features()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Lunette"

    def _load_features(self) -> None:
        """Load features from CSV file."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        self._features_df = pd.read_csv(self.features_path)

        # Set index to instance_id
        if "_instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("_instance_id")
        elif "instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("instance_id")

        # Filter to available feature columns
        available_cols = [c for c in self.feature_cols if c in self._features_df.columns]
        if len(available_cols) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_cols)
            print(f"Warning: Missing feature columns: {missing}")

        self.feature_cols = available_cols

    def _get_feature_matrix(self, task_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix for given task IDs.

        Returns:
            (X, available_task_ids) where X is (n_tasks, n_features)
        """
        if self._features_df is None:
            raise RuntimeError("Features not loaded")

        # Filter to available tasks
        available_tasks = [t for t in task_ids if t in self._features_df.index]

        if not available_tasks:
            return np.array([]).reshape(0, len(self.feature_cols)), []

        # Extract feature matrix
        X = self._features_df.loc[available_tasks, self.feature_cols].values.astype(np.float32)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        return X, available_tasks

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Ridge regression with feature selection.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from Lunette features")

        if len(available_tasks) == 0:
            raise ValueError("No tasks available for training")

        # Get corresponding ground truth values
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Step 1: Feature selection
        if self.feature_selection == "lasso_cv":
            self._fit_with_lasso_selection(X, y)
        elif self.feature_selection == "select_k_best":
            self._fit_with_kbest_selection(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")

    def _fit_with_lasso_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Lasso for feature selection, then Ridge for final model."""
        # Normalize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Lasso for feature selection
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y)

        # Get non-zero coefficients
        coef_abs = np.abs(lasso.coef_)
        nonzero_mask = coef_abs > 1e-6

        # Select features
        if self.max_features and np.sum(nonzero_mask) > self.max_features:
            # Take top k by absolute coefficient
            top_k_idx = np.argsort(coef_abs)[-self.max_features:]
            selected_mask = np.zeros(len(self.feature_cols), dtype=bool)
            selected_mask[top_k_idx] = True
        elif np.sum(nonzero_mask) == 0:
            # No features selected, use top k by correlation
            print("Warning: Lasso selected 0 features, falling back to top-k correlation")
            k = self.max_features or 5
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
        else:
            selected_mask = nonzero_mask

        self._selected_features = [
            self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]
        ]

        # Fit Ridge on selected features
        X_selected = X_scaled[:, selected_mask]
        self._model = Ridge(alpha=self.ridge_alpha)
        self._model.fit(X_selected, y)

        # Store coefficients for reporting
        self._feature_coefficients = dict(
            zip(self._selected_features, self._model.coef_.tolist())
        )
        self._selected_mask = selected_mask

    def _fit_with_kbest_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using SelectKBest for feature selection, then Ridge for final model."""
        # Normalize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # SelectKBest
        k = self.max_features or 10
        selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_mask = selector.get_support()

        self._selected_features = [
            self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]
        ]

        # Fit Ridge on selected features
        self._model = Ridge(alpha=self.ridge_alpha)
        self._model.fit(X_selected, y)

        # Store coefficients for reporting
        self._feature_coefficients = dict(
            zip(self._selected_features, self._model.coef_.tolist())
        )
        self._selected_mask = selected_mask

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if not available_tasks:
            return {}

        # Transform and select features
        X_scaled = self._scaler.transform(X)
        X_selected = X_scaled[:, self._selected_mask]

        # Predict
        preds = self._model.predict(X_selected)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def selected_features(self) -> Optional[List[str]]:
        """Return names of selected features."""
        return self._selected_features

    @property
    def feature_coefficients(self) -> Optional[Dict[str, float]]:
        """Return coefficients of selected features."""
        return self._feature_coefficients

    @property
    def n_features(self) -> int:
        """Return number of available features."""
        return len(self.feature_cols)

    @property
    def n_tasks(self) -> int:
        """Return number of tasks with features."""
        return len(self._features_df) if self._features_df is not None else 0

    def print_selected_features(self) -> None:
        """Print selected features and their coefficients."""
        if self._feature_coefficients is None:
            print("Model not fitted yet")
            return

        print(f"\nSelected features ({self.feature_selection}, n={len(self._selected_features)}):")
        sorted_features = sorted(
            self._feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, coef in sorted_features:
            sign = "+" if coef >= 0 else ""
            print(f"  {name:30s}: {sign}{coef:.4f}")


class LLMJudgePredictor(DifficultyPredictorBase):
    """Difficulty predictor using LLM-extracted semantic features + Lasso/Ridge regression.

    Uses only the 9 semantic features (no environment/sandbox features):
    - fix_in_description, problem_clarity, error_message_provided, reproduction_steps
    - fix_locality, domain_knowledge_required, fix_complexity
    - logical_reasoning_required, atypicality

    This is the ablation of LunettePredictor that doesn't use shell commands.
    """

    # The 9 semantic features (matching llm_judge_prompt.py)
    DEFAULT_FEATURE_COLS = [
        "fix_in_description",
        "problem_clarity",
        "error_message_provided",
        "reproduction_steps",
        "fix_locality",
        "domain_knowledge_required",
        "fix_complexity",
        "logical_reasoning_required",
        "atypicality",
    ]

    def __init__(
        self,
        features_path: Path,
        ridge_alpha: float = 1.0,
        max_features: Optional[int] = None,  # None = use all 9
        feature_cols: Optional[List[str]] = None,
    ):
        """Initialize LLM Judge predictor.

        Args:
            features_path: Path to CSV file with LLM judge features
            ridge_alpha: Ridge regression regularization parameter for final fit
            max_features: Maximum number of features to select (None = no limit)
            feature_cols: List of feature columns to use (None = use defaults)
        """
        self.features_path = Path(features_path)
        self.ridge_alpha = ridge_alpha
        self.max_features = max_features
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS

        self._model: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        self._feature_coefficients: Optional[Dict[str, float]] = None
        self._selected_mask: Optional[np.ndarray] = None

        # Load features immediately
        self._load_features()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "LLM Judge"

    def _load_features(self) -> None:
        """Load features from CSV file."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        self._features_df = pd.read_csv(self.features_path)

        # Set index to instance_id
        if "_instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("_instance_id")
        elif "instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("instance_id")

        # Filter to available feature columns
        available_cols = [c for c in self.feature_cols if c in self._features_df.columns]
        if len(available_cols) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_cols)
            print(f"Warning: Missing feature columns: {missing}")

        self.feature_cols = available_cols

    def _get_feature_matrix(self, task_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix for given task IDs.

        Returns:
            (X, available_task_ids) where X is (n_tasks, n_features)
        """
        if self._features_df is None:
            raise RuntimeError("Features not loaded")

        # Filter to available tasks
        available_tasks = [t for t in task_ids if t in self._features_df.index]

        if not available_tasks:
            return np.array([]).reshape(0, len(self.feature_cols)), []

        # Extract feature matrix
        X = self._features_df.loc[available_tasks, self.feature_cols].values.astype(np.float32)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        return X, available_tasks

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Lasso for feature selection, then Ridge for final model.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from LLM Judge features")

        if len(available_tasks) == 0:
            raise ValueError("No tasks available for training")

        # Get corresponding ground truth values
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Normalize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Lasso for feature selection
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y)

        # Get non-zero coefficients
        coef_abs = np.abs(lasso.coef_)
        nonzero_mask = coef_abs > 1e-6

        # Select features
        if self.max_features and np.sum(nonzero_mask) > self.max_features:
            # Take top k by absolute coefficient
            top_k_idx = np.argsort(coef_abs)[-self.max_features:]
            selected_mask = np.zeros(len(self.feature_cols), dtype=bool)
            selected_mask[top_k_idx] = True
        elif np.sum(nonzero_mask) == 0:
            # No features selected, use top k by correlation
            print("Warning: Lasso selected 0 features, falling back to top-k correlation")
            k = self.max_features or 5
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
        else:
            selected_mask = nonzero_mask

        self._selected_features = [
            self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]
        ]

        # Fit Ridge on selected features
        X_selected = X_scaled[:, selected_mask]
        self._model = Ridge(alpha=self.ridge_alpha)
        self._model.fit(X_selected, y)

        # Store coefficients for reporting
        self._feature_coefficients = dict(
            zip(self._selected_features, self._model.coef_.tolist())
        )
        self._selected_mask = selected_mask

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if not available_tasks:
            return {}

        # Transform and select features
        X_scaled = self._scaler.transform(X)
        X_selected = X_scaled[:, self._selected_mask]

        # Predict
        preds = self._model.predict(X_selected)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def selected_features(self) -> Optional[List[str]]:
        """Return names of selected features."""
        return self._selected_features

    @property
    def feature_coefficients(self) -> Optional[Dict[str, float]]:
        """Return coefficients of selected features."""
        return self._feature_coefficients

    @property
    def n_features(self) -> int:
        """Return number of available features."""
        return len(self.feature_cols)

    @property
    def n_tasks(self) -> int:
        """Return number of tasks with features."""
        return len(self._features_df) if self._features_df is not None else 0

    def print_selected_features(self) -> None:
        """Print selected features and their coefficients."""
        if self._feature_coefficients is None:
            print("Model not fitted yet")
            return

        print(f"\nSelected features (lasso_cv, n={len(self._selected_features)}):")
        sorted_features = sorted(
            self._feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, coef in sorted_features:
            sign = "+" if coef >= 0 else ""
            print(f"  {name:30s}: {sign}{coef:.4f}")
