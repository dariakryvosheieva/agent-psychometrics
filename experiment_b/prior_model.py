"""Prior models for difficulty prediction.

Supports two approaches:
1. HeuristicPriorModel: Simple features (repo, text length) + Ridge
2. EmbeddingPriorModel: Daria's embeddings + Ridge (better performance)
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_task_features(task_ids: List[str]) -> pd.DataFrame:
    """Extract simple features from task data.

    Features (kept simple per requirements):
    - problem_len: Length of problem statement
    - problem_lines: Number of lines
    - patch_len: Length of gold patch
    - patch_files: Number of files in patch
    - repo: Repository name (categorical)
    """
    # Load SWE-bench data
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        task_data = {ex["instance_id"]: ex for ex in ds}
    except Exception as e:
        print(f"Warning: Could not load SWE-bench dataset: {e}")
        # Return empty dataframe if dataset unavailable
        return pd.DataFrame(columns=["task_id", "problem_len", "problem_lines", "patch_len", "patch_files", "repo"]).set_index("task_id")

    features = []
    for task_id in task_ids:
        if task_id not in task_data:
            continue
        task = task_data[task_id]

        problem = task["problem_statement"]
        patch = task["patch"]

        features.append(
            {
                "task_id": task_id,
                "problem_len": len(problem),
                "problem_lines": problem.count("\n"),
                "patch_len": len(patch),
                "patch_files": patch.count("diff --git"),
                "repo": task["repo"],
            }
        )

    return pd.DataFrame(features).set_index("task_id")


class PriorModel:
    """Simple linear model predicting difficulty from task features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Optional[Pipeline] = None
        self.feature_names = ["problem_len", "problem_lines", "patch_len", "patch_files"]
        self._features_cache: Optional[pd.DataFrame] = None

    def fit(self, task_ids: List[str], difficulties: np.ndarray) -> "PriorModel":
        """Fit prior model on task features.

        Args:
            task_ids: List of task IDs
            difficulties: Array of IRT b values (aligned with task_ids)
        """
        # Extract features
        features_df = extract_task_features(task_ids)
        self._features_cache = features_df

        if features_df.empty:
            print("Warning: No features extracted, prior model will return zeros")
            return self

        # Build pipeline
        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), self.feature_names),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["repo"]),
            ]
        )

        self.model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=self.alpha)),
            ]
        )

        # Align features with difficulties
        aligned_tasks = [t for t in task_ids if t in features_df.index]
        if not aligned_tasks:
            print("Warning: No aligned tasks found")
            return self

        X = features_df.loc[aligned_tasks]
        y = pd.Series(difficulties, index=task_ids).loc[aligned_tasks]

        self.model.fit(X, y)
        print(f"Prior model trained on {len(aligned_tasks)} tasks")

        return self

    def predict(self, task_ids: List[str]) -> np.ndarray:
        """Predict difficulty for tasks."""
        if self.model is None:
            return np.zeros(len(task_ids))

        # Use cached features or extract new ones
        if self._features_cache is not None:
            features_df = self._features_cache
            # Extract features for any new tasks not in cache
            new_tasks = [t for t in task_ids if t not in features_df.index]
            if new_tasks:
                new_features = extract_task_features(new_tasks)
                features_df = pd.concat([features_df, new_features])
                self._features_cache = features_df
        else:
            features_df = extract_task_features(task_ids)

        valid_tasks = [t for t in task_ids if t in features_df.index]
        if not valid_tasks:
            return np.zeros(len(task_ids))

        X = features_df.loc[valid_tasks]
        predictions = self.model.predict(X)

        # Return predictions aligned with input task_ids
        result = np.zeros(len(task_ids))
        for i, t in enumerate(task_ids):
            if t in valid_tasks:
                idx = valid_tasks.index(t)
                result[i] = predictions[idx]

        return result

    def get_prior_predictions(self, task_ids: List[str]) -> Dict[str, float]:
        """Get prior predictions as a dict."""
        predictions = self.predict(task_ids)
        return dict(zip(task_ids, predictions))

    def get_feature_coefficients(self) -> Dict[str, float]:
        """Get coefficients of the linear model for interpretability."""
        if self.model is None:
            return {}

        regressor = self.model.named_steps["regressor"]
        preprocessor = self.model.named_steps["preprocessor"]

        # Get feature names after preprocessing
        num_names = self.feature_names
        cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(["repo"]))

        all_names = num_names + cat_names
        coeffs = regressor.coef_

        return dict(zip(all_names, coeffs))

    def get_prior_features(self, task_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get the raw input features used by the prior model.

        For the heuristic prior, returns the preprocessed feature vectors
        (scaled numerical + one-hot encoded categorical).

        Args:
            task_ids: List of task IDs to get features for

        Returns:
            Dict mapping task_id -> feature vector (preprocessed)
        """
        if self.model is None or self._features_cache is None:
            return {}

        preprocessor = self.model.named_steps["preprocessor"]
        valid_tasks = [t for t in task_ids if t in self._features_cache.index]

        if not valid_tasks:
            return {}

        X = self._features_cache.loc[valid_tasks]
        X_transformed = preprocessor.transform(X)

        return {task_id: X_transformed[i] for i, task_id in enumerate(valid_tasks)}

    def get_prior_feature_dim(self) -> int:
        """Get the dimensionality of prior input features."""
        if self.model is None:
            return 0
        preprocessor = self.model.named_steps["preprocessor"]
        # Count numerical + categorical features
        num_features = len(self.feature_names)
        cat_features = len(preprocessor.named_transformers_["cat"].get_feature_names_out(["repo"]))
        return num_features + cat_features


class EmbeddingPriorModel:
    """Prior model using Daria's pre-computed embeddings + Ridge regression.

    This typically gives much better performance than heuristic features.
    Requires a pre-computed embeddings .npz file from the Qwen3-VL model.
    """

    def __init__(self, embeddings_path: Path, alpha: float = 10000.0):
        """Initialize embedding prior.

        Args:
            embeddings_path: Path to pre-computed embeddings .npz file
            alpha: Ridge regression regularization parameter
        """
        self.embeddings_path = embeddings_path
        self.alpha = alpha
        self.model: Optional[Pipeline] = None
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
        print(f"Loaded {len(self._embeddings)} embeddings, dim={self._embedding_dim}")

    def fit(self, task_ids: List[str], difficulties: np.ndarray) -> "EmbeddingPriorModel":
        """Fit Ridge regression on task embeddings.

        Args:
            task_ids: List of training task identifiers
            difficulties: Array of ground truth difficulty values
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for training tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]
        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from embeddings")

        if not available_tasks:
            print("Warning: No tasks with embeddings found")
            return self

        # Build training matrix
        X = np.stack([self._embeddings[t] for t in available_tasks])
        y = np.array([difficulties[task_ids.index(t)] for t in available_tasks])

        # Fit StandardScaler + Ridge
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self.model.fit(X, y)
        print(f"Embedding prior trained on {len(available_tasks)} tasks")

        return self

    def predict(self, task_ids: List[str]) -> np.ndarray:
        """Predict difficulty for tasks."""
        if self.model is None or self._embeddings is None:
            return np.zeros(len(task_ids))

        # Get embeddings for prediction tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]

        if not available_tasks:
            return np.zeros(len(task_ids))

        X = np.stack([self._embeddings[t] for t in available_tasks])
        predictions = self.model.predict(X)

        # Return predictions aligned with input task_ids
        result = np.zeros(len(task_ids))
        pred_dict = dict(zip(available_tasks, predictions))
        for i, t in enumerate(task_ids):
            if t in pred_dict:
                result[i] = pred_dict[t]

        return result

    def get_prior_predictions(self, task_ids: List[str]) -> Dict[str, float]:
        """Get prior predictions as a dict."""
        predictions = self.predict(task_ids)
        return dict(zip(task_ids, predictions))

    def get_feature_coefficients(self) -> Dict[str, float]:
        """Get coefficients (not very interpretable for embeddings)."""
        if self.model is None:
            return {}
        # Just return summary stats since embedding dims aren't interpretable
        ridge = self.model.named_steps["ridge"]
        return {
            "n_features": len(ridge.coef_),
            "coef_mean": float(np.mean(ridge.coef_)),
            "coef_std": float(np.std(ridge.coef_)),
            "coef_max": float(np.max(np.abs(ridge.coef_))),
        }

    def get_prior_features(self, task_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get the raw embedding features used by the prior model.

        Returns the task embeddings (before scaling) that were used to train the prior.

        Args:
            task_ids: List of task IDs to get features for

        Returns:
            Dict mapping task_id -> embedding vector (raw, unscaled)
        """
        if self._embeddings is None:
            return {}

        return {
            task_id: self._embeddings[task_id]
            for task_id in task_ids
            if task_id in self._embeddings
        }

    def get_prior_feature_dim(self) -> int:
        """Get the dimensionality of prior input features (embedding dim)."""
        return self._embedding_dim if self._embedding_dim is not None else 0
