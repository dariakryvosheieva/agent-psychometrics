"""Unified task feature sources for difficulty prediction.

This module provides a common abstraction for loading task features from
various sources (embeddings, LLM judge features, etc.). All feature sources
return a (n_tasks, feature_dim) matrix that can be used with Ridge regression.

Example usage:
    # Load embeddings
    source = EmbeddingFeatureSource(Path("embeddings.npz"))
    X = source.get_features(["task1", "task2"])  # (2, 768)

    # Load CSV features
    source = CSVFeatureSource(Path("features.csv"), ["col1", "col2"])
    X = source.get_features(["task1", "task2"])  # (2, 2)

    # Combine sources
    grouped = GroupedFeatureSource([emb_source, csv_source])
    X = grouped.get_features(["task1", "task2"])  # (2, 770)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TaskFeatureSource(ABC):
    """Abstract base class for loading task features from any format.

    All feature sources must:
    1. Load features lazily or eagerly (implementation choice)
    2. Return features for requested task IDs via get_features()
    3. Raise errors for missing tasks (fail loudly, don't skip silently)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this feature source (e.g., 'Embedding', 'LLM Judge')."""
        ...

    @property
    @abstractmethod
    def task_ids(self) -> List[str]:
        """List of all task IDs that have features available."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the feature vectors."""
        ...

    @abstractmethod
    def get_features(self, task_ids: List[str]) -> np.ndarray:
        """Get feature matrix for the given task IDs.

        Args:
            task_ids: List of task identifiers to get features for.

        Returns:
            Feature matrix of shape (len(task_ids), feature_dim).

        Raises:
            ValueError: If any task_id is not available in this source.
        """
        ...

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Optional list of feature names (for interpretability).

        Returns None for sources like embeddings where features are unnamed.
        Returns a list of column names for sources like CSV features.
        """
        return None


class EmbeddingFeatureSource(TaskFeatureSource):
    """Load task features from a .npz embedding file.

    Expected .npz format:
        - task_ids: array of task identifiers
        - X: (n_tasks, embedding_dim) matrix of embeddings

    Example:
        source = EmbeddingFeatureSource(Path("embeddings.npz"))
        X = source.get_features(["django__django-12345"])
    """

    def __init__(self, embeddings_path: Path, name: Optional[str] = None):
        """Initialize embedding feature source.

        Args:
            embeddings_path: Path to .npz file with task_ids and X arrays.
            name: Optional custom name (defaults to "Embedding").
        """
        self._embeddings_path = Path(embeddings_path)
        self._name = name or "Embedding"

        # Load embeddings
        if not self._embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self._embeddings_path}")

        data = np.load(self._embeddings_path, allow_pickle=True)

        # Extract task IDs and embedding matrix
        self._task_ids = [str(x) for x in data["task_ids"].tolist()]
        self._X = data["X"].astype(np.float32)
        self._embedding_dim = int(self._X.shape[1])

        # Build task_id -> index mapping for fast lookup
        self._task_to_idx: Dict[str, int] = {
            task_id: i for i, task_id in enumerate(self._task_ids)
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def task_ids(self) -> List[str]:
        return self._task_ids.copy()

    @property
    def feature_dim(self) -> int:
        return self._embedding_dim

    def get_features(self, task_ids: List[str]) -> np.ndarray:
        """Get embedding vectors for the given task IDs.

        Args:
            task_ids: List of task identifiers.

        Returns:
            Embedding matrix of shape (len(task_ids), embedding_dim).

        Raises:
            ValueError: If any task_id is not found in the embeddings.
        """
        # Check for missing tasks
        missing = [t for t in task_ids if t not in self._task_to_idx]
        if missing:
            raise ValueError(
                f"{len(missing)} tasks missing from embeddings. First 5: {missing[:5]}"
            )

        # Extract embeddings in order
        indices = [self._task_to_idx[t] for t in task_ids]
        return self._X[indices]


class CSVFeatureSource(TaskFeatureSource):
    """Load task features from a CSV file with named columns.

    The CSV must have an index column with task IDs and one or more
    feature columns. Common index column names: _instance_id, instance_id, task_id.

    Example:
        source = CSVFeatureSource(
            Path("llm_judge_features.csv"),
            feature_cols=["fix_complexity", "domain_knowledge_required"],
        )
        X = source.get_features(["django__django-12345"])
    """

    def __init__(
        self,
        features_path: Path,
        feature_cols: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        """Initialize CSV feature source.

        Args:
            features_path: Path to CSV file with features.
            feature_cols: List of column names to use as features. If None,
                auto-detects all numeric columns (excluding metadata columns
                starting with '_' and 'reasoning').
            name: Optional custom name (defaults to "CSV Features").

        Raises:
            FileNotFoundError: If features_path doesn't exist.
            ValueError: If any feature_cols are missing from the CSV, or if
                no numeric columns found when auto-detecting.
        """
        self._features_path = Path(features_path)
        self._name = name or "CSV Features"

        # Load CSV
        if not self._features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self._features_path}")

        df = pd.read_csv(self._features_path)

        # Set index to task/instance ID column
        # Prefer non-prefixed columns (task_id, instance_id) over prefixed ones (_task_id, _instance_id)
        # since prefixed columns are typically metadata
        if "task_id" in df.columns:
            df = df.set_index("task_id")
        elif "instance_id" in df.columns:
            df = df.set_index("instance_id")
        elif "_task_id" in df.columns:
            df = df.set_index("_task_id")
        elif "_instance_id" in df.columns:
            df = df.set_index("_instance_id")
        else:
            # Assume first column is the index
            df = df.set_index(df.columns[0])

        # Convert index to strings for consistent lookup
        df.index = df.index.astype(str)

        # Strip 'instance_' prefix from task IDs if present
        df.index = df.index.str.replace(r"^instance_", "", regex=True)

        # Auto-detect feature columns if not specified
        if feature_cols is None:
            # Find all numeric columns, excluding metadata
            feature_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and not c.startswith("_")
                and c != "reasoning"
            ]
            if not feature_cols:
                raise ValueError(
                    f"No numeric feature columns found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )
            print(f"Auto-detected {len(feature_cols)} feature columns: {feature_cols}")

        self._feature_cols = list(feature_cols)

        # Validate feature columns
        available_cols = [c for c in self._feature_cols if c in df.columns]
        missing_cols = set(self._feature_cols) - set(available_cols)
        if missing_cols:
            raise ValueError(
                f"CSV missing required feature columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        self._df = df
        self._task_ids = list(df.index)

    @property
    def name(self) -> str:
        return self._name

    @property
    def task_ids(self) -> List[str]:
        return self._task_ids.copy()

    @property
    def feature_dim(self) -> int:
        return len(self._feature_cols)

    @property
    def feature_names(self) -> Optional[List[str]]:
        return self._feature_cols.copy()

    def get_features(self, task_ids: List[str]) -> np.ndarray:
        """Get feature vectors for the given task IDs.

        Args:
            task_ids: List of task identifiers.

        Returns:
            Feature matrix of shape (len(task_ids), len(feature_cols)).

        Raises:
            ValueError: If any task_id is not found in the CSV.
        """
        # Check for missing tasks
        missing = [t for t in task_ids if t not in self._df.index]
        if missing:
            raise ValueError(
                f"{len(missing)} tasks missing from CSV features. First 5: {missing[:5]}"
            )

        # Extract features
        X = self._df.loc[task_ids, self._feature_cols].values.astype(np.float32)

        # Check for NaN values and fail loudly
        if np.any(np.isnan(X)):
            nan_mask = np.isnan(X)
            nan_rows, nan_cols = np.where(nan_mask)
            # Get the first few examples
            examples = []
            for i in range(min(5, len(nan_rows))):
                task_id = task_ids[nan_rows[i]]
                col_name = self._feature_cols[nan_cols[i]]
                examples.append(f"{task_id}:{col_name}")
            raise ValueError(
                f"Found {nan_mask.sum()} NaN values in features. "
                f"First examples: {examples}. "
                f"Either exclude these columns or ensure all tasks have values."
            )

        return X



class GroupedFeatureSource(TaskFeatureSource):
    """Combines multiple feature sources, preserving source boundaries.

    Source boundaries are used by GroupedRidgePredictor to apply per-source
    regularization via alpha grids keyed on source name.

    Example:
        grouped = GroupedFeatureSource([emb_source, csv_source])
        # Use with GroupedRidgePredictor for per-source regularization
    """

    def __init__(
        self,
        sources: List[TaskFeatureSource],
        name: Optional[str] = None,
    ):
        """Initialize grouped feature source.

        Args:
            sources: List of feature sources to combine.
            name: Optional custom name (defaults to "source1 + source2 + ...").

        Raises:
            ValueError: If sources list is empty.
        """
        if not sources:
            raise ValueError("At least one source is required")

        self._sources = sources
        self._name = name or " + ".join(s.name for s in self._sources)

        # Compute group boundaries (slices for extracting each source's features)
        self._group_slices: List[slice] = []
        offset = 0
        for s in self._sources:
            dim = s.feature_dim
            self._group_slices.append(slice(offset, offset + dim))
            offset += dim
        self._feature_dim = offset

        # Compute intersection of task IDs across all sources
        self._task_ids = self._compute_common_tasks()

    def _compute_common_tasks(self) -> List[str]:
        """Compute intersection of task IDs across all sources."""
        task_sets = [set(s.task_ids) for s in self._sources]
        common = task_sets[0].intersection(*task_sets[1:])
        return list(common)

    @property
    def name(self) -> str:
        return self._name

    @property
    def task_ids(self) -> List[str]:
        return self._task_ids.copy()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def sources(self) -> List[TaskFeatureSource]:
        """Access to underlying sources."""
        return self._sources

    @property
    def group_slices(self) -> List[slice]:
        """Slices for extracting each source's features from concatenated matrix."""
        return self._group_slices

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Concatenate feature names from all sources."""
        names = []
        for source in self._sources:
            source_names = source.feature_names
            if source_names is None:
                # For unnamed features (like embeddings), use generic names
                source_names = [f"{source.name}_{i}" for i in range(source.feature_dim)]
            names.extend(source_names)
        return names

    def get_features(self, task_ids: List[str]) -> np.ndarray:
        """Get concatenated feature vectors for the given task IDs.

        Args:
            task_ids: List of task identifiers.

        Returns:
            Feature matrix of shape (len(task_ids), sum of all feature dims).

        Raises:
            ValueError: If any task_id is missing from any source.
        """
        # Get features from each source
        feature_matrices = [s.get_features(task_ids) for s in self._sources]

        # Concatenate along feature dimension
        return np.concatenate(feature_matrices, axis=1)

    def fit_scalers(
        self, X: np.ndarray
    ) -> Tuple[Dict[str, StandardScaler], np.ndarray]:
        """Fit per-source StandardScalers on X and return them with the scaled matrix.

        Scalers are fit independently per source so that high-dim sources (e.g.
        embeddings) don't dominate the statistics of low-dim sources (e.g. LLM
        judge features).

        Returns:
            (scalers, X_std) where scalers maps source_name -> fitted StandardScaler.
        """
        scalers: Dict[str, StandardScaler] = {}
        X_std = np.empty_like(X)
        for source, slice_obj in zip(self._sources, self._group_slices):
            scaler = StandardScaler()
            X_std[:, slice_obj] = scaler.fit_transform(X[:, slice_obj])
            scalers[source.name] = scaler
        return scalers, X_std

    def apply_scalers(
        self, X: np.ndarray, scalers: Dict[str, StandardScaler]
    ) -> np.ndarray:
        """Apply pre-fitted per-source StandardScalers to X."""
        X_std = np.empty_like(X)
        for source, slice_obj in zip(self._sources, self._group_slices):
            X_std[:, slice_obj] = scalers[source.name].transform(X[:, slice_obj])
        return X_std


def build_feature_sources(
    embeddings_path: Optional[Path] = None,
    llm_judge_path: Optional[Path] = None,
    llm_judge_feature_cols: Optional[List[str]] = None,
    trajectory_features_path: Optional[Path] = None,  # Experiment B only; not used in Experiment A
    trajectory_feature_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Tuple[str, TaskFeatureSource]]:
    """Build list of available feature sources from paths.

    This is the shared utility for both Experiment A and Experiment B.

    Args:
        embeddings_path: Path to embeddings .npz file (None to skip)
        llm_judge_path: Path to LLM judge features CSV (None to skip)
        llm_judge_feature_cols: Optional list of feature columns for LLM Judge.
            If None, auto-detects numeric columns from CSV.
        trajectory_features_path: Path to trajectory features CSV (None to skip).
            Only used in Experiment B; Experiment A callers should pre-merge any
            extra feature sources into the judge CSV before passing it in.
        trajectory_feature_cols: Optional list of feature columns for trajectory.
            If None, auto-detects numeric columns from CSV.
        verbose: Print messages about missing paths (default True)

    Returns:
        List of (source_name, feature_source) tuples for each valid source.
    """
    sources: List[Tuple[str, TaskFeatureSource]] = []

    if embeddings_path and embeddings_path.exists():
        sources.append(("Embedding", EmbeddingFeatureSource(embeddings_path)))
    elif verbose and embeddings_path:
        print(f"\nEmbeddings not found: {embeddings_path}")

    if llm_judge_path and llm_judge_path.exists():
        sources.append((
            "LLM Judge",
            CSVFeatureSource(llm_judge_path, llm_judge_feature_cols, name="LLM Judge"),
        ))
    elif verbose and llm_judge_path:
        print(f"\nLLM Judge features not found: {llm_judge_path}")

    if trajectory_features_path and trajectory_features_path.exists():
        sources.append((
            "Trajectory",
            CSVFeatureSource(
                trajectory_features_path, trajectory_feature_cols, name="Trajectory"
            ),
        ))
    elif verbose and trajectory_features_path:
        print(f"\nTrajectory features not found: {trajectory_features_path}")

    return sources
