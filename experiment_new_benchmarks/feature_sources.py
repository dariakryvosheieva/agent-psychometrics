"""Cross-benchmark task feature sources with benchmark-prefixed task IDs."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiment_new_responses.dataset import benchmark_key_for_dataset
from experiment_new_tasks.feature_source import (
    CSVFeatureSource,
    EmbeddingFeatureSource,
    TaskFeatureSource,
)


def prefixed_task_id(benchmark: str, task_id: str) -> str:
    return f"{benchmark}::{task_id}"


class PrefixedMultiBenchmarkFeatureSource(TaskFeatureSource):
    """Expose per-benchmark feature files as one feature source.

    Task IDs are namespaced as ``benchmark::task_id`` so tasks from different
    benchmarks cannot collide while training a single cross-benchmark regressor.
    """

    def __init__(
        self,
        sources_by_dataset: Dict[str, TaskFeatureSource],
        *,
        name: str,
    ):
        if not sources_by_dataset:
            raise ValueError("At least one benchmark feature source is required")
        self._sources_by_benchmark = {
            benchmark_key_for_dataset(dataset): source
            for dataset, source in sources_by_dataset.items()
        }
        dims = {source.feature_dim for source in self._sources_by_benchmark.values()}
        if len(dims) != 1:
            raise ValueError(f"Feature dimensions differ across benchmarks: {sorted(dims)}")
        names_by_benchmark = {
            benchmark: tuple(source.feature_names or ())
            for benchmark, source in self._sources_by_benchmark.items()
        }
        named = {benchmark: names for benchmark, names in names_by_benchmark.items() if names}
        if named and len(set(named.values())) != 1:
            raise ValueError(
                f"Feature names differ across benchmarks for {name}: {names_by_benchmark}"
            )
        self._feature_dim = int(next(iter(dims)))
        self._name = str(name)
        self._task_ids: List[str] = []
        for benchmark, source in self._sources_by_benchmark.items():
            self._task_ids.extend(
                prefixed_task_id(benchmark, task_id) for task_id in source.task_ids
            )
        self._task_id_set = set(self._task_ids)

    @property
    def name(self) -> str:
        return self._name

    @property
    def task_ids(self) -> List[str]:
        return list(self._task_ids)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def feature_names(self) -> Optional[List[str]]:
        first = next(iter(self._sources_by_benchmark.values()))
        return first.feature_names

    def get_features(self, task_ids: List[str]) -> np.ndarray:
        missing = [task_id for task_id in task_ids if task_id not in self._task_id_set]
        if missing:
            raise ValueError(
                f"{len(missing)} tasks missing from {self.name} features. First 5: {missing[:5]}"
            )

        rows: List[np.ndarray] = []
        for task_id in task_ids:
            benchmark, raw_task_id = _split_prefixed_task_id(task_id)
            source = self._sources_by_benchmark[benchmark]
            rows.append(source.get_features([raw_task_id])[0])
        return np.stack(rows, axis=0).astype(np.float32)


def build_embedding_source(
    embeddings_paths: Dict[str, Path],
    datasets: List[str],
) -> Optional[PrefixedMultiBenchmarkFeatureSource]:
    sources = {
        dataset: EmbeddingFeatureSource(Path(embeddings_paths[dataset]))
        for dataset in datasets
        if dataset in embeddings_paths and Path(embeddings_paths[dataset]).exists()
    }
    if not sources:
        return None
    if set(sources) != set(datasets):
        missing = sorted(set(datasets) - set(sources))
        raise FileNotFoundError(f"Missing embedding features for datasets: {missing}")
    return PrefixedMultiBenchmarkFeatureSource(sources, name="Embedding")


def build_judge_source(
    llm_judge_features_paths: Dict[str, Path],
    datasets: List[str],
) -> Optional[PrefixedMultiBenchmarkFeatureSource]:
    sources = {
        dataset: CSVFeatureSource(
            Path(llm_judge_features_paths[dataset]),
            name="LLM Judge",
        )
        for dataset in datasets
        if dataset in llm_judge_features_paths
        and Path(llm_judge_features_paths[dataset]).exists()
    }
    if not sources:
        return None
    if set(sources) != set(datasets):
        missing = sorted(set(datasets) - set(sources))
        raise FileNotFoundError(f"Missing LLM judge features for datasets: {missing}")
    return PrefixedMultiBenchmarkFeatureSource(sources, name="LLM Judge")


def _split_prefixed_task_id(task_id: str) -> tuple[str, str]:
    parts = str(task_id).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"Task ID is not benchmark-prefixed: {task_id!r}")
    return parts[0], parts[1]

