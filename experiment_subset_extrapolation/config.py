"""Configuration for the subset extrapolation experiment.

Wraps `ExperimentAConfig` from `experiment_new_tasks/` so we reuse all dataset
paths (responses, full IRT, embeddings, LLM judge features) while adding the
sweep-specific settings (subset sizes, number of seeds, methods, output dir).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_new_tasks.config import DATASET_DEFAULTS, ExperimentAConfig


DEFAULT_SUBSET_SIZES: Tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
)

DEFAULT_METHODS: Tuple[str, ...] = (
    "empirical",
    "combined",
    "combined_calibrated",
    "oracle",
)

# All methods this experiment knows how to evaluate.
SUPPORTED_METHODS: Tuple[str, ...] = (
    "empirical",
    "embedding",
    "embedding_calibrated",
    "combined",
    "combined_calibrated",
    "oracle",
)

DEFAULT_DATASETS: Tuple[str, ...] = (
    "swebench_verified", "swebench_pro", "gso", "terminalbench",
)

# Cells the user has explicitly excluded from the sweep because the IRT/Ridge
# training set would be too small (< ~12 tasks) for reliable estimation.
EXCLUDED_CELLS: frozenset = frozenset({
    ("terminalbench", 0.10),
})


@dataclass
class SubsetExtrapolationConfig:
    """Sweep settings + per-dataset path config.

    Per-dataset paths are resolved via `ExperimentAConfig.for_dataset(dataset)`
    so we automatically inherit any changes made there.
    """

    output_root: Path = Path("output/experiment_subset_extrapolation")
    subset_sizes: Tuple[float, ...] = DEFAULT_SUBSET_SIZES
    methods: Tuple[str, ...] = DEFAULT_METHODS
    datasets: Tuple[str, ...] = DEFAULT_DATASETS
    target_n_seeds: int = 20
    seed_start: int = 0
    max_seed_attempts_per_cell: Optional[int] = None
    excluded_cells: frozenset = EXCLUDED_CELLS

    def __post_init__(self) -> None:
        if self.max_seed_attempts_per_cell is None:
            # Allow up to 3x the target before giving up — covers the worst-case
            # Pyro hierarchical-1PL failure rate observed on TerminalBench.
            self.max_seed_attempts_per_cell = 3 * self.target_n_seeds
        for d in self.datasets:
            if d not in DATASET_DEFAULTS:
                raise ValueError(
                    f"Unknown dataset {d!r}; valid: {list(DATASET_DEFAULTS.keys())}"
                )
        for m in self.methods:
            if m not in SUPPORTED_METHODS:
                raise ValueError(f"Unknown method {m!r}; valid: {SUPPORTED_METHODS}")
        for s in self.subset_sizes:
            if not 0.0 < s <= 1.0:
                raise ValueError(f"subset_size must be in (0, 1], got {s}")

    def base_config(self, dataset: str) -> ExperimentAConfig:
        """Resolve dataset paths via experiment_new_tasks config."""
        return ExperimentAConfig.for_dataset(dataset)

    def dataset_output_dir(self, dataset: str) -> Path:
        return self.output_root / dataset

    def cache_dir_for(self, dataset: str, size: float, seed: int) -> Path:
        """Per-(size, seed) IRT cache directory.

        CRITICAL: each (size, seed) gets its own directory because
        `get_or_train_split_irt` keys its cache only by (split_seed, fold_idx,
        k_folds, model_type, exclude_unsolved) — not by the actual train_tasks
        list. A shared cache directory would silently return a stale IRT
        trained on a different subset.
        """
        size_tag = f"size{int(round(size * 1000)):04d}"
        return self.dataset_output_dir(dataset) / "irt_splits" / f"{size_tag}_seed{seed}"

    def is_excluded(self, dataset: str, size: float) -> bool:
        return (dataset, float(size)) in self.excluded_cells

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "subset_sizes": list(self.subset_sizes),
            "methods": list(self.methods),
            "datasets": list(self.datasets),
            "target_n_seeds": self.target_n_seeds,
            "seed_start": self.seed_start,
            "max_seed_attempts_per_cell": self.max_seed_attempts_per_cell,
            "excluded_cells": sorted([f"{d}@{s}" for d, s in self.excluded_cells]),
        }
