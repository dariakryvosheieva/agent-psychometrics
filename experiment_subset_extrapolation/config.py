"""Configuration for the subset extrapolation experiment.

Wraps `ExperimentAConfig` from `experiment_new_tasks/` so we reuse all dataset
paths (responses, full IRT) while adding the sweep-specific settings (subset
counts, number of seeds, methods, output dir).

A "cell" is one (dataset, count, seed) configuration. `count` is the absolute
number of observed target-benchmark tasks (not a fraction) — chosen this way
so the small-subset regime is legible and the step (2 tasks) is meaningful.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_new_tasks.config import DATASET_DEFAULTS, ExperimentAConfig


DEFAULT_DATASETS: Tuple[str, ...] = (
    "swebench_verified", "swebench_pro", "gso", "terminalbench",
)

# Sweep: start at the lowest count that still trains the multi-bench IRT
# cleanly (2 — re-run a few seeds at 2 and bump if Pyro fails), step by 2, cap
# at ~20% of the benchmark (the regime where the empirical baseline is
# already strong).
DEFAULT_SUBSET_COUNTS_BY_DATASET: Dict[str, Tuple[int, ...]] = {
    "swebench_verified": tuple(range(2, 101, 2)),   # 50 counts (~20% of 500)
    "swebench_pro":      tuple(range(2, 147, 2)),   # 73 counts (~20% of 730)
    "terminalbench":     tuple(range(2, 19, 2)),    #  9 counts (~20% of 89)
    "gso":               tuple(range(2, 21, 2)),    # 10 counts (~20% of 102)
}

DEFAULT_METHODS: Tuple[str, ...] = (
    "empirical",
    "combined_calibrated",
    "oracle",
)

SUPPORTED_METHODS: Tuple[str, ...] = (
    "empirical",
    "combined_calibrated",
    "oracle",
)


@dataclass
class SubsetExtrapolationConfig:
    """Sweep settings + per-dataset path config.

    Per-dataset paths are resolved via `ExperimentAConfig.for_dataset(dataset)`
    so we automatically inherit any changes made there.
    """

    output_root: Path = Path("output/experiment_subset_extrapolation")
    subset_counts_by_dataset: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SUBSET_COUNTS_BY_DATASET)
    )
    methods: Tuple[str, ...] = DEFAULT_METHODS
    datasets: Tuple[str, ...] = DEFAULT_DATASETS
    target_n_seeds: int = 20
    seed_start: int = 0
    max_seed_attempts_per_cell: Optional[int] = None
    irt_epochs: int = 5000
    irt_lr: float = 0.01
    irt_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_seed_attempts_per_cell is None:
            # 3x the target covers worst-case Pyro hierarchical-1PL failures.
            self.max_seed_attempts_per_cell = 3 * self.target_n_seeds
        for d in self.datasets:
            if d not in DATASET_DEFAULTS:
                raise ValueError(
                    f"Unknown dataset {d!r}; valid: {list(DATASET_DEFAULTS.keys())}"
                )
            if d not in self.subset_counts_by_dataset:
                raise ValueError(
                    f"Dataset {d!r} has no entry in subset_counts_by_dataset"
                )
            for c in self.subset_counts_by_dataset[d]:
                if int(c) < 1:
                    raise ValueError(f"Subset count must be >= 1, got {c} for {d!r}")
        for m in self.methods:
            if m not in SUPPORTED_METHODS:
                raise ValueError(f"Unknown method {m!r}; valid: {SUPPORTED_METHODS}")

    def base_config(self, dataset: str) -> ExperimentAConfig:
        """Resolve dataset paths via experiment_new_tasks config."""
        return ExperimentAConfig.for_dataset(dataset)

    def dataset_output_dir(self, dataset: str) -> Path:
        return self.output_root / dataset

    def cache_dir_for(self, dataset: str, count: int, seed: int) -> Path:
        """Per-(count, seed) IRT cache directory.

        Each (count, seed) gets its own directory: the target subset is
        determined by (count, seed) and feeds directly into the IRT training
        data, so a shared cache would silently return a stale IRT.
        """
        return self.dataset_output_dir(dataset) / "irt_splits" / f"count{int(count):04d}_seed{int(seed)}"

    def counts_for(self, dataset: str) -> Tuple[int, ...]:
        return self.subset_counts_by_dataset[dataset]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "subset_counts_by_dataset": {k: list(v) for k, v in self.subset_counts_by_dataset.items()},
            "methods": list(self.methods),
            "datasets": list(self.datasets),
            "target_n_seeds": self.target_n_seeds,
            "seed_start": self.seed_start,
            "max_seed_attempts_per_cell": self.max_seed_attempts_per_cell,
            "irt_epochs": self.irt_epochs,
            "irt_lr": self.irt_lr,
            "irt_device": self.irt_device,
        }
