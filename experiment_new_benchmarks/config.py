"""Configuration for held-out benchmark generalization experiments."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_new_tasks.config import DATASET_DEFAULTS as TASK_DATASET_DEFAULTS


ALL_DATASETS = ["swebench_verified", "swebench_pro", "terminalbench", "gso"]
DEFAULT_HELDOUT_DATASETS = ["swebench_pro", "gso"]

DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    dataset: {
        "display_name": TASK_DATASET_DEFAULTS[dataset]["display_name"],
        "responses_path": TASK_DATASET_DEFAULTS[dataset]["responses_path"],
        "embeddings_path": TASK_DATASET_DEFAULTS[dataset]["embeddings_path"],
        "llm_judge_features_path": TASK_DATASET_DEFAULTS[dataset]["llm_judge_features_path"],
    }
    for dataset in ALL_DATASETS
}

_PATH_FIELDS = {
    "output_dir",
}

_DICT_OF_PATHS_FIELDS = {
    "responses_paths",
    "embeddings_paths",
    "llm_judge_features_paths",
}


@dataclass
class ExperimentNewBenchmarksConfig:
    """Shared config for train-on-three, evaluate-on-one benchmark experiments."""

    output_dir: Path = Path("output/experiment_new_benchmarks")
    responses_paths: Dict[str, Path] = field(
        default_factory=lambda: {
            dataset: DATASET_DEFAULTS[dataset]["responses_path"] for dataset in ALL_DATASETS
        }
    )
    embeddings_paths: Dict[str, Path] = field(
        default_factory=lambda: {
            dataset: DATASET_DEFAULTS[dataset]["embeddings_path"] for dataset in ALL_DATASETS
        }
    )
    llm_judge_features_paths: Dict[str, Path] = field(
        default_factory=lambda: {
            dataset: DATASET_DEFAULTS[dataset]["llm_judge_features_path"]
            for dataset in ALL_DATASETS
        }
    )
    ridge_alphas: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )
    irt_epochs: int = 2000
    irt_device: str = "cuda"
    irt_lr: float = 0.01
    irt_model: str = "1d_1pl"
    theta_combine: str = "sum"
    split_seed: int = 0

    @property
    def irt_cache_dir(self) -> Path:
        return self.output_dir / "irt_splits"

    @classmethod
    def with_overrides(cls, **overrides: Any) -> "ExperimentNewBenchmarksConfig":
        return cls(**overrides)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
            elif key in _DICT_OF_PATHS_FIELDS and isinstance(value, dict):
                d[key] = {dataset: str(path) for dataset, path in value.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentNewBenchmarksConfig":
        converted: Dict[str, Any] = {}
        for key, value in d.items():
            if key in _PATH_FIELDS and isinstance(value, str):
                converted[key] = Path(value)
            elif key in _DICT_OF_PATHS_FIELDS and isinstance(value, dict):
                converted[key] = {dataset: Path(path) for dataset, path in value.items()}
            else:
                converted[key] = value
        return cls(**converted)


def display_name_for_dataset(dataset: str) -> str:
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset!r}. Valid: {list(DATASET_DEFAULTS)}")
    return str(DATASET_DEFAULTS[dataset]["display_name"])


def expand_dataset_path_template(
    template: Optional[str],
    *,
    defaults: Dict[str, Path],
) -> Dict[str, Path]:
    if template is None:
        return dict(defaults)
    return {dataset: Path(template.replace("{dataset}", dataset)) for dataset in ALL_DATASETS}

