"""Configuration for the new-agents experiment.

This mirrors ``experiment_new_tasks.config`` but the unit held out by cross
validation is an agent model/scaffold pair rather than a task.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from experiment_new_tasks.config import DATASET_DEFAULTS as TASK_DATASET_DEFAULTS


DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "swebench_verified": {
        "display_name": TASK_DATASET_DEFAULTS["swebench_verified"]["display_name"],
        "responses_path": TASK_DATASET_DEFAULTS["swebench_verified"]["responses_path"],
        "output_dir": Path("output/experiment_new_agents_verified"),
    },
    "terminalbench": {
        "display_name": TASK_DATASET_DEFAULTS["terminalbench"]["display_name"],
        "responses_path": TASK_DATASET_DEFAULTS["terminalbench"]["responses_path"],
        "output_dir": Path("output/experiment_new_agents_terminalbench"),
    },
}

_PATH_FIELDS = {"responses_path", "output_dir"}


@dataclass
class ExperimentNewAgentsConfig:
    """Shared configuration for holding out unseen model/scaffold pairs."""

    display_name: str = ""
    responses_path: Path = Path("")
    output_dir: Path = Path("")
    split_seed: int = 0
    irt_epochs: int = 2000
    irt_device: str = "cuda"
    irt_lr: float = 0.01
    irt_model: str = "1d_1pl"
    theta_combine: str = "sum"

    @property
    def irt_cache_dir(self) -> Path:
        return self.output_dir / "irt_splits"

    @property
    def oracle_cache_dir(self) -> Path:
        return self.output_dir / "irt_oracle"

    @classmethod
    def for_dataset(cls, dataset: str, **overrides) -> "ExperimentNewAgentsConfig":
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(
                f"Unknown dataset: {dataset}. Valid: {list(DATASET_DEFAULTS.keys())}"
            )
        defaults = dict(DATASET_DEFAULTS[dataset])
        defaults.update(overrides)
        return cls(**defaults)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentNewAgentsConfig":
        converted = {}
        for key, value in d.items():
            if key in _PATH_FIELDS and isinstance(value, str):
                converted[key] = Path(value)
            else:
                converted[key] = value
        return cls(**converted)
