"""Configuration for Experiment Pass@K."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentPassKConfig:
    """Configuration for Pass@K signal analysis experiment."""

    # Data paths
    items_path: Path = Path("clean_data/swebench_verified_20251120_full/1d/items.csv")
    responses_path: Path = Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    output_dir: Path = Path("chris_output/experiment_pass_at_k")

    # Experiment parameters
    n_tasks: int = 5
    k_attempts: int = 10

    # Models (date-checkpointed versions)
    # M1: Best reasoning model before M1 cutoff (Dec 12, 2024)
    # M2: Best reasoning model before M2 cutoff (July 20, 2025)
    m1_model: str = "openai/o1-2024-12-17"
    m2_model: str = "openai/o3-2025-04-16"

    # Inspect eval settings
    # message_limit: 500 allows agents to use as many messages as needed
    # (99th percentile in training data is 496)
    message_limit: int = 500
    sandbox: str = "docker"  # Use docker sandbox for SWE-bench evaluation

    # Task selection criteria
    # Select tasks where strong models typically fail
    max_pass_rate_for_selection: float = 0.3  # Max 30% pass rate among top agents
    min_overall_pass_rate: float = 0.05  # At least 5% overall (not impossible)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentPassKConfig":
        """Create config from dict, converting strings to Paths."""
        path_fields = {"items_path", "responses_path", "output_dir"}
        converted = {}
        for k, v in d.items():
            if k in path_fields and isinstance(v, str):
                converted[k] = Path(v)
            else:
                converted[k] = v
        return cls(**converted)
