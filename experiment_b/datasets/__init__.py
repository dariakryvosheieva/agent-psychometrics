"""Dataset configurations for Experiment B.

This module provides dataset-specific configurations for frontier task
difficulty prediction experiments. Each dataset has its own config class
that handles data paths, agent date extraction, and feature column mapping.

Available datasets:
- swebench: SWE-bench Verified (500 tasks, ~130 agents)
- terminalbench: TerminalBench 2.0 (89 tasks, 83 agents)
"""

from experiment_b.datasets.base import DatasetConfig
from experiment_b.datasets.swebench import SWEBenchConfig
from experiment_b.datasets.terminalbench import TerminalBenchConfig


# Registry of available dataset configurations
DATASET_CONFIGS = {
    "swebench": SWEBenchConfig,
    "terminalbench": TerminalBenchConfig,
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get a dataset configuration by name.

    Args:
        name: Dataset name (e.g., "swebench", "terminalbench")

    Returns:
        DatasetConfig instance for the specified dataset

    Raises:
        ValueError: If the dataset name is not recognized
    """
    if name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    return DATASET_CONFIGS[name]()


def list_datasets() -> list:
    """List available dataset names."""
    return list(DATASET_CONFIGS.keys())


__all__ = [
    "DatasetConfig",
    "SWEBenchConfig",
    "TerminalBenchConfig",
    "get_dataset_config",
    "list_datasets",
    "DATASET_CONFIGS",
]
