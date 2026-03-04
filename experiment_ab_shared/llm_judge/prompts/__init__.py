"""Prompt configuration registry for LLM Judge feature extraction.

Provides a unified way to access prompt configurations for different datasets.
"""

from typing import Dict, List

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig

# Test patch features (SWE-bench with test patch analysis)
from experiment_ab_shared.llm_judge.prompts.swebench_with_test import (
    SWEBENCH_WITH_TEST_CONFIG,
)
from experiment_ab_shared.llm_judge.prompts.swebench_test_quality import (
    SWEBENCH_TEST_QUALITY_CONFIG,
)
from experiment_ab_shared.llm_judge.prompts.swebench_test_quality_no_solution import (
    SWEBENCH_TEST_QUALITY_NO_SOLUTION_CONFIG,
)

# Registry of all available prompt configurations
_PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    # Test patch features
    "swebench_with_test": SWEBENCH_WITH_TEST_CONFIG,
    # NOTE: swebench_test_quality included solution in prompt - use _no_solution for clean ablation
    "swebench_test_quality_with_solution": SWEBENCH_TEST_QUALITY_CONFIG,  # DEPRECATED: has solution leakage
    "swebench_test_quality_no_solution": SWEBENCH_TEST_QUALITY_NO_SOLUTION_CONFIG,  # Clean version for ablation
}


def get_prompt_config(name: str) -> PromptConfig:
    """Get a prompt configuration by name.

    Args:
        name: Dataset name (e.g., "swebench_with_test", "swebench_test_quality_no_solution")

    Returns:
        PromptConfig for the specified dataset

    Raises:
        ValueError: If the dataset name is not registered
    """
    if name not in _PROMPT_CONFIGS:
        available = ", ".join(_PROMPT_CONFIGS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return _PROMPT_CONFIGS[name]


def list_datasets() -> List[str]:
    """List all available dataset names.

    Returns:
        List of registered dataset names
    """
    return list(_PROMPT_CONFIGS.keys())


def register_prompt_config(name: str, config: PromptConfig) -> None:
    """Register a custom prompt configuration.

    This allows adding new datasets at runtime without modifying the registry.

    Args:
        name: Dataset name to register
        config: PromptConfig instance for the dataset

    Raises:
        ValueError: If the name is already registered
    """
    if name in _PROMPT_CONFIGS:
        raise ValueError(f"Dataset '{name}' is already registered")
    _PROMPT_CONFIGS[name] = config


__all__ = [
    "get_prompt_config",
    "list_datasets",
    "register_prompt_config",
    # Test patch features
    "SWEBENCH_WITH_TEST_CONFIG",
    "SWEBENCH_TEST_QUALITY_CONFIG",
    "SWEBENCH_TEST_QUALITY_NO_SOLUTION_CONFIG",
]
