"""Prompt configuration registry for LLM Judge feature extraction.

Provides a unified way to access prompt configurations for different datasets.
"""

from typing import Dict, List

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.prompts.swebench import SWEBENCH_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_pro import SWEBENCH_PRO_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_pro_v2 import SWEBENCH_PRO_V2_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_pro_v3 import SWEBENCH_PRO_V3_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_pro_v4 import SWEBENCH_PRO_V4_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_pro_v5 import SWEBENCH_PRO_V5_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_v2 import SWEBENCH_V2_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_v3 import SWEBENCH_V3_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_v4 import SWEBENCH_V4_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_v5 import SWEBENCH_V5_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_v6 import SWEBENCH_V6_CONFIG
from experiment_ab_shared.llm_judge.prompts.swebench_selected import SWEBENCH_SELECTED_CONFIG
from experiment_ab_shared.llm_judge.prompts.terminalbench import TERMINALBENCH_CONFIG
from experiment_ab_shared.llm_judge.prompts.terminalbench_v2 import TERMINALBENCH_V2_CONFIG
from experiment_ab_shared.llm_judge.prompts.gso import GSO_CONFIG

# Unified prompts (standardized features across all datasets)
from experiment_ab_shared.llm_judge.prompts.swebench_unified import (
    SWEBENCH_UNIFIED_CONFIG,
    SWEBENCH_UNIFIED_NO_SOLUTION_CONFIG,
    SWEBENCH_UNIFIED_PROBLEM_ONLY_CONFIG,
)
from experiment_ab_shared.llm_judge.prompts.swebench_pro_unified import (
    SWEBENCH_PRO_UNIFIED_CONFIG,
    SWEBENCH_PRO_UNIFIED_NO_SOLUTION_CONFIG,
    SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_CONFIG,
)
from experiment_ab_shared.llm_judge.prompts.terminalbench_unified import (
    TERMINALBENCH_UNIFIED_CONFIG,
    TERMINALBENCH_UNIFIED_NO_SOLUTION_CONFIG,
    TERMINALBENCH_UNIFIED_PROBLEM_ONLY_CONFIG,
)
from experiment_ab_shared.llm_judge.prompts.gso_unified import (
    GSO_UNIFIED_CONFIG,
    GSO_UNIFIED_NO_SOLUTION_CONFIG,
    GSO_UNIFIED_PROBLEM_ONLY_CONFIG,
)

# Registry of all available prompt configurations
_PROMPT_CONFIGS: Dict[str, PromptConfig] = {
    "swebench": SWEBENCH_CONFIG,
    "swebench_pro": SWEBENCH_PRO_CONFIG,
    "swebench_pro_v2": SWEBENCH_PRO_V2_CONFIG,
    "swebench_pro_v3": SWEBENCH_PRO_V3_CONFIG,
    "swebench_pro_v4": SWEBENCH_PRO_V4_CONFIG,
    "swebench_pro_v5": SWEBENCH_PRO_V5_CONFIG,
    "swebench_v2": SWEBENCH_V2_CONFIG,
    "swebench_v3": SWEBENCH_V3_CONFIG,
    "swebench_v4": SWEBENCH_V4_CONFIG,
    "swebench_v5": SWEBENCH_V5_CONFIG,
    "swebench_v6": SWEBENCH_V6_CONFIG,
    "swebench_selected": SWEBENCH_SELECTED_CONFIG,
    "terminalbench": TERMINALBENCH_CONFIG,
    "terminalbench_v2": TERMINALBENCH_V2_CONFIG,
    "gso": GSO_CONFIG,
    # Unified prompts (standardized features for fair comparison)
    "swebench_unified": SWEBENCH_UNIFIED_CONFIG,
    "swebench_unified_no_solution": SWEBENCH_UNIFIED_NO_SOLUTION_CONFIG,
    "swebench_unified_problem_only": SWEBENCH_UNIFIED_PROBLEM_ONLY_CONFIG,
    "swebench_pro_unified": SWEBENCH_PRO_UNIFIED_CONFIG,
    "swebench_pro_unified_no_solution": SWEBENCH_PRO_UNIFIED_NO_SOLUTION_CONFIG,
    "swebench_pro_unified_problem_only": SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_CONFIG,
    "terminalbench_unified": TERMINALBENCH_UNIFIED_CONFIG,
    "terminalbench_unified_no_solution": TERMINALBENCH_UNIFIED_NO_SOLUTION_CONFIG,
    "terminalbench_unified_problem_only": TERMINALBENCH_UNIFIED_PROBLEM_ONLY_CONFIG,
    "gso_unified": GSO_UNIFIED_CONFIG,
    "gso_unified_no_solution": GSO_UNIFIED_NO_SOLUTION_CONFIG,
    "gso_unified_problem_only": GSO_UNIFIED_PROBLEM_ONLY_CONFIG,
}


def get_prompt_config(name: str) -> PromptConfig:
    """Get a prompt configuration by name.

    Args:
        name: Dataset name (e.g., "swebench", "terminalbench")

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
    "SWEBENCH_CONFIG",
    "SWEBENCH_PRO_CONFIG",
    "SWEBENCH_PRO_V2_CONFIG",
    "SWEBENCH_PRO_V3_CONFIG",
    "SWEBENCH_PRO_V4_CONFIG",
    "SWEBENCH_PRO_V5_CONFIG",
    "SWEBENCH_V2_CONFIG",
    "SWEBENCH_V3_CONFIG",
    "SWEBENCH_V4_CONFIG",
    "SWEBENCH_V5_CONFIG",
    "SWEBENCH_V6_CONFIG",
    "SWEBENCH_SELECTED_CONFIG",
    "TERMINALBENCH_CONFIG",
    "TERMINALBENCH_V2_CONFIG",
    "GSO_CONFIG",
    # Unified prompts
    "SWEBENCH_UNIFIED_CONFIG",
    "SWEBENCH_UNIFIED_NO_SOLUTION_CONFIG",
    "SWEBENCH_UNIFIED_PROBLEM_ONLY_CONFIG",
    "SWEBENCH_PRO_UNIFIED_CONFIG",
    "SWEBENCH_PRO_UNIFIED_NO_SOLUTION_CONFIG",
    "SWEBENCH_PRO_UNIFIED_PROBLEM_ONLY_CONFIG",
    "TERMINALBENCH_UNIFIED_CONFIG",
    "TERMINALBENCH_UNIFIED_NO_SOLUTION_CONFIG",
    "TERMINALBENCH_UNIFIED_PROBLEM_ONLY_CONFIG",
    "GSO_UNIFIED_CONFIG",
    "GSO_UNIFIED_NO_SOLUTION_CONFIG",
    "GSO_UNIFIED_PROBLEM_ONLY_CONFIG",
]
