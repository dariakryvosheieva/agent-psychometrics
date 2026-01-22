"""LLM Judge feature extraction module.

This module provides unified infrastructure for extracting semantic features
from tasks using LLMs. It supports multiple datasets (SWE-bench, TerminalBench)
and LLM providers (Anthropic, OpenAI).

Example usage:
    from experiment_ab_shared.llm_judge import (
        LLMFeatureExtractor,
        get_prompt_config,
    )

    # Get prompt config for a dataset
    config = get_prompt_config("swebench")

    # Create extractor
    extractor = LLMFeatureExtractor(
        prompt_config=config,
        output_dir=Path("output"),
        provider="anthropic",
    )

    # Run extraction
    csv_path = extractor.run(tasks)

CLI usage:
    python -m experiment_ab_shared.llm_judge extract --dataset swebench --dry-run
    python -m experiment_ab_shared.llm_judge extract --dataset terminalbench
"""

from experiment_ab_shared.llm_judge.api_client import LLMApiClient
from experiment_ab_shared.llm_judge.extractor import LLMFeatureExtractor
from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, PromptConfig
from experiment_ab_shared.llm_judge.prompts import (
    get_prompt_config,
    list_datasets,
    register_prompt_config,
)
from experiment_ab_shared.llm_judge.response_parser import (
    parse_llm_response,
    validate_features,
)

__all__ = [
    # Main extractor
    "LLMFeatureExtractor",
    # Configuration
    "PromptConfig",
    "FeatureDefinition",
    # Prompt registry
    "get_prompt_config",
    "list_datasets",
    "register_prompt_config",
    # API client
    "LLMApiClient",
    # Parsing utilities
    "parse_llm_response",
    "validate_features",
]
