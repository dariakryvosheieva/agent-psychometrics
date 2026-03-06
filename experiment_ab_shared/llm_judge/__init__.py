"""LLM Judge feature extraction module.

Provides batched feature extraction from tasks using LLMs, with per-feature
info level isolation and prefix caching.

Example usage:
    from experiment_ab_shared.llm_judge import (
        BatchedFeatureExtractor,
        get_task_context,
        load_tasks,
    )

    ctx = get_task_context("swebench_verified")
    extractor = BatchedFeatureExtractor(
        feature_names=["solution_hint", "problem_clarity", "solution_complexity"],
        task_context=ctx,
    )

    tasks = load_tasks("swebench_verified")
    csv_path = extractor.run(tasks, output_dir=Path("output/"))

CLI usage:
    python -m experiment_ab_shared.llm_judge extract --all --dataset swebench_verified --dry-run
    python -m experiment_ab_shared.llm_judge extract --all --dataset terminalbench
"""

from experiment_ab_shared.llm_judge.api_client import LLMApiClient
from experiment_ab_shared.llm_judge.batched_extractor import BatchedFeatureExtractor
from experiment_ab_shared.llm_judge.feature_registry import (
    ALL_FEATURES,
    get_all_feature_names,
    get_features,
    get_features_by_level,
)
from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, InfoLevel
from experiment_ab_shared.llm_judge.response_parser import (
    parse_llm_response,
    validate_features,
)
from experiment_ab_shared.llm_judge.task_context import get_task_context
from experiment_ab_shared.llm_judge.task_loaders import load_tasks

__all__ = [
    "BatchedFeatureExtractor",
    "LLMApiClient",
    "FeatureDefinition",
    "InfoLevel",
    "ALL_FEATURES",
    "get_all_feature_names",
    "get_features",
    "get_features_by_level",
    "get_task_context",
    "load_tasks",
    "parse_llm_response",
    "validate_features",
]
