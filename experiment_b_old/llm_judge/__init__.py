"""LLM-as-judge approach for trajectory feature extraction.

Uses large language models to analyze agent trajectories and extract
features for difficulty prediction.
"""

# Export the latest version (v7) as the default
from .features_v7 import (
    LLM_JUDGE_V7_FEATURE_NAMES,
    LLMJudgeV7Features,
    load_llm_judge_v7_features,
    load_llm_judge_v7_features_for_task,
    aggregate_llm_judge_v7_features,
)

__all__ = [
    "LLM_JUDGE_V7_FEATURE_NAMES",
    "LLMJudgeV7Features",
    "load_llm_judge_v7_features",
    "load_llm_judge_v7_features_for_task",
    "aggregate_llm_judge_v7_features",
]
