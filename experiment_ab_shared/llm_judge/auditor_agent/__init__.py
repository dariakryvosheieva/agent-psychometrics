"""Auditor agent for task difficulty assessment.

This module provides an LLM-based agent that explores task environments
via Docker shell access and rates them on difficulty-related axes.
"""

from experiment_ab_shared.llm_judge.auditor_agent.prompts_v4 import (
    build_auditor_system_prompt_v4,
    get_feature_names_v4,
    AUDITOR_FEATURES_V4,
)

__all__ = [
    "build_auditor_system_prompt_v4",
    "get_feature_names_v4",
    "AUDITOR_FEATURES_V4",
]
