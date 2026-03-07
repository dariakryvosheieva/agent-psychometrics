"""Auditor agent for task difficulty assessment.

This module provides an LLM-based agent that explores task environments
via Docker shell access and rates them on difficulty-related axes.
"""

from llm_judge_feature_extraction.task_context import build_auditor_system_prompt

__all__ = [
    "build_auditor_system_prompt",
]
