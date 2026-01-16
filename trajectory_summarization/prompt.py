"""Summarization prompt template for trajectory summarization.

This module re-exports from the shared trajectory_summarization_api for consistency.
"""

from trajectory_summarization_api.prompt import (
    SUMMARIZATION_PROMPT,
    format_summarization_prompt,
)

__all__ = [
    "SUMMARIZATION_PROMPT",
    "format_summarization_prompt",
]
