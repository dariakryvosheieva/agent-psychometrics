"""Trajectory summarization pipeline for SWE-bench agent trajectories.

Uses vLLM with Qwen3-8B-Instruct to summarize full agent trajectories into
concise summaries (~500 tokens) that capture difficulty-relevant signals.

The summaries are designed for downstream use in:
1. Embedding as features for difficulty prediction
2. Training SAD-IRT models with trajectory context
"""

from .config import SummarizationConfig
from .data_loader import discover_trajectories, load_trajectory, format_trajectory
from .prompt import format_summarization_prompt

__all__ = [
    "SummarizationConfig",
    "discover_trajectories",
    "load_trajectory",
    "format_trajectory",
    "format_summarization_prompt",
]
