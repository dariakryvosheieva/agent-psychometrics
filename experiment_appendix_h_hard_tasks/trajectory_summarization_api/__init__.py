"""Trajectory summarization pipeline using OpenAI API (GPT-5-mini).

Uses async OpenAI API calls to summarize full agent trajectories into
concise summaries (~500 words) that capture difficulty-relevant signals.

The summaries are designed for downstream use in:
1. Embedding as features for difficulty prediction
2. Training SAD-IRT models with trajectory context
"""

from .config import SummarizationConfig
from .summarizer import TrajectorySummarizer

__all__ = [
    "SummarizationConfig",
    "TrajectorySummarizer",
]
