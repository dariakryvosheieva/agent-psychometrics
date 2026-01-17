"""Iterative LLM Judge Prompt Refinement.

A simplified iterative prompt refinement system that improves LLM judge features
for difficulty prediction by:
1. Evaluating prompts cheaply on small subsets (n=30 tasks)
2. Using entropy/variance and correlation metrics for early pruning
3. Leveraging high-residual tasks to guide feature/prompt refinement
4. Using context caching to reduce API costs

Usage:
    python -m llm_judge.iterative_refinement.run_iteration --dry_run
    python -m llm_judge.iterative_refinement.run_iteration
"""

from llm_judge.iterative_refinement.config import IterativeRefinementConfig
from llm_judge.iterative_refinement.feature_metrics import (
    compute_entropy,
    compute_pairwise_correlations,
    find_redundant_features,
)
from llm_judge.iterative_refinement.prompt_store import (
    FeatureDefinition,
    PromptVersion,
    PromptStore,
)

__all__ = [
    "IterativeRefinementConfig",
    "compute_entropy",
    "compute_pairwise_correlations",
    "find_redundant_features",
    "FeatureDefinition",
    "PromptVersion",
    "PromptStore",
]