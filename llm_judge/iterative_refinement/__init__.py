"""Iterative LLM Judge Prompt Refinement.

A simplified iterative prompt refinement system that improves LLM judge features
for difficulty prediction by:
1. Evaluating prompts cheaply on small subsets (n=30 tasks)
2. Using entropy/variance and correlation metrics for early pruning
3. Leveraging high-residual tasks to guide feature/prompt refinement
4. Using context caching to reduce API costs

Usage:
    # Dry run to see what would happen
    python -m llm_judge.iterative_refinement.run_iteration --dry_run

    # Run with defaults (5 iterations, n=30 tasks)
    python -m llm_judge.iterative_refinement.run_iteration

    # Custom settings
    python -m llm_judge.iterative_refinement.run_iteration \
        --max_iterations 10 \
        --quick_eval_tasks 50 \
        --model gpt-5.2
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
    create_initial_feature_schema,
    generate_prompt_from_schema,
)
from llm_judge.iterative_refinement.quick_evaluator import (
    QuickEvalResult,
    run_quick_evaluation_sync,
    stratified_sample_tasks,
)
from llm_judge.iterative_refinement.residual_analyzer import (
    ResidualAnalysis,
    HighResidualTask,
    analyze_residuals,
    format_residual_analysis_for_llm,
)
from llm_judge.iterative_refinement.prompt_refiner import (
    RefinementProposal,
    propose_refinement,
    apply_refinement_constraints,
)

__all__ = [
    # Config
    "IterativeRefinementConfig",
    # Feature metrics
    "compute_entropy",
    "compute_pairwise_correlations",
    "find_redundant_features",
    # Prompt store
    "FeatureDefinition",
    "PromptVersion",
    "PromptStore",
    "create_initial_feature_schema",
    "generate_prompt_from_schema",
    # Quick evaluator
    "QuickEvalResult",
    "run_quick_evaluation_sync",
    "stratified_sample_tasks",
    # Residual analyzer
    "ResidualAnalysis",
    "HighResidualTask",
    "analyze_residuals",
    "format_residual_analysis_for_llm",
    # Prompt refiner
    "RefinementProposal",
    "propose_refinement",
    "apply_refinement_constraints",
]