"""Main training and evaluation pipeline for Experiment A (TerminalBench).

This is a thin wrapper around the shared pipeline in experiment_a.shared.pipeline.

Supports two modes:
- Binomial (default): Uses k/n successes per agent-task pair
- Binary (--binary flag): Uses collapsed binary (any success = 1)
"""

from pathlib import Path
from typing import Any, Dict, List

from experiment_a.terminalbench.config import TerminalBenchConfig
from experiment_a.terminalbench.data_loader import load_task_data_from_repo
from experiment_a.shared.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[2]

# TerminalBench-specific LLM judge features (4 pre-selected features)
# Pre-selected subset that works well with Ridge regression (verified by comparing
# Ridge-only vs Lasso+Ridge performance). The full 8 features extracted by
# experiment_a/terminalbench/llm_judge_prompt.py are available, but using all 8
# with Ridge-only hurts performance compared to this pre-selected subset.
TERMINALBENCH_LLM_JUDGE_FEATURES = [
    "task_clarity",
    "domain_knowledge_required",
    "task_complexity",
    "atypicality",
]


def get_spec(use_binary: bool) -> ExperimentSpec:
    """Get experiment specification based on binary/binomial mode.

    Args:
        use_binary: If True, use binary mode (any success = 1).
                   If False, use binomial mode (k/n successes).

    Returns:
        ExperimentSpec configured for the appropriate mode.
    """
    if use_binary:
        return ExperimentSpec(
            name="TerminalBench (Binary)",
            is_binomial=False,  # Use binary IRT
            irt_cache_dir=ROOT / "chris_output" / "experiment_a_terminalbench_binary" / "irt_splits",
            llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
        )
    else:
        return ExperimentSpec(
            name="TerminalBench",
            is_binomial=True,  # Use binomial IRT
            irt_cache_dir=ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits",
            llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
        )


# Default spec (binomial mode)
SPEC = get_spec(use_binary=False)


def create_metadata_loader(config: TerminalBenchConfig):
    """Create a metadata loader that loads task data from the terminal-bench repo.

    Args:
        config: TerminalBench configuration containing repo_path

    Returns:
        Callable that loads task metadata from the repo
    """
    repo_path = ROOT / config.repo_path

    def loader(task_ids: List[str]) -> Dict[str, Any]:
        return {"task_data": load_task_data_from_repo(task_ids, repo_path)}

    return loader


def main():
    """Run Experiment A on TerminalBench.

    Default is binomial mode (k/n successes per agent-task pair).
    Use --binary flag to use collapsed binary mode (any success = 1).
    """
    run_experiment_main(
        TerminalBenchConfig,
        SPEC,  # Default spec (binomial) - will be overridden if spec_factory is used
        ROOT,
        metadata_loader_factory=create_metadata_loader,
        spec_factory=get_spec,  # Pass the factory for dynamic spec selection
    )


if __name__ == "__main__":
    main()
