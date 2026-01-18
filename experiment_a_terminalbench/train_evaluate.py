"""Main training and evaluation pipeline for Experiment A (TerminalBench).

This is a thin wrapper around the shared pipeline in experiment_a_common.pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List

from experiment_a_terminalbench.config import TerminalBenchConfig
from experiment_a_terminalbench.data_loader import load_task_data_from_repo
from experiment_a_common.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[1]

# TerminalBench-specific LLM judge features (8 semantic features)
# These match the features extracted by experiment_a_terminalbench/llm_judge_prompt.py
TERMINALBENCH_LLM_JUDGE_FEATURES = [
    "solution_in_instruction",
    "task_clarity",
    "solution_size",
    "domain_knowledge_required",
    "task_complexity",
    "logical_reasoning_required",
    "atypicality",
    "tooling_complexity",
]

# Experiment specification for TerminalBench
SPEC = ExperimentSpec(
    name="TerminalBench",
    is_binomial=True,  # TerminalBench uses binomial (successes/trials) responses
    irt_cache_dir=ROOT / "chris_output" / "experiment_a_terminalbench" / "irt_splits",
    llm_judge_features=TERMINALBENCH_LLM_JUDGE_FEATURES,
)


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
    """Run Experiment A on TerminalBench."""
    run_experiment_main(
        TerminalBenchConfig, SPEC, ROOT, metadata_loader_factory=create_metadata_loader
    )


if __name__ == "__main__":
    main()