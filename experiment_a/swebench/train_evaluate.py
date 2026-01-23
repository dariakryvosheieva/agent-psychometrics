"""Main training and evaluation pipeline for Experiment A (SWE-bench).

This is a thin wrapper around the shared pipeline in experiment_a.shared.pipeline.
"""

from pathlib import Path

from experiment_a.swebench.config import ExperimentAConfig
from experiment_a.shared.pipeline import ExperimentSpec, run_experiment_main


# Root directory for resolving relative paths
ROOT = Path(__file__).resolve().parents[2]

# SWE-bench-specific LLM judge features (9 semantic features)
# These match the features extracted by experiment_a/swebench/llm_judge_prompt.py
SWEBENCH_LLM_JUDGE_FEATURES = [
    "fix_in_description",
    "problem_clarity",
    "error_message_provided",
    "reproduction_steps",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
]

# SWE-bench V2 LLM judge features (9 LLM + 4 deterministic = 13 features)
# Based on pilot analysis - only includes significant features (p<0.05)
SWEBENCH_LLM_JUDGE_V2_FEATURES = [
    # LLM features (9)
    "fix_in_description",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
    "verification_difficulty",
    "standard_pattern_available",
    "integration_complexity",
    # Deterministic features (4)
    "num_files_modified",
    "num_hunks",
    "num_lines_changed",
    "log_lines_changed",
]

# Experiment specification for SWE-bench
SPEC = ExperimentSpec(
    name="SWE-bench",
    is_binomial=False,  # SWE-bench uses binary 0/1 responses
    irt_cache_dir=ROOT / "chris_output" / "experiment_a" / "irt_splits",
    llm_judge_features=SWEBENCH_LLM_JUDGE_FEATURES,
)


def main():
    """Run Experiment A on SWE-bench."""
    run_experiment_main(ExperimentAConfig, SPEC, ROOT)


if __name__ == "__main__":
    main()
