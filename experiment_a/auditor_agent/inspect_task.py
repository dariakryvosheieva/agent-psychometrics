"""Inspect task definition for the auditor agent.

This task uses an LLM-based agent with bash access to audit SWE-bench
task environments and rate them on difficulty-related axes.

Usage:
    # Test on 1 task with verification prompt (use Claude Opus 4.5)
    inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_verification \\
        --model anthropic/claude-opus-4-5-20251101 --limit 1

    # Run full auditor on 2 tasks
    inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_task \\
        --model anthropic/claude-opus-4-5-20251101 --limit 2

    # Run on specific instance
    inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_task \\
        --model anthropic/claude-opus-4-5-20251101 --sample-id django__django-11099
"""

import sys
from pathlib import Path

# Add project root to path so we can import experiment_a modules
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset

# Reuse sandbox config functions from env_features
from experiment_a.env_features.inspect_task import (
    get_swebench_image_name,
    get_sandbox_config,
)

from experiment_a.auditor_agent.prompts import (
    build_auditor_system_prompt,
    VERIFICATION_PROMPT,
)
from experiment_a.auditor_agent.prompts_v2 import build_auditor_system_prompt_v2


def load_swebench_samples(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
):
    """Load SWE-bench dataset with sandbox configs."""
    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input="problem_statement",
            id="instance_id",
            metadata=[
                "base_commit",
                "patch",
                "repo",
                "version",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
            ],
        ),
    )

    # Add sandbox config to each sample
    for sample in samples:
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(str(sample.id)),
        )

    return samples


@task
def auditor_task(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run the auditor agent on SWE-bench tasks.

    The agent uses whatever model is specified via --model flag.
    Recommended: --model anthropic/claude-opus-4-5-20251101

    Args:
        dataset: HuggingFace dataset name
        split: Dataset split (default: test)
        max_attempts: Max submissions (we only want 1 - the JSON output)
        message_limit: Max total messages in conversation

    Returns:
        Inspect Task configured with auditor agent
    """
    samples = load_swebench_samples(dataset, split)

    # Build the agent with bash access
    # Use init parameter to replace the default system message
    # max_attempts=1 because we only want one submission (the JSON output)
    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt()),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 6 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,  # No scoring - we extract features from completion
        name="swe_bench_auditor",
    )


@task
def auditor_verification(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> Task:
    """Verification task to test agent can run commands correctly.

    Runs a simpler prompt that asks for specific command outputs
    so we can manually verify the agent sees the right results.
    """
    samples = load_swebench_samples(dataset, split)

    verification_agent = basic_agent(
        init=system_message(VERIFICATION_PROMPT),
        tools=[bash(timeout=60)],
        max_attempts=1,
        message_limit=10,
        submit_description="Submit your JSON summary with the command results.",
    )

    return Task(
        dataset=samples,
        solver=verification_agent,
        scorer=None,
        name="swe_bench_auditor_verification",
    )


@task
def auditor_task_mini(split: str = "test") -> Task:
    """Run auditor on SWE-bench verified mini (smaller dataset for testing)."""
    return auditor_task(
        dataset="MariusHobbhahn/swe-bench-verified-mini",
        split=split,
    )


@task
def auditor_task_v2(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run the v2 auditor agent on SWE-bench tasks.

    V2 features (kept from v1):
    - entry_point_clarity, change_blast_radius, test_feedback_quality

    V2 features (new):
    - fix_localization, test_specificity, debugging_setup_ease

    Args:
        dataset: HuggingFace dataset name
        split: Dataset split (default: test)
        max_attempts: Max submissions (we only want 1 - the JSON output)
        message_limit: Max total messages in conversation

    Returns:
        Inspect Task configured with v2 auditor agent
    """
    samples = load_swebench_samples(dataset, split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt_v2()),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 6 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="swe_bench_auditor_v2",
    )
