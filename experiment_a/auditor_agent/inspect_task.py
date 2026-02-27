"""Inspect task definitions for the auditor agent.

Uses an LLM-based agent with bash access to audit task environments and rate
them on difficulty-related axes. Supports multiple datasets:
- SWE-bench Verified & Pro (bug fixes)
- Terminal Bench (terminal automation)
- GSO (performance optimization)

Usage:
    # V4 auditor on SWE-bench Verified (default)
    inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_task_v4 \\
        --model anthropic/claude-opus-4-5-20251101 --limit 1

    # V4 on specific instance
    inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_task_v4 \\
        --model anthropic/claude-opus-4-5-20251101 --sample-id django__django-11099

    # Legacy V3 auditor
    inspect eval experiment_a/auditor_agent/inspect_task.py@auditor_task_v3 \\
        --model anthropic/claude-opus-4-5-20251101 --limit 2
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
from experiment_a.auditor_agent.prompts_v3 import build_auditor_system_prompt_v3
from experiment_a.auditor_agent.prompts_v4 import build_auditor_system_prompt_v4


def load_hf_samples_with_sandbox(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    input_field: str = "problem_statement",
    id_field: str = "instance_id",
    metadata_fields: list[str] | None = None,
    image_name_fn=None,
    working_dir: str = "/testbed",
):
    """Load a HuggingFace dataset and attach Docker sandbox configs to each sample.

    Args:
        dataset: HuggingFace dataset path.
        split: Dataset split (default: test).
        input_field: Field name for the task input/problem statement.
        id_field: Field name for the task ID.
        metadata_fields: Additional fields to include as metadata. If None,
            uses the default SWE-bench fields.
        image_name_fn: Optional callable(sample) -> str that returns the Docker
            image name for a sample. If None, uses SWE-bench naming convention.
        working_dir: Working directory inside the container.

    Returns:
        List of Inspect samples with sandbox configs attached.
    """
    if metadata_fields is None:
        metadata_fields = [
            "base_commit",
            "patch",
            "repo",
            "version",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ]

    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input=input_field,
            id=id_field,
            metadata=metadata_fields,
        ),
    )

    # Add sandbox config to each sample
    for sample in samples:
        if image_name_fn is not None:
            image = image_name_fn(sample)
            sample.sandbox = SandboxEnvironmentSpec(
                type="docker",
                config=get_sandbox_config(
                    str(sample.id), image_name=image, working_dir=working_dir
                ),
            )
        else:
            sample.sandbox = SandboxEnvironmentSpec(
                type="docker",
                config=get_sandbox_config(str(sample.id), working_dir=working_dir),
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
    samples = load_hf_samples_with_sandbox(dataset, split)

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
    samples = load_hf_samples_with_sandbox(dataset, split)

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
    samples = load_hf_samples_with_sandbox(dataset, split)

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


@task
def auditor_task_v3(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run the v3 auditor agent on SWE-bench tasks.

    V3 features (5 total, all showing strong correlation with IRT difficulty):
    - entry_point_clarity (-0.502)
    - change_blast_radius (+0.502)
    - test_feedback_quality (-0.301)
    - fix_localization (-0.587) - strongest predictor
    - debugging_setup_ease (-0.305)

    Dropped from v2: test_specificity (weak at -0.065)

    Args:
        dataset: HuggingFace dataset name
        split: Dataset split (default: test)
        max_attempts: Max submissions (we only want 1 - the JSON output)
        message_limit: Max total messages in conversation

    Returns:
        Inspect Task configured with v3 auditor agent
    """
    samples = load_hf_samples_with_sandbox(dataset, split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt_v3()),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 5 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="swe_bench_auditor_v3",
    )


@task
def auditor_task_v4(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    task_type: str = "swebench",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run the V4 auditor agent — multi-dataset superset with 8 features.

    V4 features (8 total):
    Retained from V3:
    - fix_localization, entry_point_clarity, change_blast_radius

    New for multi-dataset coverage:
    - environment_setup_complexity, implementation_language_complexity,
      testing_infrastructure_quality, dependency_complexity, codebase_scale

    Args:
        dataset: HuggingFace dataset name.
        split: Dataset split (default: test).
        task_type: One of "swebench", "swebench_pro", "terminalbench", "gso".
            Controls task-type-specific context in the prompt.
        max_attempts: Max submissions (we only want 1 - the JSON output).
        message_limit: Max total messages in conversation.

    Returns:
        Inspect Task configured with V4 auditor agent.
    """
    samples = load_hf_samples_with_sandbox(dataset, split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt_v4(task_type=task_type)),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name=f"auditor_v4_{task_type}",
    )
