"""Inspect task definition for auditor agent on GSO (software optimization benchmark).

GSO uses Docker images from slimshetty/gso with tags following the pattern:
    gso.eval.x86_64.{instance_id}

Usage:
    inspect eval experiment_a/gso/inspect_task.py@auditor_task_v4_gso \
        --model anthropic/claude-opus-4-5-20251101 --limit 1
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset

from experiment_a.env_features.inspect_task import get_sandbox_config
from experiment_a.auditor_agent.prompts_v4 import build_auditor_system_prompt_v4


GSO_DATASET = "gso-bench/gso"
DOCKER_REPO = "slimshetty/gso"


def _get_gso_image(sample) -> str:
    """Get Docker image name for a GSO sample.

    GSO images follow the naming: slimshetty/gso:gso.eval.x86_64.{instance_id}
    """
    instance_id = str(sample.id)
    return f"{DOCKER_REPO}:gso.eval.x86_64.{instance_id}"


def load_gso_samples(
    dataset: str = GSO_DATASET,
    split: str = "test",
):
    """Load GSO dataset with Docker sandbox configs."""
    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input="prob_script",
            id="instance_id",
            metadata=[
                "repo",
                "base_commit",
                "api",
                "gt_diff",
                "hints_text",
                "tests",
            ],
        ),
    )

    for sample in samples:
        image = _get_gso_image(sample)
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(str(sample.id), image_name=image),
        )

    return samples


@task
def auditor_task_v4_gso(
    split: str = "test",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run V4 auditor agent on GSO tasks."""
    samples = load_gso_samples(split=split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt_v4(task_type="gso")),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_gso",
    )
