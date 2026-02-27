"""Inspect task definition for auditor agent on SWE-bench Pro.

SWE-bench Pro uses Docker images from jefzda/sweap-images with tags
specified in the `dockerhub_tag` column of the HuggingFace dataset.

Usage:
    inspect eval experiment_a/swebench_pro/inspect_task.py@auditor_task_v4_swebench_pro \
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


SWEBENCH_PRO_DATASET = "ScaleAI/SWE-bench_Pro"
DOCKER_REPO = "jefzda/sweap-images"


def _get_swebench_pro_image(sample) -> str:
    """Get Docker image name for a SWE-bench Pro sample."""
    tag = sample.metadata["dockerhub_tag"]
    return f"{DOCKER_REPO}:{tag}"


def load_swebench_pro_samples(
    dataset: str = SWEBENCH_PRO_DATASET,
    split: str = "test",
):
    """Load SWE-bench Pro dataset with Docker sandbox configs."""
    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input="problem_statement",
            id="instance_id",
            metadata=[
                "base_commit",
                "patch",
                "test_patch",
                "repo",
                "repo_language",
                "fail_to_pass",
                "pass_to_pass",
                "dockerhub_tag",
            ],
        ),
    )

    for sample in samples:
        image = _get_swebench_pro_image(sample)
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(str(sample.id), image_name=image),
        )

    return samples


@task
def auditor_task_v4_swebench_pro(
    split: str = "test",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run V4 auditor agent on SWE-bench Pro tasks."""
    samples = load_swebench_pro_samples(split=split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt_v4(task_type="swebench_pro")),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_swebench_pro",
    )
