"""Inspect task definitions for the V4 auditor agent across all datasets.

Uses an LLM-based agent with bash access to audit task environments and rate
them on difficulty-related axes. Supports:
- SWE-bench Verified (bug fixes)
- SWE-bench Pro (harder bug fixes)
- Terminal Bench (terminal automation)
- GSO (performance optimization)

Usage:
    # SWE-bench Verified
    inspect eval experiment_ab_shared/llm_judge/auditor_agent/inspect_tasks.py@auditor_task_v4 \
        --model anthropic/claude-opus-4-5-20251101 --limit 1

    # SWE-bench Pro
    inspect eval experiment_ab_shared/llm_judge/auditor_agent/inspect_tasks.py@auditor_task_v4_swebench_pro \
        --model anthropic/claude-opus-4-5-20251101 --limit 1

    # Terminal Bench
    inspect eval experiment_ab_shared/llm_judge/auditor_agent/inspect_tasks.py@auditor_task_v4_terminalbench \
        --model anthropic/claude-opus-4-5-20251101 --limit 1

    # GSO
    inspect eval experiment_ab_shared/llm_judge/auditor_agent/inspect_tasks.py@auditor_task_v4_gso \
        --model anthropic/claude-opus-4-5-20251101 --limit 1
"""

import sys
from pathlib import Path

# Add project root to path so we can import project modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset

from experiment_ab_shared.llm_judge.sandbox_utils import (
    get_swebench_image_name,
    get_sandbox_config,
)
from experiment_ab_shared.llm_judge.auditor_agent.prompts_v4 import (
    build_auditor_system_prompt_v4,
)


# =============================================================================
# Shared helpers
# =============================================================================

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


# =============================================================================
# SWE-bench Verified
# =============================================================================

@task
def auditor_task_v4(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    task_type: str = "swebench",
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run the V4 auditor agent on SWE-bench Verified tasks.

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
        max_attempts: Max submissions (we only want 1 - the JSON output).
        message_limit: Max total messages in conversation.
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


# =============================================================================
# SWE-bench Pro
# =============================================================================

SWEBENCH_PRO_DATASET = "ScaleAI/SWE-bench_Pro"
SWEBENCH_PRO_DOCKER_REPO = "jefzda/sweap-images"


def _get_swebench_pro_image(sample) -> str:
    """Get Docker image name for a SWE-bench Pro sample."""
    tag = sample.metadata["dockerhub_tag"]
    return f"{SWEBENCH_PRO_DOCKER_REPO}:{tag}"


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


# =============================================================================
# Terminal Bench
# =============================================================================

TERMINALBENCH_DOCKER_REPO = "xiangyangli"
TERMINALBENCH_DOCKER_TAG = "20260204"
DEFAULT_TB_ITEMS_PATH = Path("data/terminalbench/irt/1d_1pl/items.csv")
DEFAULT_TB_REPO_PATH = Path("terminal-bench-2")


def _get_terminalbench_image(task_id: str) -> str:
    """Get Docker image name for a Terminal Bench task."""
    return f"{TERMINALBENCH_DOCKER_REPO}/{task_id}:{TERMINALBENCH_DOCKER_TAG}"


def load_terminalbench_samples(
    items_path: Path = DEFAULT_TB_ITEMS_PATH,
    repo_path: Path = DEFAULT_TB_REPO_PATH,
):
    """Load Terminal Bench tasks as Inspect samples with Docker sandbox configs.

    Task IDs come from the IRT items file. Task instructions come from
    the terminal-bench-2 repo's instruction.md files.

    Args:
        items_path: Path to IRT items.csv (provides the list of task IDs).
        repo_path: Path to the cloned terminal-bench-2 repo.

    Returns:
        List of Inspect samples with sandbox configs attached.

    Raises:
        FileNotFoundError: If items_path or repo_path doesn't exist.
        ValueError: If a task directory is missing required files.
    """
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not repo_path.exists():
        raise FileNotFoundError(
            f"terminal-bench-2 repo not found: {repo_path}. "
            f"Clone it with: git clone https://github.com/harbor-framework/terminal-bench-2"
        )

    items_df = pd.read_csv(items_path, index_col=0)
    task_ids = list(items_df.index)

    samples = []
    for task_id in task_ids:
        task_dir = repo_path / task_id
        instruction_md = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"

        if not instruction_md.exists():
            raise ValueError(
                f"instruction.md not found for '{task_id}' at {instruction_md}"
            )
        if not task_toml_path.exists():
            raise ValueError(
                f"task.toml not found for '{task_id}' at {task_toml_path}"
            )

        instruction = instruction_md.read_text(encoding="utf-8").strip()
        if not instruction:
            raise ValueError(f"Empty instruction.md for '{task_id}'")

        with open(task_toml_path, "rb") as f:
            task_toml = tomllib.load(f)
        metadata_section = task_toml.get("metadata", {})

        image = _get_terminalbench_image(task_id)

        sample = Sample(
            input=instruction,
            id=task_id,
            metadata={
                "category": metadata_section.get("category", ""),
                "tags": metadata_section.get("tags", []),
                "difficulty": metadata_section.get("difficulty", ""),
            },
        )
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(
                task_id, image_name=image, working_dir="/app"
            ),
        )
        samples.append(sample)

    return samples


@task
def auditor_task_v4_terminalbench(
    max_attempts: int = 1,
    message_limit: int = 50,
) -> Task:
    """Run V4 auditor agent on Terminal Bench tasks."""
    samples = load_terminalbench_samples()

    auditor_agent = basic_agent(
        init=system_message(
            build_auditor_system_prompt_v4(task_type="terminalbench")
        ),
        tools=[bash(timeout=120)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_terminalbench",
    )


# =============================================================================
# GSO
# =============================================================================

GSO_DATASET = "gso-bench/gso"
GSO_DOCKER_REPO = "slimshetty/gso"


def _get_gso_image(sample) -> str:
    """Get Docker image name for a GSO sample.

    GSO images follow the naming: slimshetty/gso:gso.eval.x86_64.{instance_id}
    """
    instance_id = str(sample.id)
    return f"{GSO_DOCKER_REPO}:gso.eval.x86_64.{instance_id}"


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
