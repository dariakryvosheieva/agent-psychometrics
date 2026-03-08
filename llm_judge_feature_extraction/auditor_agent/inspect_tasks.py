"""Inspect task definitions for the V4 auditor agent across all datasets.

Uses an LLM-based agent with bash access to audit task environments and rate
them on difficulty-related axes. Supports:
- SWE-bench Verified (bug fixes)
- SWE-bench Pro (harder bug fixes)
- Terminal Bench (terminal automation)
- GSO (performance optimization)

Usage:
    # SWE-bench Verified
    inspect eval llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_task_v4_swebench_verified \
        --model anthropic/claude-opus-4-6 --limit 1

    # SWE-bench Pro
    inspect eval llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_task_v4_swebench_pro \
        --model anthropic/claude-opus-4-6 --limit 1

    # Terminal Bench
    inspect eval llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_task_v4_terminalbench \
        --model anthropic/claude-opus-4-6 --limit 1

    # GSO
    inspect eval llm_judge_feature_extraction/auditor_agent/inspect_tasks.py@auditor_task_v4_gso \
        --model anthropic/claude-opus-4-6 --limit 1
"""

import sys
from pathlib import Path

# Add project root to path so we can import project modules
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash, python
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset

from llm_judge_feature_extraction.auditor_agent.sandbox_utils import (
    get_sandbox_config,
    get_swebench_image_name,
)
from llm_judge_feature_extraction.task_context import build_auditor_system_prompt


# =============================================================================
# Shared helpers
# =============================================================================

def load_swebench_verified_samples(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
):
    """Load SWE-bench Verified dataset with Docker sandbox configs."""
    samples = hf_dataset(
        path=dataset,
        split=split,
        revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
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

    for sample in samples:
        image = get_swebench_image_name(str(sample.id))
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(str(sample.id), image_name=image),
        )

    return samples


# =============================================================================
# SWE-bench Verified
# =============================================================================

@task
def auditor_task_v4_swebench_verified(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    task_type: str = "swebench_verified",
    max_attempts: int = 1,
    message_limit: int = 100,
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
        task_type: One of "swebench_verified", "swebench_pro", "terminalbench", "gso".
        max_attempts: Max submissions (we only want 1 - the JSON output).
        message_limit: Max total messages in conversation.
    """
    samples = load_swebench_verified_samples(dataset=dataset, split=split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt(task_type=task_type)),
        tools=[bash(timeout=240), python(timeout=240)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name=f"auditor_v4_{task_type}",
        config=GenerateConfig(max_tokens=16384),
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
        revision="7ab5114912baf22bb098818e604c02fe7ad2c11f",
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
    message_limit: int = 100,
) -> Task:
    """Run V4 auditor agent on SWE-bench Pro tasks."""
    samples = load_swebench_pro_samples(split=split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt(task_type="swebench_pro")),
        tools=[bash(timeout=240), python(timeout=240)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_swebench_pro",
        config=GenerateConfig(max_tokens=16384),
    )


# =============================================================================
# Terminal Bench
# =============================================================================

TERMINALBENCH_DOCKER_REPO = "xiangyangli"
TERMINALBENCH_DOCKER_TAG = "20260204"
DEFAULT_TB_ITEMS_PATH = _project_root / "data/terminalbench/irt/1d_1pl/items.csv"
DEFAULT_TB_REPO_PATH = _project_root / "terminal-bench-2"


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
    message_limit: int = 100,
) -> Task:
    """Run V4 auditor agent on Terminal Bench tasks."""
    samples = load_terminalbench_samples()

    auditor_agent = basic_agent(
        init=system_message(
            build_auditor_system_prompt(task_type="terminalbench")
        ),
        tools=[bash(timeout=240), python(timeout=240)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_terminalbench",
        config=GenerateConfig(max_tokens=16384),
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
    instance_id = str(sample.id).lower()
    return f"{GSO_DOCKER_REPO}:gso.eval.x86_64.{instance_id}"


def load_gso_samples(
    dataset: str = GSO_DATASET,
    split: str = "test",
    input_field: str = "prob_script",
):
    """Load GSO dataset with Docker sandbox configs.

    Args:
        dataset: HuggingFace dataset path.
        split: Dataset split.
        input_field: Which field to use as the agent's input message.
            "prob_script" (default) gives the full benchmark script (TEST-level).
            "api" gives only the function name (PROBLEM-level), used for the
            information ablation where ENVIRONMENT must sit below TEST.
    """
    # When using "api" as input, omit "prob_script" and "tests" from metadata
    # so no benchmark scripts are accessible to the agent.
    metadata = ["repo", "base_commit", "gt_diff", "hints_text"]
    if input_field == "prob_script":
        metadata.extend(["api", "tests"])

    samples = hf_dataset(
        path=dataset,
        split=split,
        revision="00b25e92aba52f9bab4026f1ecb511df40e98c67",
        sample_fields=FieldSpec(
            input=input_field,
            id="instance_id",
            metadata=metadata,
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
    message_limit: int = 100,
) -> Task:
    """Run V4 auditor agent on GSO tasks."""
    samples = load_gso_samples(split=split)

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt(task_type="gso")),
        tools=[bash(timeout=240), python(timeout=240)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_gso",
        config=GenerateConfig(max_tokens=16384),
    )


@task
def auditor_task_v4_gso_ablation(
    split: str = "test",
    max_attempts: int = 1,
    message_limit: int = 100,
) -> Task:
    """V4 auditor on GSO with PROBLEM-level input only (no benchmark script).

    For the information ablation study: the agent sees only the API function
    name, not the full benchmark script, so ENVIRONMENT sits below TEST.
    """
    samples = load_gso_samples(split=split, input_field="api")

    auditor_agent = basic_agent(
        init=system_message(build_auditor_system_prompt(task_type="gso")),
        tools=[bash(timeout=240), python(timeout=240)],
        max_attempts=max_attempts,
        message_limit=message_limit,
        submit_description="Submit your JSON audit report with all 8 features rated.",
    )

    return Task(
        dataset=samples,
        solver=auditor_agent,
        scorer=None,
        name="auditor_v4_gso_ablation",
        config=GenerateConfig(max_tokens=16384),
    )
