"""Inspect task definition for auditor agent on Terminal Bench.

Terminal Bench uses pre-built Docker images from Docker Hub at:
    xiangyangli/{task_id}:20260204

Each container provides an Ubuntu/Debian environment with the task
setup pre-configured. The agent interacts via /app as the working directory.

Task metadata (instruction, category, etc.) is loaded from the local
terminal-bench repo at terminal-bench/tasks/{task_id}/task.yaml.

Usage:
    inspect eval experiment_a/terminalbench/inspect_task.py@auditor_task_v4_terminalbench \
        --model anthropic/claude-opus-4-5-20251101 --limit 1
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import yaml

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec

from experiment_a.env_features.inspect_task import get_sandbox_config
from experiment_a.auditor_agent.prompts_v4 import build_auditor_system_prompt_v4


DOCKER_REPO = "xiangyangli"
DOCKER_TAG = "20260204"

# Default paths
DEFAULT_ITEMS_PATH = Path("chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv")
DEFAULT_REPO_PATH = Path("terminal-bench")


def _get_terminalbench_image(task_id: str) -> str:
    """Get Docker image name for a Terminal Bench task."""
    return f"{DOCKER_REPO}/{task_id}:{DOCKER_TAG}"


def load_terminalbench_samples(
    items_path: Path = DEFAULT_ITEMS_PATH,
    repo_path: Path = DEFAULT_REPO_PATH,
):
    """Load Terminal Bench tasks as Inspect samples with Docker sandbox configs.

    Task IDs come from the IRT items file. Task instructions come from
    the terminal-bench repo's task.yaml files.

    Args:
        items_path: Path to IRT items.csv (provides the list of task IDs).
        repo_path: Path to the cloned terminal-bench repo.

    Returns:
        List of Inspect samples with sandbox configs attached.

    Raises:
        FileNotFoundError: If items_path or repo_path doesn't exist.
        ValueError: If a task directory is missing its task.yaml.
    """
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Terminal-bench repo not found: {repo_path}. "
            f"Clone it with: git clone https://github.com/laude-institute/terminal-bench"
        )

    items_df = pd.read_csv(items_path, index_col=0)
    task_ids = list(items_df.index)

    samples = []
    for task_id in task_ids:
        task_dir = repo_path / "tasks" / task_id
        task_yaml_path = task_dir / "task.yaml"

        if not task_yaml_path.exists():
            raise ValueError(
                f"task.yaml not found for '{task_id}' at {task_yaml_path}"
            )

        with open(task_yaml_path) as f:
            task_yaml = yaml.safe_load(f)

        instruction = task_yaml.get("instruction", "")
        if not instruction:
            raise ValueError(f"Empty instruction in task.yaml for '{task_id}'")

        image = _get_terminalbench_image(task_id)

        sample = Sample(
            input=instruction,
            id=task_id,
            metadata={
                "category": task_yaml.get("category", ""),
                "tags": task_yaml.get("tags", []),
                "difficulty": task_yaml.get("difficulty", ""),
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
