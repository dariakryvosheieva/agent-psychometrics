"""Inspect task definition for environment feature extraction.

This task uses dynamic sandboxing to run each SWE-bench instance in its own
Docker container, then extracts deterministic features from the environment.

Usage:
    # Test on 2 tasks
    inspect eval experiment_a/env_features/inspect_task.py --limit 2

    # With parallelism
    inspect eval experiment_a/env_features/inspect_task.py --limit 10 --max-connections 10
"""

import platform
import sys
import tempfile
from pathlib import Path

# Add project root to path so we can import experiment_a modules
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_evals.utils.huggingface import hf_dataset

from experiment_a.env_features.extractor_solver import env_feature_extractor


def get_swebench_image_name(instance_id: str) -> str:
    """Get Docker image name for a SWE-bench instance.

    Uses the prebuilt 'eval' images which have the repo already at /testbed.
    NOT the 'env' images which only have dependencies.

    NOTE: Always use x86_64 images because arm64 images don't exist for all
    instances on DockerHub. On Apple Silicon, Docker will use Rosetta emulation.
    """
    # SWE-bench uses _1776_ as a separator in image names (historical quirk)
    updated_id = instance_id.replace("__", "_1776_")

    # Always use x86_64 - arm64 images are not available for all instances
    # Docker Desktop on Apple Silicon will use Rosetta emulation automatically
    arch = "x86_64"

    return f"swebench/sweb.eval.{arch}.{updated_id}:latest"


def get_sandbox_config(instance_id: str) -> str:
    """Generate Docker compose config for a SWE-bench instance.

    Returns path to a temporary compose.yaml file that Inspect will use.
    """
    image_name = get_swebench_image_name(instance_id)

    # Create compose config with root user access
    content = f"""services:
  default:
    image: {image_name}
    command: "sleep infinity"
    working_dir: /testbed
    user: root
    deploy:
      resources:
        limits:
          cpus: '1'
"""

    # Write to temp file (Inspect reads from file path)
    config_dir = Path(tempfile.gettempdir()) / "env_features_configs"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / f"{instance_id}-compose.yaml"
    config_file.write_text(content)

    return str(config_file)


@task
def env_feature_extraction(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> Task:
    """Extract environment features from SWE-bench tasks.

    Args:
        dataset: HuggingFace dataset name
        split: Dataset split (default: test)

    Returns:
        Inspect Task configured with dynamic sandboxing per instance
    """
    # Load dataset from HuggingFace
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
            ],
        ),
    )

    # Add sandbox config to each sample (dynamic per-instance Docker image)
    for sample in samples:
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=get_sandbox_config(str(sample.id)),
        )

    return Task(
        dataset=samples,
        solver=env_feature_extractor(),
        scorer=None,  # No scoring needed - we just extract features
        name="swe_bench_env_features",
    )


@task
def env_feature_extraction_mini(
    split: str = "test",
) -> Task:
    """Extract features from SWE-bench verified mini (smaller dataset for testing)."""
    return env_feature_extraction(
        dataset="MariusHobbhahn/swe-bench-verified-mini",
        split=split,
    )
