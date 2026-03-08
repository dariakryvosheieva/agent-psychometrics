"""Shared Docker/sandbox utilities for experiment_a.

Provides sandbox configuration and Docker cleanup functions used across
multiple sub-modules (env_features, auditor_agent, swebench_pro, terminalbench, gso).
"""

import subprocess
import tempfile
from pathlib import Path


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


def get_sandbox_config(
    instance_id: str,
    image_name: str,
    working_dir: str = "/testbed",
) -> str:
    """Generate Docker compose config for a task instance.

    Args:
        instance_id: Unique identifier for this task (used for temp file naming).
        image_name: Docker image name.
        working_dir: Working directory inside the container (default: /testbed).

    Returns:
        Path to a temporary compose.yaml file that Inspect will use.
    """
    # Create compose config with root user access.
    # Override entrypoint to /bin/sh because some images (e.g. SWE-bench Pro
    # NodeBB) have a broken /bin/bash. Use "sleep infinity" via /bin/sh -c
    # so it works even when the default entrypoint is broken.
    content = f"""services:
  default:
    image: {image_name}
    entrypoint: ["/bin/sh", "-c"]
    command: ["sleep infinity"]
    working_dir: {working_dir}
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



def run_docker_cleanup(remove_images: bool = True):
    """Clean up Docker state to free memory.

    Args:
        remove_images: If True, remove all unused images (frees ~500MB per image)
    """
    print("\n--- Cleaning Docker state ---")

    # Force-remove ALL containers (running + stopped), including Compose ones
    result = subprocess.run(
        ["docker", "ps", "-aq"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        container_ids = result.stdout.strip().split("\n")
        print(f"  Force-removing {len(container_ids)} containers...")
        subprocess.run(
            ["docker", "rm", "-f"] + container_ids, capture_output=True
        )

    # Remove orphaned networks (Compose leaves these behind)
    subprocess.run(
        ["docker", "network", "prune", "-f"],
        capture_output=True,
        text=True,
    )
    print("  Containers and networks cleaned")

    # Remove unused volumes
    result = subprocess.run(
        ["docker", "volume", "prune", "-f"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"  Volumes pruned")

    # Remove all unused images
    if remove_images:
        result = subprocess.run(
            ["docker", "image", "prune", "-af"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"  Images pruned: {result.stdout.strip().splitlines()[-1]}")

    # Final prune to clean up any dangling layers
    subprocess.run(
        ["docker", "system", "prune", "-f"],
        capture_output=True,
        text=True,
    )

    print("--- Docker cleanup complete ---\n")
