"""Load and format trajectories for API-based summarization."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """Loaded trajectory data."""

    task_id: str
    agent: str
    resolved: bool
    messages: List[Dict[str, str]]
    filepath: Path


def discover_trajectories(
    trajectory_dir: Path,
    agents: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str, Path]]:
    """Discover all trajectories with optional filtering.

    Args:
        trajectory_dir: Root directory containing agent subdirectories
        agents: Optional list of agent IDs to include (None = all)
        task_ids: Optional list of task IDs to include (None = all)

    Returns:
        List of (agent_id, task_id, filepath) tuples
    """
    results = []

    for agent_dir in sorted(trajectory_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        # Skip hidden dirs and special dirs starting with _
        if agent_dir.name.startswith(".") or agent_dir.name.startswith("_"):
            continue

        agent_id = agent_dir.name

        # Filter by agent if specified
        if agents is not None and agent_id not in agents:
            continue

        for json_file in sorted(agent_dir.glob("*.json")):
            # Skip special files starting with _
            if json_file.name.startswith("_"):
                continue
            task_id = json_file.stem

            # Filter by task if specified
            if task_ids is not None and task_id not in task_ids:
                continue

            results.append((agent_id, task_id, json_file))

    # Sort for deterministic ordering
    results = sorted(results, key=lambda x: (x[0], x[1]))

    return results


def load_trajectory(filepath: Path) -> Optional[TrajectoryData]:
    """Load a single trajectory file.

    Args:
        filepath: Path to trajectory JSON file

    Returns:
        TrajectoryData object or None if loading failed
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
        return TrajectoryData(
            task_id=data.get("task_id", ""),
            agent=data.get("agent", ""),
            resolved=data.get("resolved", False),
            messages=data.get("messages", []),
            filepath=filepath,
        )
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load trajectory {filepath}: {e}")
        return None


def format_trajectory(messages: List[Dict[str, str]]) -> str:
    """Convert trajectory messages to text format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted trajectory text with role markers
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip empty content
        if not content.strip():
            continue

        # Handle content that might be a list (normalize to string)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        parts.append(f"[{role.upper()}]\n{content}")

    return "\n\n".join(parts)
