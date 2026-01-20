"""Extract simple features from agent trajectories."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrajectoryFeatures:
    """Simple trajectory features."""

    message_count: int
    total_chars: int
    assistant_messages: int
    user_messages: int
    avg_message_length: float
    resolved: bool


# Feature names for the aggregated feature vector
TRAJECTORY_FEATURE_NAMES = [
    "avg_message_count",
    "avg_total_chars",
    "avg_assistant_ratio",
    "avg_message_length",
    "resolved_rate",
]


def extract_trajectory_features(trajectory_path: Path) -> Optional[TrajectoryFeatures]:
    """Extract simple features from a unified trajectory JSON.

    Args:
        trajectory_path: Path to trajectory JSON file

    Returns:
        TrajectoryFeatures or None if file doesn't exist
    """
    if not trajectory_path.exists():
        return None

    try:
        with open(trajectory_path) as f:
            traj = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    messages = traj.get("messages", [])
    if not messages:
        return TrajectoryFeatures(
            message_count=0,
            total_chars=0,
            assistant_messages=0,
            user_messages=0,
            avg_message_length=0.0,
            resolved=traj.get("resolved", False),
        )

    total_chars = sum(len(m.get("content", "")) for m in messages)
    assistant_msgs = sum(1 for m in messages if m.get("role") == "assistant")
    user_msgs = sum(1 for m in messages if m.get("role") == "user")

    return TrajectoryFeatures(
        message_count=len(messages),
        total_chars=total_chars,
        assistant_messages=assistant_msgs,
        user_messages=user_msgs,
        avg_message_length=total_chars / len(messages) if messages else 0,
        resolved=traj.get("resolved", False),
    )


def load_trajectories_for_task(
    task_id: str,
    agents: List[str],
    trajectories_dir: Path,
) -> Dict[str, TrajectoryFeatures]:
    """Load trajectory features for a task across multiple agents.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        trajectories_dir: Base directory for trajectories

    Returns:
        Dict mapping agent -> TrajectoryFeatures
    """
    result = {}
    for agent in agents:
        traj_path = trajectories_dir / agent / f"{task_id}.json"
        features = extract_trajectory_features(traj_path)
        if features is not None:
            result[agent] = features
    return result


def aggregate_trajectory_features(trajectories: Dict[str, TrajectoryFeatures]) -> np.ndarray:
    """Aggregate features across multiple trajectories.

    Returns feature vector: [avg_msg_count, avg_total_chars, avg_assistant_ratio, avg_msg_length, resolved_rate]
    """
    if not trajectories:
        return np.zeros(len(TRAJECTORY_FEATURE_NAMES))

    msg_counts = [t.message_count for t in trajectories.values()]
    total_chars = [t.total_chars for t in trajectories.values()]
    assistant_ratios = [
        t.assistant_messages / t.message_count if t.message_count > 0 else 0
        for t in trajectories.values()
    ]
    avg_lengths = [t.avg_message_length for t in trajectories.values()]
    resolved_rates = [1 if t.resolved else 0 for t in trajectories.values()]

    return np.array(
        [
            np.mean(msg_counts),
            np.mean(total_chars),
            np.mean(assistant_ratios),
            np.mean(avg_lengths),
            np.mean(resolved_rates),
        ]
    )


def load_and_aggregate_for_tasks(
    task_ids: List[str],
    agents: List[str],
    trajectories_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load and aggregate trajectory features for multiple tasks.

    Args:
        task_ids: List of task IDs
        agents: List of agent names whose trajectories to use
        trajectories_dir: Base directory for trajectories

    Returns:
        Dict mapping task_id -> aggregated feature vector
    """
    result = {}
    for task_id in task_ids:
        trajs = load_trajectories_for_task(task_id, agents, trajectories_dir)
        if trajs:
            result[task_id] = aggregate_trajectory_features(trajs)
    return result
