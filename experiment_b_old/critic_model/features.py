"""
Feature definitions for OpenHands Critic Model rewards.

Extracts temporal features from per-step reward predictions V_t
to capture trajectory dynamics for difficulty prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Feature names for the critic model output
CRITIC_FEATURE_NAMES = [
    "mean_reward",           # avg(V_t) - overall trajectory quality
    "min_reward",            # min(V_t) - worst point in trajectory
    "max_reward",            # max(V_t) - best point in trajectory
    "final_reward",          # V_T - final state value
    "initial_reward",        # V_0 - starting state value
    "max_drop",              # min(V_{t+1} - V_t) - the "death drop"
    "max_drop_step_frac",    # t/T where max drop occurred (normalized)
    "reward_std",            # σ(V_t) - resilience/variance measure
    "trend_slope",           # linear regression slope of V_t
    "late_collapse",         # mean(V_T-k:T) - mean(V_0:k) for late drops
    "early_promise",         # max(V_t for t < T/4) - did it start well?
    "recovery_count",        # number of times V increases after decrease
]


@dataclass
class CriticFeatures:
    """Container for critic model features extracted from a trajectory."""

    mean_reward: float
    min_reward: float
    max_reward: float
    final_reward: float
    initial_reward: float
    max_drop: float
    max_drop_step_frac: float
    reward_std: float
    trend_slope: float
    late_collapse: float
    early_promise: float
    recovery_count: float

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array in feature name order."""
        return np.array([
            self.mean_reward,
            self.min_reward,
            self.max_reward,
            self.final_reward,
            self.initial_reward,
            self.max_drop,
            self.max_drop_step_frac,
            self.reward_std,
            self.trend_slope,
            self.late_collapse,
            self.early_promise,
            self.recovery_count,
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "CriticFeatures":
        """Create from numpy array."""
        return cls(
            mean_reward=float(vec[0]),
            min_reward=float(vec[1]),
            max_reward=float(vec[2]),
            final_reward=float(vec[3]),
            initial_reward=float(vec[4]),
            max_drop=float(vec[5]),
            max_drop_step_frac=float(vec[6]),
            reward_std=float(vec[7]),
            trend_slope=float(vec[8]),
            late_collapse=float(vec[9]),
            early_promise=float(vec[10]),
            recovery_count=float(vec[11]),
        )


def featurize_rewards(rewards: np.ndarray) -> CriticFeatures:
    """
    Extract temporal features from a per-step reward vector.

    Args:
        rewards: Array of shape (T,) containing V_t values for each step

    Returns:
        CriticFeatures dataclass with extracted features
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    T = len(rewards)

    if T == 0:
        # Return zeros for empty trajectories
        return CriticFeatures(
            mean_reward=0.0,
            min_reward=0.0,
            max_reward=0.0,
            final_reward=0.0,
            initial_reward=0.0,
            max_drop=0.0,
            max_drop_step_frac=0.0,
            reward_std=0.0,
            trend_slope=0.0,
            late_collapse=0.0,
            early_promise=0.0,
            recovery_count=0.0,
        )

    if T == 1:
        # Single step trajectory
        val = float(rewards[0])
        return CriticFeatures(
            mean_reward=val,
            min_reward=val,
            max_reward=val,
            final_reward=val,
            initial_reward=val,
            max_drop=0.0,
            max_drop_step_frac=0.0,
            reward_std=0.0,
            trend_slope=0.0,
            late_collapse=0.0,
            early_promise=val,
            recovery_count=0.0,
        )

    # Basic statistics
    mean_reward = float(np.mean(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    final_reward = float(rewards[-1])
    initial_reward = float(rewards[0])
    reward_std = float(np.std(rewards))

    # Compute step-to-step differences
    diffs = np.diff(rewards)

    # Max drop (most negative difference)
    max_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    max_drop_idx = int(np.argmin(diffs)) if len(diffs) > 0 else 0
    max_drop_step_frac = max_drop_idx / (T - 1) if T > 1 else 0.0

    # Trend slope via linear regression
    t = np.arange(T, dtype=np.float32)
    if T > 1:
        # Simple linear regression: slope = cov(t, rewards) / var(t)
        t_centered = t - t.mean()
        r_centered = rewards - rewards.mean()
        trend_slope = float(np.sum(t_centered * r_centered) / np.sum(t_centered ** 2))
    else:
        trend_slope = 0.0

    # Late collapse: compare last quarter to first quarter
    quarter = max(1, T // 4)
    early_mean = float(np.mean(rewards[:quarter]))
    late_mean = float(np.mean(rewards[-quarter:]))
    late_collapse = late_mean - early_mean  # Negative = collapsed

    # Early promise: max reward in first quarter
    early_promise = float(np.max(rewards[:quarter]))

    # Recovery count: number of increases after decreases
    recovery_count = 0.0
    if len(diffs) > 1:
        for i in range(1, len(diffs)):
            if diffs[i] > 0 and diffs[i-1] < 0:
                recovery_count += 1

    return CriticFeatures(
        mean_reward=mean_reward,
        min_reward=min_reward,
        max_reward=max_reward,
        final_reward=final_reward,
        initial_reward=initial_reward,
        max_drop=max_drop,
        max_drop_step_frac=max_drop_step_frac,
        reward_std=reward_std,
        trend_slope=trend_slope,
        late_collapse=late_collapse,
        early_promise=early_promise,
        recovery_count=recovery_count,
    )


def load_critic_rewards(filepath: Path) -> Optional[np.ndarray]:
    """
    Load raw per-step rewards from a saved NPZ file.

    Args:
        filepath: Path to the .npz file

    Returns:
        Array of rewards or None if file doesn't exist
    """
    if not filepath.exists():
        return None
    try:
        data = np.load(filepath)
        return data["rewards"]
    except (IOError, KeyError):
        return None


def load_critic_features_for_task(
    task_id: str,
    agents: List[str],
    critic_rewards_dir: Path,
) -> Dict[str, CriticFeatures]:
    """
    Load and featurize critic rewards for all agents on a task.

    Args:
        task_id: SWE-bench task ID
        agents: List of agent IDs to load
        critic_rewards_dir: Directory containing per-agent reward files

    Returns:
        Dict mapping agent_id -> CriticFeatures
    """
    result = {}
    for agent in agents:
        reward_file = critic_rewards_dir / agent / f"{task_id}.npz"
        rewards = load_critic_rewards(reward_file)
        if rewards is not None:
            features = featurize_rewards(rewards)
            result[agent] = features
    return result


def aggregate_critic_features(
    features: Dict[str, CriticFeatures],
) -> np.ndarray:
    """
    Aggregate critic features across multiple agents.

    Computes mean and std for each feature, resulting in 2x the number
    of base features.

    Args:
        features: Dict mapping agent_id -> CriticFeatures

    Returns:
        Array of shape (2 * len(CRITIC_FEATURE_NAMES),) with mean and std
    """
    if not features:
        # Return zeros if no features available
        n_features = len(CRITIC_FEATURE_NAMES)
        return np.zeros(2 * n_features, dtype=np.float32)

    # Stack all feature vectors
    vectors = np.stack([f.to_vector() for f in features.values()], axis=0)

    # Compute mean and std across agents
    mean_features = np.mean(vectors, axis=0)
    std_features = np.std(vectors, axis=0)

    # Concatenate mean and std
    return np.concatenate([mean_features, std_features]).astype(np.float32)


# Extended feature names including aggregation
CRITIC_FEATURE_NAMES_AGGREGATED = (
    [f"{name}_mean" for name in CRITIC_FEATURE_NAMES] +
    [f"{name}_std" for name in CRITIC_FEATURE_NAMES]
)
