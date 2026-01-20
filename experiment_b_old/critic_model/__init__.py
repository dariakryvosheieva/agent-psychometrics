"""
OpenHands Critic Model integration for Experiment B.

Extracts per-step reward predictions (V_t) from agent trajectories
using the OpenHands critic model, then featurizes the value curves
to predict IRT task difficulty.

Usage:
    # Compute rewards on GPU cluster
    python -m experiment_b.critic_model.compute_rewards \
        --trajectories_dir trajectory_data/unified_trajs \
        --output_dir chris_output/experiment_b/critic_rewards

    # Evaluate features
    python -m experiment_b.train_evaluate --feature_source critic_model
"""

from .features import (
    CRITIC_FEATURE_NAMES,
    CRITIC_FEATURE_NAMES_AGGREGATED,
    CriticFeatures,
    featurize_rewards,
    aggregate_critic_features,
    load_critic_features_for_task,
)

__all__ = [
    "CRITIC_FEATURE_NAMES",
    "CRITIC_FEATURE_NAMES_AGGREGATED",
    "CriticFeatures",
    "featurize_rewards",
    "aggregate_critic_features",
    "load_critic_features_for_task",
]
