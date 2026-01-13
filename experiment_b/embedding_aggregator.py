"""
Embedding aggregation strategies for multi-trajectory tasks.

In Experiment B, each task has trajectories from multiple agents. This module
provides different strategies for aggregating per-agent embeddings into a single
feature vector per task.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.special import softmax

AggregationType = Literal["mean_only", "mean_std", "weighted", "all_stats"]


class EmbeddingAggregator:
    """Aggregates embeddings from multiple agents for a single task."""

    def __init__(
        self,
        aggregation: AggregationType = "mean_std",
        abilities: Optional[Dict[str, float]] = None,
    ):
        """Initialize aggregator.

        Args:
            aggregation: Aggregation strategy:
                - "mean_only": Just take mean across agents
                - "mean_std": Mean + std deviation (captures spread)
                - "weighted": Weight by agent ability (θ from IRT)
                - "all_stats": Mean + std + min + max
            abilities: Dict mapping agent_id -> theta (IRT ability).
                       Required for "weighted" aggregation.
        """
        self.aggregation = aggregation
        self.abilities = abilities or {}

        if aggregation == "weighted" and not abilities:
            raise ValueError("abilities dict required for weighted aggregation")

    def aggregate(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Aggregate embeddings from multiple agents.

        Args:
            embeddings: Dict mapping agent_id -> embedding vector

        Returns:
            Aggregated feature vector, or None if no valid embeddings
        """
        if not embeddings:
            return None

        agents = list(embeddings.keys())
        vecs = np.stack([embeddings[a] for a in agents])  # (n_agents, embedding_dim)

        if self.aggregation == "mean_only":
            return np.mean(vecs, axis=0)

        elif self.aggregation == "mean_std":
            mean = np.mean(vecs, axis=0)
            std = np.std(vecs, axis=0)
            return np.concatenate([mean, std])

        elif self.aggregation == "weighted":
            # Weight by softmax of agent abilities
            weights = np.array([
                self.abilities.get(a, 0.0) for a in agents
            ])
            weights = softmax(weights)  # Normalize to sum to 1
            weighted_mean = np.average(vecs, axis=0, weights=weights)
            return weighted_mean

        elif self.aggregation == "all_stats":
            mean = np.mean(vecs, axis=0)
            std = np.std(vecs, axis=0)
            vmin = np.min(vecs, axis=0)
            vmax = np.max(vecs, axis=0)
            return np.concatenate([mean, std, vmin, vmax])

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def get_output_dim(self, embedding_dim: int) -> int:
        """Return output dimension given embedding dimension."""
        if self.aggregation == "mean_only":
            return embedding_dim
        elif self.aggregation == "mean_std":
            return embedding_dim * 2
        elif self.aggregation == "weighted":
            return embedding_dim
        elif self.aggregation == "all_stats":
            return embedding_dim * 4
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


def load_embeddings_for_task(
    task_id: str,
    agents: List[str],
    embeddings_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load pre-computed embeddings for a task from multiple agents.

    Args:
        task_id: Task identifier
        agents: List of agent IDs to load
        embeddings_dir: Directory containing agent subdirectories with .npz files

    Returns:
        Dict mapping agent_id -> embedding vector
    """
    result = {}

    for agent in agents:
        embedding_file = embeddings_dir / agent / f"{task_id}.npz"
        if not embedding_file.exists():
            continue

        try:
            data = np.load(embedding_file)
            embedding = data["embedding"]
            result[agent] = embedding.astype(np.float32)
        except Exception:
            continue

    return result


def aggregate_task_embeddings(
    task_id: str,
    agents: List[str],
    embeddings_dir: Path,
    aggregator: EmbeddingAggregator,
) -> Optional[np.ndarray]:
    """Load and aggregate embeddings for a single task.

    Args:
        task_id: Task identifier
        agents: List of agents whose embeddings to load
        embeddings_dir: Directory containing embeddings
        aggregator: Aggregator instance

    Returns:
        Aggregated feature vector, or None if no embeddings found
    """
    embeddings = load_embeddings_for_task(task_id, agents, embeddings_dir)
    if not embeddings:
        return None
    return aggregator.aggregate(embeddings)


def batch_aggregate_embeddings(
    task_ids: List[str],
    agents: List[str],
    embeddings_dir: Path,
    aggregator: EmbeddingAggregator,
    min_agents_per_task: int = 1,
) -> Tuple[List[str], np.ndarray]:
    """Aggregate embeddings for multiple tasks.

    Args:
        task_ids: List of task identifiers
        agents: List of agents to consider
        embeddings_dir: Directory containing embeddings
        aggregator: Aggregator instance
        min_agents_per_task: Minimum number of agent embeddings required

    Returns:
        Tuple of (valid_task_ids, feature_matrix)
    """
    valid_tasks = []
    features = []

    for task_id in task_ids:
        embeddings = load_embeddings_for_task(task_id, agents, embeddings_dir)

        if len(embeddings) < min_agents_per_task:
            continue

        aggregated = aggregator.aggregate(embeddings)
        if aggregated is None:
            continue

        valid_tasks.append(task_id)
        features.append(aggregated)

    if not features:
        return [], np.array([])

    return valid_tasks, np.stack(features)


def get_embedding_stats(
    embeddings_dir: Path,
    agents: List[str],
    sample_size: int = 10,
) -> Dict:
    """Get statistics about available embeddings.

    Args:
        embeddings_dir: Directory containing embeddings
        agents: List of agents to check
        sample_size: Number of agents to sample for dim check

    Returns:
        Dict with embedding statistics
    """
    stats = {
        "agents_checked": 0,
        "agents_with_embeddings": 0,
        "embedding_dim": None,
        "total_files": 0,
    }

    for agent in agents[:sample_size]:
        agent_dir = embeddings_dir / agent
        if not agent_dir.exists():
            continue

        stats["agents_checked"] += 1
        npz_files = list(agent_dir.glob("*.npz"))

        if npz_files:
            stats["agents_with_embeddings"] += 1
            stats["total_files"] += len(npz_files)

            # Get embedding dim from first file
            if stats["embedding_dim"] is None:
                try:
                    data = np.load(npz_files[0])
                    stats["embedding_dim"] = data["embedding"].shape[0]
                except Exception:
                    pass

    return stats
