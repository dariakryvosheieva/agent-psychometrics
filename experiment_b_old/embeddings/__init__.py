"""Embedding-based posterior difficulty prediction.

Uses VLM trajectory embeddings to predict IRT difficulty residuals.
"""

from .aggregator import (
    AggregationType,
    EmbeddingAggregator,
    aggregate_task_embeddings,
    batch_aggregate_embeddings,
)
from .posterior_model import EmbeddingPosteriorModel

__all__ = [
    "AggregationType",
    "EmbeddingAggregator",
    "EmbeddingPosteriorModel",
    "aggregate_task_embeddings",
    "batch_aggregate_embeddings",
]
