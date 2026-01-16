"""Configuration for trajectory summarization pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SummarizationConfig:
    """Configuration for trajectory summarization."""

    # Model settings
    # Using Qwen2.5-14B-Instruct (dense model) - MoE models require custom CUDA
    # kernels (_moe_C) that aren't compiled in the cluster's vLLM installation
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    quantization: str = "fp8"  # FP8 for H200
    tensor_parallel_size: int = 1  # Single GPU per instance (data parallelism)
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 16  # Concurrent sequences for continuous batching

    # Context limits
    # 32K context fits in H200 memory with model weights + KV cache
    # Longer trajectories will be truncated (keeping start + end)
    max_model_len: int = 32768
    max_output_tokens: int = 700  # Target ~500 tokens, buffer for safety

    # Data paths
    trajectory_dir: str = "trajectory_data/unified_trajs"
    output_dir: str = "chris_output/trajectory_summaries"

    # Sharding for data parallelism
    shard_id: int = 0
    num_shards: int = 1

    # Processing settings
    batch_size: int = 16  # Requests to submit at once (should match max_num_seqs)
    skip_existing: bool = True  # Resume capability

    # Logging
    log_every: int = 50

    # Debug
    dry_run: bool = False
    limit: Optional[int] = None  # Limit trajectories for testing

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.trajectory_dir, str):
            self.trajectory_dir = Path(self.trajectory_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
