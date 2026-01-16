"""Configuration for trajectory summarization pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SummarizationConfig:
    """Configuration for trajectory summarization."""

    # Model settings
    # Qwen3-Coder-30B-A3B is MoE with 30B total params, 3B active per forward pass
    # Inference speed ~= 3B dense model, but needs 30B memory (fine for H200)
    model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    quantization: str = "fp8"  # FP8 for H200
    tensor_parallel_size: int = 1  # Single GPU per instance (data parallelism)
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 40  # Concurrent sequences for continuous batching (limited by KV cache at 128K)

    # Context limits
    max_model_len: int = 131072  # Qwen3-Coder supports 256K; 128K captures 95% of trajectories fully
    max_output_tokens: int = 700  # Target ~500 tokens, buffer for safety

    # Data paths
    trajectory_dir: str = "trajectory_data/unified_trajs"
    output_dir: str = "chris_output/trajectory_summaries"

    # Sharding for data parallelism
    shard_id: int = 0
    num_shards: int = 1

    # Processing settings
    batch_size: int = 16  # Requests to submit at once
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
