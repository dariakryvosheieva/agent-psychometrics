"""Configuration for OpenAI API-based trajectory summarization."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SummarizationConfig:
    """Configuration for trajectory summarization using OpenAI API."""

    # Model settings
    model: str = "gpt-5-mini"
    max_output_tokens: int = 1000  # ~500 words ≈ 650-750 tokens, buffer for safety
    temperature: float = 0.3  # Low for consistent summaries

    # Parallelization - Tier 5 has 180M TPM for gpt-5-mini
    max_concurrent_requests: int = 200  # High concurrency for Tier 5
    requests_per_minute: int = 5000  # Conservative RPM (Tier 5 allows much more)

    # Data paths
    trajectory_dir: Path = Path("trajectory_data/unified_trajs")
    output_dir: Path = Path("chris_output/trajectory_summaries_api")
    checkpoint_file: Path = Path("chris_output/trajectory_summaries_api/.checkpoint.json")

    # Processing settings
    batch_size: int = 100  # Save checkpoint every N completions
    skip_existing: bool = True  # Resume capability

    # Retry settings
    max_retries: int = 3
    base_retry_delay: float = 1.0  # Exponential backoff base

    # Filtering
    agents: Optional[List[str]] = None  # None = all agents
    task_ids: Optional[List[str]] = None  # None = all tasks

    # Debug
    dry_run: bool = False
    limit: Optional[int] = None  # Limit trajectories for testing

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.trajectory_dir, str):
            self.trajectory_dir = Path(self.trajectory_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.checkpoint_file, str):
            self.checkpoint_file = Path(self.checkpoint_file)
