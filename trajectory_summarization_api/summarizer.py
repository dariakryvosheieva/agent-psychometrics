"""API-based trajectory summarizer using OpenAI Responses API."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import SummarizationConfig
from .data_loader import TrajectoryData, format_trajectory, load_trajectory
from .openai_client import AsyncOpenAIClient
from .prompt import format_summarization_prompt

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySummary:
    """Summary of a single agent trajectory."""

    task_id: str
    agent: str
    resolved: bool
    summary: str
    metadata: dict


@dataclass
class CheckpointState:
    """Persistent state for resumable processing."""

    completed: set
    failed: set
    api_failures: set  # Track trajectories where API returned None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    started_at: str = ""
    last_updated: str = ""

    @classmethod
    def load(cls, path: Path) -> "CheckpointState":
        """Load checkpoint from file or create new one."""
        if not path.exists():
            return cls(
                completed=set(),
                failed=set(),
                api_failures=set(),
                started_at=datetime.now().isoformat(),
            )
        with open(path) as f:
            data = json.load(f)
        return cls(
            completed=set(data.get("completed", [])),
            failed=set(data.get("failed", [])),
            api_failures=set(data.get("api_failures", [])),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            started_at=data.get("started_at", datetime.now().isoformat()),
        )

    def save(self, path: Path):
        """Save checkpoint to file."""
        self.last_updated = datetime.now().isoformat()
        data = {
            "completed": list(self.completed),
            "failed": list(self.failed),
            "api_failures": list(self.api_failures),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def mark_completed(self, agent_id: str, task_id: str):
        """Mark a trajectory as completed."""
        key = f"{agent_id}/{task_id}"
        self.completed.add(key)
        self.failed.discard(key)
        self.api_failures.discard(key)

    def mark_failed(self, agent_id: str, task_id: str):
        """Mark a trajectory as failed (loading error)."""
        key = f"{agent_id}/{task_id}"
        self.failed.add(key)

    def mark_api_failure(self, agent_id: str, task_id: str):
        """Mark a trajectory where API returned None after retries."""
        key = f"{agent_id}/{task_id}"
        self.api_failures.add(key)

    def is_done(self, agent_id: str, task_id: str) -> bool:
        """Check if trajectory is already completed."""
        key = f"{agent_id}/{task_id}"
        return key in self.completed


class TrajectorySummarizer:
    """Summarize trajectories using OpenAI Responses API."""

    def __init__(self, config: SummarizationConfig):
        """Initialize the summarizer.

        Args:
            config: Summarization configuration
        """
        self.config = config
        self.checkpoint = CheckpointState.load(config.checkpoint_file)
        self.swebench_data = self._load_swebench_data()

        if not config.dry_run:
            self.client = AsyncOpenAIClient(
                model=config.model,
                max_concurrent=config.max_concurrent_requests,
                requests_per_minute=config.requests_per_minute,
                max_retries=config.max_retries,
                base_retry_delay=config.base_retry_delay,
            )
        else:
            self.client = None

    def _load_swebench_data(self) -> dict:
        """Load SWE-bench dataset for problem statements."""
        try:
            from datasets import load_dataset

            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            return {ex["instance_id"]: ex for ex in ds}
        except Exception as e:
            logger.warning(f"Could not load SWE-bench dataset: {e}")
            return {}

    async def summarize_single(
        self,
        agent_id: str,
        task_id: str,
        filepath: Path,
    ) -> Optional[TrajectorySummary]:
        """Summarize a single trajectory.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
            filepath: Path to trajectory JSON file

        Returns:
            TrajectorySummary or None if failed
        """
        # Load trajectory
        traj = load_trajectory(filepath)
        if traj is None:
            logger.warning(f"Failed to load trajectory: {agent_id}/{task_id}")
            return None

        # Get task context from SWE-bench
        task_info = self.swebench_data.get(task_id, {})
        problem_statement = task_info.get("problem_statement", "")
        repo = task_info.get("repo", "unknown")

        # Format trajectory text (with truncation for very long trajectories)
        trajectory_text = format_trajectory(
            traj.messages,
            max_chars=self.config.max_trajectory_chars,
        )

        # Build prompt
        prompt = format_summarization_prompt(
            task_id=task_id,
            repo=repo,
            resolved=traj.resolved,
            problem_statement=problem_statement,
            trajectory_text=trajectory_text,
        )

        # Call API (note: temperature not supported for gpt-5-mini)
        summary_text = await self.client.summarize(
            prompt,
            max_tokens=self.config.max_output_tokens,
        )

        if summary_text is None:
            logger.error(
                f"API returned None for {agent_id}/{task_id} after {self.config.max_retries} retries"
            )
            return None

        return TrajectorySummary(
            task_id=task_id,
            agent=agent_id,
            resolved=traj.resolved,
            summary=summary_text,
            metadata={
                "model": self.config.model,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def save_summary(self, summary: TrajectorySummary) -> Path:
        """Save a single summary to disk.

        Args:
            summary: The summary to save

        Returns:
            Path to the saved file
        """
        agent_dir = self.config.output_dir / summary.agent
        agent_dir.mkdir(parents=True, exist_ok=True)

        output_file = agent_dir / f"{summary.task_id}.json"
        with open(output_file, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        return output_file

    async def process_batch(
        self,
        items: List[tuple],
    ) -> tuple:
        """Process a batch of trajectories concurrently.

        Args:
            items: List of (agent_id, task_id, filepath) tuples

        Returns:
            Tuple of (successes, failures, api_failures)
        """
        tasks = [
            self.summarize_single(agent_id, task_id, filepath)
            for agent_id, task_id, filepath in items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = 0
        failures = 0
        api_failures = 0

        for (agent_id, task_id, _), result in zip(items, results):
            if isinstance(result, Exception):
                logger.error(f"Exception processing {agent_id}/{task_id}: {result}")
                self.checkpoint.mark_failed(agent_id, task_id)
                failures += 1
            elif result is None:
                # This could be either a load failure or API failure
                # The summarize_single function logs which one it was
                self.checkpoint.mark_api_failure(agent_id, task_id)
                api_failures += 1
            else:
                self.save_summary(result)
                self.checkpoint.mark_completed(agent_id, task_id)
                successes += 1

        return successes, failures, api_failures

    def log_checkpoint_summary(self):
        """Log a summary of the checkpoint state."""
        logger.info(f"Completed: {len(self.checkpoint.completed)}")
        logger.info(f"Failed (load errors): {len(self.checkpoint.failed)}")
        logger.info(f"API failures (returned None): {len(self.checkpoint.api_failures)}")
        if self.checkpoint.api_failures:
            logger.warning(
                f"API failures need investigation: {list(self.checkpoint.api_failures)[:10]}..."
                if len(self.checkpoint.api_failures) > 10
                else f"API failures need investigation: {list(self.checkpoint.api_failures)}"
            )
