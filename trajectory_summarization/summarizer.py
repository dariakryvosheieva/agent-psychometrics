"""vLLM-based trajectory summarizer."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import SummarizationConfig
from .data_loader import TrajectoryData, format_trajectory, estimate_tokens
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


class TrajectorySummarizer:
    """Summarize trajectories using vLLM."""

    def __init__(self, config: SummarizationConfig):
        """Initialize the summarizer with vLLM.

        Args:
            config: Summarization configuration
        """
        self.config = config

        if config.dry_run:
            logger.info("Dry run mode - skipping model initialization")
            self.llm = None
            self.sampling_params = None
            return

        # Import vLLM only when needed
        from vllm import LLM, SamplingParams

        logger.info(f"Loading model {config.model_name}...")
        self.llm = LLM(
            model=config.model_name,
            quantization=config.quantization,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            max_tokens=config.max_output_tokens,
            temperature=0.1,  # Low temperature for consistent summaries
            top_p=0.95,
        )

        logger.info("Model loaded successfully")

    def summarize_batch(
        self,
        trajectories: List[TrajectoryData],
    ) -> List[TrajectorySummary]:
        """Summarize a batch of trajectories.

        Args:
            trajectories: List of trajectory data to summarize

        Returns:
            List of trajectory summaries
        """
        if self.config.dry_run:
            # Return mock summaries in dry run mode
            return [
                TrajectorySummary(
                    task_id=traj.task_id,
                    agent=traj.agent,
                    resolved=traj.resolved,
                    summary="[DRY RUN - no summary generated]",
                    metadata={
                        "input_tokens": estimate_tokens(format_trajectory(traj.messages)),
                        "output_tokens": 0,
                        "model": self.config.model_name,
                        "dry_run": True,
                    },
                )
                for traj in trajectories
            ]

        # Prepare prompts
        prompts = []
        for traj in trajectories:
            # Format full trajectory (truncate if exceeds model context)
            # Leave room for prompt overhead (~500 tokens)
            max_traj_chars = (self.config.max_model_len - 500) * 4
            full_trajectory = format_trajectory(traj.messages, max_chars=max_traj_chars)

            prompt = format_summarization_prompt(
                task_id=traj.task_id,
                agent=traj.agent,
                resolved=traj.resolved,
                full_trajectory=full_trajectory,
            )
            prompts.append(prompt)

        # Run inference
        start_time = time.time()
        outputs = self.llm.generate(prompts, self.sampling_params)
        elapsed = time.time() - start_time

        # Parse results
        summaries = []
        for traj, output in zip(trajectories, outputs):
            raw_text = output.outputs[0].text.strip()

            summary = TrajectorySummary(
                task_id=traj.task_id,
                agent=traj.agent,
                resolved=traj.resolved,
                summary=raw_text,
                metadata={
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                    "processing_time_seconds": elapsed / len(trajectories),
                    "model": self.config.model_name,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            summaries.append(summary)

        return summaries

    def save_summary(self, summary: TrajectorySummary, output_dir: Path) -> Path:
        """Save a single summary to disk.

        Args:
            summary: The summary to save
            output_dir: Root output directory

        Returns:
            Path to the saved file
        """
        agent_dir = output_dir / summary.agent
        agent_dir.mkdir(parents=True, exist_ok=True)

        output_file = agent_dir / f"{summary.task_id}.json"
        with open(output_file, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        return output_file
