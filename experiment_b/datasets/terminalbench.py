"""TerminalBench dataset configuration for Experiment B."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

from experiment_b.datasets.base import DatasetConfig


def parse_detail_url_to_subject_id(detail_url: str) -> str:
    """Convert TerminalBench detail_url to subject_id format.

    The detail_url has format:
        https://www.tbench.ai/leaderboard/terminal-bench/2.0/{Agent}/{variant}/{model}@{provider}

    The subject_id format is:
        {agent}_{model}_at_{provider}

    Examples:
        "Factory%20Droid/unknown/gpt-5.2@openai" -> "factory_droid_gpt-5_2_at_openai"
        "ante/unknown/gemini-3-pro-preview@Google" -> "ante_gemini-3-pro-preview_at_google"

    Args:
        detail_url: Full URL from metadata

    Returns:
        subject_id string matching response matrix format
    """
    # Extract path after the version number
    # URL: https://www.tbench.ai/leaderboard/terminal-bench/2.0/{Agent}/{variant}/{model}@{provider}
    parts = detail_url.split("/terminal-bench/2.0/")
    if len(parts) != 2:
        return ""

    path = parts[1]  # e.g., "Factory%20Droid/unknown/gpt-5.2@openai"
    path_parts = path.split("/")
    if len(path_parts) < 3:
        return ""

    # URL decode and parse
    agent = unquote(path_parts[0])  # "Factory%20Droid" -> "Factory Droid"
    # variant = path_parts[1]  # "unknown" (not used in subject_id)
    model_provider = unquote(path_parts[2])  # "gpt-5.2%40openai" -> "gpt-5.2@openai"

    # Parse model@provider
    if "@" not in model_provider:
        return ""
    model, provider = model_provider.rsplit("@", 1)

    # Convert to subject_id format:
    # - Lowercase
    # - Spaces to underscores
    # - Dots to underscores
    # - @ to _at_
    agent_clean = agent.lower().replace(" ", "_")
    model_clean = model.lower().replace(".", "_")
    provider_clean = provider.lower()

    subject_id = f"{agent_clean}_{model_clean}_at_{provider_clean}"
    return subject_id


def load_terminalbench_agent_dates(metadata_path: Path) -> Dict[str, str]:
    """Load agent submission dates from TerminalBench metadata.

    Args:
        metadata_path: Path to terminal_bench_2.0.meta.json

    Returns:
        Dict mapping subject_id -> date string (YYYYMMDD)
    """
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r") as f:
        data = json.load(f)

    agent_dates = {}
    for result in data.get("results", []):
        detail_url = result.get("detail_url", "")
        date_str = result.get("agent_org", "")  # Format: "2025-12-24"

        subject_id = parse_detail_url_to_subject_id(detail_url)
        if not subject_id or not date_str:
            continue

        # Convert date from YYYY-MM-DD to YYYYMMDD
        date_clean = date_str.replace("-", "")
        if len(date_clean) == 8 and date_clean.isdigit():
            agent_dates[subject_id] = date_clean

    return agent_dates


@dataclass
class TerminalBenchConfig(DatasetConfig):
    """Configuration for TerminalBench dataset.

    TerminalBench agents have submission dates in the metadata file
    (not in the agent name). Dates are extracted from agent_org field.
    """

    # Data paths
    responses_path: Path = field(
        default_factory=lambda: Path("data/terminal_bench/terminal_bench_2.0.jsonl")
    )
    oracle_irt_path: Path = field(
        default_factory=lambda: Path("chris_output/terminal_bench_2.0/1d/items.csv")
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path("chris_output/terminal_bench_2.0/1d/abilities.csv")
    )
    metadata_path: Path = field(
        default_factory=lambda: Path("data/terminal_bench/terminal_bench_2.0.meta.json")
    )

    # No pre-computed baseline IRT for TerminalBench (will use oracle or skip)
    baseline_irt_path: Optional[Path] = None

    embeddings_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/experiment_a_terminalbench/embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__maxlen8192.npz"
        )
    )
    llm_judge_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/experiment_a_terminalbench/llm_judge_features/llm_judge_features.csv"
        )
    )

    # Frontier split settings
    # Default cutoff targets ~80% pre-frontier / ~20% post-frontier
    cutoff_date: str = "20251117"

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("chris_output/experiment_b/terminalbench")
    )

    # Cache for loaded agent dates
    _agent_dates_cache: Optional[Dict[str, str]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        return "TerminalBench"

    def get_agent_dates(self, agents: List[str]) -> Dict[str, str]:
        """Get dates from metadata file.

        TerminalBench stores submission dates in metadata JSON, not in agent names.

        Args:
            agents: List of agent IDs from response matrix

        Returns:
            Dict mapping agent_id -> date string (YYYYMMDD)
        """
        if self._agent_dates_cache is None:
            object.__setattr__(
                self,
                "_agent_dates_cache",
                load_terminalbench_agent_dates(self.metadata_path),
            )

        # Return only dates for agents in the provided list
        return {
            agent: self._agent_dates_cache[agent]
            for agent in agents
            if agent in self._agent_dates_cache
        }

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        """TerminalBench-specific LLM judge feature columns.

        These differ from SWE-bench features because the task domain is different.
        """
        return [
            "solution_in_instruction",
            "task_clarity",
            "solution_size",
            "domain_knowledge_required",
            "task_complexity",
            "logical_reasoning_required",
            "atypicality",
            "tooling_complexity",
        ]
