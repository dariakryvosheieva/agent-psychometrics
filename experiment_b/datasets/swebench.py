"""SWE-bench Verified dataset configuration for Experiment B."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from experiment_b.datasets.base import DatasetConfig


def extract_date_prefix(agent_name: str) -> str:
    """Extract YYYYMMDD date prefix from SWE-bench agent name.

    SWE-bench agent names follow the pattern: YYYYMMDD_agent_name
    e.g., "20240620_sweagent_claude3.5sonnet"

    Args:
        agent_name: Agent identifier string

    Returns:
        Date string in YYYYMMDD format, or empty string if no valid prefix
    """
    match = re.match(r"^(\d{8})_", agent_name)
    if match:
        return match.group(1)
    return ""


@dataclass
class SWEBenchConfig(DatasetConfig):
    """Configuration for SWE-bench Verified dataset.

    SWE-bench agent names include date prefixes (YYYYMMDD_agent_name),
    which are used for frontier splitting.
    """

    # Data paths
    responses_path: Path = field(
        default_factory=lambda: Path(
            "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
        )
    )
    oracle_irt_path: Path = field(
        default_factory=lambda: Path(
            "clean_data/swebench_verified_20251120_full/1d/items.csv"
        )
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path(
            "clean_data/swebench_verified_20251120_full/1d/abilities.csv"
        )
    )
    baseline_irt_path: Optional[Path] = field(
        default_factory=lambda: Path("chris_output/sad_irt/baseline_irt/items.csv")
    )
    embeddings_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/experiment_a/embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__merged.npz"
        )
    )
    llm_judge_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/experiment_a/llm_judge_features/llm_judge_features.csv"
        )
    )

    # Frontier split settings
    cutoff_date: str = "20250807"  # gpt-5-mini release date

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("chris_output/experiment_b/swebench")
    )

    @property
    def name(self) -> str:
        return "SWE-bench Verified"

    def get_agent_dates(self, agents: List[str]) -> Dict[str, str]:
        """Extract dates from agent name prefixes.

        SWE-bench agents have names like "20240620_sweagent_claude3.5sonnet".

        Args:
            agents: List of agent IDs from response matrix

        Returns:
            Dict mapping agent_id -> date string (YYYYMMDD)
        """
        agent_dates = {}
        for agent in agents:
            date = extract_date_prefix(agent)
            if date:
                agent_dates[agent] = date
        return agent_dates

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        """SWE-bench LLM judge feature columns."""
        return [
            "fix_in_description",
            "problem_clarity",
            "error_message_provided",
            "reproduction_steps",
            "fix_locality",
            "domain_knowledge_required",
            "fix_complexity",
            "logical_reasoning_required",
            "atypicality",
        ]
