"""SWE-bench Verified dataset configuration for Experiment B."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from experiment_appendix_h_hard_tasks.shared.config_base import DatasetConfig


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
            "data/swebench_verified/responses.jsonl"
        )
    )
    oracle_irt_path: Path = field(
        default_factory=lambda: Path(
            "data/swebench_verified/irt/1d_1pl/items.csv"
        )
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path(
            "data/swebench_verified/irt/1d_1pl/abilities.csv"
        )
    )
    # Baseline IRT is trained on-demand with proper cache validation
    # The cache key includes (responses_file, pre_frontier_agents, cutoff_date)
    # Setting a hardcoded path here would bypass cache validation and use stale data
    baseline_irt_path: Optional[Path] = None
    embeddings_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "princeton-nlp_SWE-bench_Verified__test__maxlen8192.npz"
        )
    )
    llm_judge_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/llm_judge_features/ablation_studies/swebench_ablation_controlled_v3/4_full_15.csv"
        )
    )
    trajectory_features_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/trajectory_features/aggregated_features.csv"
        )
    )

    # Frontier split settings
    cutoff_date: str = "20250501"  # After all feature model releases (Opus 4.5: 2025-11-01)

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

    @property
    def trajectory_feature_cols(self) -> Optional[List[str]]:
        """Trajectory feature columns.

        Returns None to auto-detect all numeric columns from the CSV.
        This includes mean, std, and ability_weighted aggregations for each
        of the 9 trajectory features.
        """
        return None  # Auto-detect all numeric columns
