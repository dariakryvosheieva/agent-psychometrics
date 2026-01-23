"""SWE-bench Pro dataset configuration for Experiment B."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from experiment_b.shared.config_base import DatasetConfig


@dataclass
class SWEBenchProConfig(DatasetConfig):
    """Configuration for SWE-bench Pro dataset.

    SWE-bench Pro agent names don't follow the standard YYYYMMDD_ prefix pattern.
    Instead, agent release dates are stored in an auxiliary JSON file.
    """

    # Data paths
    responses_path: Path = field(
        default_factory=lambda: Path("out/chris_irt/swebench_pro.jsonl")
    )
    oracle_irt_path: Path = field(
        default_factory=lambda: Path("chris_output/swebench_pro_irt/1d/items.csv")
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path("chris_output/swebench_pro_irt/1d/abilities.csv")
    )
    # Baseline IRT is trained on-demand with proper cache validation
    baseline_irt_path: Optional[Path] = None
    embeddings_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "out/swebench_pro/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__"
            "pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__"
            "idnorm_instance-v1__ScaleAI_SWE-bench_Pro__test__n2000__maxlen8192__seed0.npz"
        )
    )
    llm_judge_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "chris_output/experiment_a_swebench_pro/llm_judge_features/llm_judge_features.csv"
        )
    )

    # Agent dates file (release dates from public announcements)
    agent_dates_path: Path = field(
        default_factory=lambda: Path("data/swebench_pro_agent_dates.json")
    )

    # Frontier split settings
    # Cutoff 2025-09-01 splits:
    #   Pre-frontier (10): GPT-4o, Kimi, Gemini 2.5 Pro, Claude Sonnet 4, Claude 4 Sonnet,
    #                      GLM-4.5, Claude Opus 4.1, GPT-5, GPT-5 High, GPT OSS
    #   Post-frontier (4): GPT-5 Codex, Claude 4.5 Sonnet, Claude 4.5 Haiku, Gemini debug-oct22
    cutoff_date: str = "20250901"

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("chris_output/experiment_b/swebench_pro")
    )

    # Cached agent dates from JSON
    _agent_dates_json: Optional[Dict[str, str]] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "SWE-bench Pro"

    def _load_agent_dates_json(self) -> Dict[str, str]:
        """Load agent dates from JSON file."""
        if self._agent_dates_json is None:
            if not self.agent_dates_path.exists():
                raise FileNotFoundError(
                    f"Agent dates file not found: {self.agent_dates_path}. "
                    "This file contains release dates for SWE-bench Pro agents."
                )
            with open(self.agent_dates_path) as f:
                self._agent_dates_json = json.load(f)
        return self._agent_dates_json

    def get_agent_dates(self, agents: List[str]) -> Dict[str, str]:
        """Get release dates from JSON file.

        SWE-bench Pro agent names don't follow the YYYYMMDD_ prefix pattern.
        Dates are loaded from data/swebench_pro_agent_dates.json, which maps
        agent names to their public release dates in YYYYMMDD format.

        Args:
            agents: List of agent IDs from response matrix

        Returns:
            Dict mapping agent_id -> date string (YYYYMMDD)
        """
        dates_json = self._load_agent_dates_json()
        agent_dates = {}
        for agent in agents:
            if agent in dates_json:
                agent_dates[agent] = dates_json[agent]
            else:
                # Log warning but don't fail - agent may have been added after dates file
                import warnings

                warnings.warn(
                    f"Agent '{agent}' not found in {self.agent_dates_path}. "
                    "This agent will be excluded from frontier analysis."
                )
        return agent_dates

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        """SWE-bench Pro LLM judge feature columns.

        Uses v5 prompt features + deterministic patch features.
        """
        return [
            # v5 LLM-extracted features
            "fix_complexity",
            "verification_difficulty",
            "standard_pattern_available",
            "integration_complexity",
            # Deterministic patch features
            "num_files_modified",
            "num_hunks",
            "num_lines_changed",
            "log_lines_changed",
        ]
