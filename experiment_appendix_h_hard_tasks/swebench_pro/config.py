"""SWE-bench Pro dataset configuration for Experiment B."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from experiment_appendix_h_hard_tasks.shared.config_base import DatasetConfig


@dataclass
class SWEBenchProConfig(DatasetConfig):
    """Configuration for SWE-bench Pro dataset.

    SWE-bench Pro agent names don't follow the standard YYYYMMDD_ prefix pattern.
    Instead, agent release dates are stored in an auxiliary JSON file.
    """

    # Data paths
    responses_path: Path = field(
        default_factory=lambda: Path("data/swebench_pro/responses.jsonl")
    )
    oracle_irt_path: Path = field(
        default_factory=lambda: Path("data/swebench_pro/irt/1d_1pl/items.csv")
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path("data/swebench_pro/irt/1d_1pl/abilities.csv")
    )
    # Baseline IRT is trained on-demand with proper cache validation
    baseline_irt_path: Optional[Path] = None
    embeddings_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v1__"
            "ScaleAI_SWE-bench_Pro__test__maxlen8192.npz"
        )
    )
    llm_judge_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "llm_judge_features/defaults/swebench_pro/llm_judge_features.csv"
        )
    )

    # Agent dates file (release dates from public announcements)
    agent_dates_path: Path = field(
        default_factory=lambda: Path("data/swebench_pro/agent_dates.json")
    )

    # Frontier split settings
    # Cutoff 2025-06-01 splits:
    #   Pre-frontier (5): GPT-4o, Kimi, Gemini 2.5 Pro Preview, Claude Sonnet 4, Claude 4 Sonnet
    #   Post-frontier (9): Gemini 2.5 Pro Preview (debug), GLM-4.5, Claude Opus 4.1,
    #                      GPT-5, GPT-5 High, GPT OSS, GPT-5 Codex, Claude 4.5 Sonnet, Claude 4.5 Haiku
    cutoff_date: str = "20250601"

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("output/experiment_b/swebench_pro")
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
        Dates are loaded from data/swebench_pro/agent_dates.json, which maps
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
        """SWE-bench Pro LLM judge feature columns (unified 15 features)."""
        return [
            "atypicality",
            "codebase_scale",
            "codebase_scope",
            "debugging_complexity",
            "domain_knowledge_required",
            "error_specificity",
            "fix_localization",
            "implementation_language_complexity",
            "logical_reasoning_required",
            "side_effect_risk",
            "similar_issue_likelihood",
            "solution_complexity",
            "solution_hint",
            "test_edge_case_coverage",
            "verification_difficulty",
        ]
