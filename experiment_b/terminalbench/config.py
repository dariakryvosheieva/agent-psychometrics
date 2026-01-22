"""TerminalBench dataset configuration for Experiment B."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

from experiment_b.shared.config_base import DatasetConfig

logger = logging.getLogger(__name__)


def parse_detail_url_to_subject_id(detail_url: str) -> str:
    """Convert TerminalBench detail_url to subject_id format.

    The detail_url has format:
        https://www.tbench.ai/leaderboard/terminal-bench/2.0/{Agent}/{variant}/{model}@{provider}

    For multi-model ensembles:
        https://www.tbench.ai/leaderboard/terminal-bench/2.0/{Agent}/{variant}/{model1}@{provider1},{model2}@{provider2}

    The subject_id format is:
        {agent}_{model}_at_{provider}

    For multi-model ensembles:
        {agent}_{model1}_at_{provider1},{model2}_at_{provider2}

    Examples:
        "Factory%20Droid/unknown/gpt-5.2@openai" -> "factory_droid_gpt-5_2_at_openai"
        "ante/unknown/gemini-3-pro-preview@Google" -> "ante_gemini-3-pro-preview_at_google"
        "warp/unknown/claude-haiku-4-5@anthropic,gpt-5.2@openai" -> "warp_claude-haiku-4-5_at_anthropic,gpt-5_2_at_openai"

    Args:
        detail_url: Full URL from metadata

    Returns:
        subject_id string matching response matrix format

    Raises:
        ValueError: If the URL cannot be parsed into a valid subject_id
    """
    # Extract path after the version number
    # URL: https://www.tbench.ai/leaderboard/terminal-bench/2.0/{Agent}/{variant}/{model}@{provider}
    parts = detail_url.split("/terminal-bench/2.0/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid TerminalBench URL format (missing /terminal-bench/2.0/): {detail_url}"
        )

    path = parts[1]  # e.g., "Factory%20Droid/unknown/gpt-5.2@openai"
    path_parts = path.split("/")
    if len(path_parts) < 3:
        raise ValueError(
            f"Invalid TerminalBench URL format (expected agent/variant/model@provider): {detail_url}"
        )

    # URL decode and parse
    agent = unquote(path_parts[0])  # "Factory%20Droid" -> "Factory Droid"
    # variant = path_parts[1]  # "unknown" (not used in subject_id)
    model_provider = unquote(path_parts[2])  # "gpt-5.2%40openai" -> "gpt-5.2@openai"

    # Parse model@provider - handle multi-model ensembles (comma-separated)
    if "@" not in model_provider:
        raise ValueError(
            f"Invalid TerminalBench URL format (missing @ in model@provider): {detail_url}"
        )

    # Check if this is a multi-model ensemble (contains commas)
    if "," in model_provider:
        # Multi-model format: model1@provider1,model2@provider2,...
        model_parts = []
        invalid_parts = []
        for mp in model_provider.split(","):
            if "@" not in mp:
                invalid_parts.append(mp)
                continue
            model, provider = mp.rsplit("@", 1)
            model_clean = model.lower().replace(".", "_")
            provider_clean = provider.lower()
            model_parts.append(f"{model_clean}_at_{provider_clean}")

        if invalid_parts:
            logger.warning(
                f"Skipped {len(invalid_parts)} invalid model parts in ensemble URL {detail_url}: {invalid_parts}"
            )

        if not model_parts:
            raise ValueError(
                f"Invalid TerminalBench URL format (no valid model@provider parts in ensemble): {detail_url}"
            )

        agent_clean = agent.lower().replace(" ", "_")
        subject_id = f"{agent_clean}_{','.join(model_parts)}"
    else:
        # Single model format: model@provider
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

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If results in metadata have missing required fields
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"TerminalBench metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        raise ValueError(f"TerminalBench metadata has no 'results' field: {metadata_path}")

    agent_dates = {}
    skipped_no_url = 0
    skipped_no_date = 0
    skipped_invalid_date = 0

    for i, result in enumerate(results):
        detail_url = result.get("detail_url", "")
        date_str = result.get("agent_org", "")  # Format: "2025-12-24"

        if not detail_url:
            skipped_no_url += 1
            continue

        if not date_str:
            skipped_no_date += 1
            continue

        # parse_detail_url_to_subject_id will raise ValueError on invalid URLs
        subject_id = parse_detail_url_to_subject_id(detail_url)

        # Convert date from YYYY-MM-DD to YYYYMMDD
        date_clean = date_str.replace("-", "")
        if len(date_clean) != 8 or not date_clean.isdigit():
            skipped_invalid_date += 1
            logger.warning(
                f"Invalid date format for agent {subject_id}: '{date_str}' (expected YYYY-MM-DD)"
            )
            continue

        agent_dates[subject_id] = date_clean

    # Log summary of skipped entries
    if skipped_no_url or skipped_no_date or skipped_invalid_date:
        logger.warning(
            f"Skipped entries when loading TerminalBench agent dates: "
            f"no_url={skipped_no_url}, no_date={skipped_no_date}, invalid_date={skipped_invalid_date}"
        )

    if not agent_dates:
        raise ValueError(
            f"No valid agent dates found in TerminalBench metadata. "
            f"Total results: {len(results)}, skipped: {skipped_no_url + skipped_no_date + skipped_invalid_date}"
        )

    return agent_dates


@dataclass
class TerminalBenchConfig(DatasetConfig):
    """Configuration for TerminalBench dataset.

    TerminalBench agents have submission dates in the metadata file
    (not in the agent name). Dates are extracted from agent_org field.

    Data format options:
    - Binary (default): Uses binarized pass@5 data (any success → 1)
    - Binomial (opt-in): Uses raw data with full trial counts (k successes out of n trials)

    Binary is the default because empirical testing showed slightly better ROC-AUC
    on the same frontier task set (18 tasks, pass-rate definition):
      - Oracle: 0.8224 (binary) vs 0.7832 (binomial)
      - Feature-IRT: 0.7417 (binary) vs 0.7191 (binomial)

    To use binomial likelihood, override paths to point to raw data:
        responses_path = Path("data/terminal_bench/terminal_bench_2.0_raw.jsonl")
        oracle_irt_path = Path("chris_output/terminal_bench_2.0_binomial/1d_1pl/items.csv")
        oracle_abilities_path = Path("chris_output/terminal_bench_2.0_binomial/1d_1pl/abilities.csv")
    """

    # Data paths - default to binarized data (binary likelihood)
    responses_path: Path = field(
        default_factory=lambda: Path("data/terminal_bench/terminal_bench_2.0.jsonl")
    )
    # IRT trained with binary likelihood on binarized data
    oracle_irt_path: Path = field(
        default_factory=lambda: Path("chris_output/terminal_bench_2.0/1d_1pl/items.csv")
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path("chris_output/terminal_bench_2.0/1d_1pl/abilities.csv")
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
    # Default cutoff targets ~58% pre-frontier / ~42% post-frontier
    # Earlier cutoff gives more frontier tasks for IRT-based evaluation
    cutoff_date: str = "20251105"

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

        Raises:
            ValueError: If any agent is missing from the metadata dates
        """
        if self._agent_dates_cache is None:
            object.__setattr__(
                self,
                "_agent_dates_cache",
                load_terminalbench_agent_dates(self.metadata_path),
            )

        # Check for missing agents
        missing_agents = [a for a in agents if a not in self._agent_dates_cache]
        if missing_agents:
            raise ValueError(
                f"{len(missing_agents)} agents missing from TerminalBench metadata dates. "
                f"First 5: {missing_agents[:5]}"
            )

        # Return dates for all agents
        return {agent: self._agent_dates_cache[agent] for agent in agents}

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        """TerminalBench-specific LLM judge feature columns.

        Pre-selected subset of 4 features that work well with Ridge regression
        (verified by comparing Ridge-only vs Lasso+Ridge performance).
        These differ from SWE-bench features because the task domain is different.
        """
        return [
            "task_clarity",
            "domain_knowledge_required",
            "task_complexity",
            "atypicality",
        ]
