"""TerminalBench dataset configuration for Experiment B."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

from experiment_appendix_h_hard_tasks.shared.config_base import DatasetConfig

logger = logging.getLogger(__name__)

# Known model prefixes for parsing agent IDs
MODEL_PREFIXES = [
    "accounts/fireworks/models/",
    "openai/",
    "moonshotai/",
    "qwen/",
    "claude-",
    "gemini-",
    "gpt-",
    "grok-",
    "minimax-",
]


def extract_models_from_agent_id(agent_id: str) -> List[str]:
    """Extract model names from a TerminalBench agent ID.

    Agent IDs have the format:
        {agent_name}_{model}_at_{provider}
        {agent_name}_{model1}_at_{provider1},{model2}_at_{provider2}  (multi-model)

    Args:
        agent_id: Agent ID from response matrix (e.g., "terminus-2_gpt-5_2_at_openai")

    Returns:
        List of model names (e.g., ["gpt-5_2"])

    Raises:
        ValueError: If model cannot be extracted from agent ID
    """
    models = []

    # Handle multi-model (comma separated)
    parts = agent_id.split(",")
    is_first = True

    for part in parts:
        # Find _at_ to split model from provider
        if "_at_" not in part:
            continue

        # Everything before the last _at_ is agent+model (for first) or just model (for rest)
        idx = part.rfind("_at_")
        model_part = part[:idx]  # e.g., 'terminus-2_gpt-5_2' or 'gpt-5_2'

        found = False

        # For non-first parts in multi-model, the model name is the entire model_part
        if not is_first:
            models.append(model_part)
            found = True
        else:
            # For first part, find the model name using known prefixes
            # Look for the LAST occurrence of a known prefix after an underscore
            best_match = None
            best_idx = -1

            for prefix in MODEL_PREFIXES:
                # Find all occurrences of this prefix
                search_idx = 0
                while True:
                    prefix_idx = model_part.find(prefix, search_idx)
                    if prefix_idx == -1:
                        break

                    # Check if this occurrence is valid (preceded by underscore)
                    if prefix_idx > 0 and model_part[prefix_idx - 1] == "_":
                        # Take the last valid occurrence for this prefix
                        if prefix_idx > best_idx:
                            best_idx = prefix_idx
                            best_match = model_part[prefix_idx:]

                    search_idx = prefix_idx + 1

            if best_match:
                models.append(best_match)
                found = True

        if not found:
            raise ValueError(
                f"Could not extract model from agent ID part: '{part}' "
                f"(full agent_id: '{agent_id}')"
            )

        is_first = False

    if not models:
        raise ValueError(f"No models found in agent ID: '{agent_id}'")

    return models


def load_model_release_dates(json_path: Path) -> Dict[str, str]:
    """Load model release dates from JSON file.

    Args:
        json_path: Path to terminalbench_model_release_dates.json

    Returns:
        Dict mapping model_name -> date string (YYYYMMDD)

    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Model release dates file not found: {json_path}. "
            "This file contains release dates for TerminalBench models."
        )

    with open(json_path, "r") as f:
        return json.load(f)


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

    TerminalBench agent dates are derived from the underlying model's release date,
    loaded from data/terminalbench/model_release_dates.json. For multi-model agents,
    the latest (max) release date is used since the agent can't exist before all
    its constituent models are released.
    """

    # Data paths
    responses_path: Path = field(
        default_factory=lambda: Path("data/terminalbench/responses.jsonl")
    )
    oracle_irt_path: Path = field(
        default_factory=lambda: Path("data/terminalbench/irt/1d_1pl/items.csv")
    )
    oracle_abilities_path: Path = field(
        default_factory=lambda: Path("data/terminalbench/irt/1d_1pl/abilities.csv")
    )
    metadata_path: Path = field(
        default_factory=lambda: Path("data/terminalbench/meta.json")
    )

    # Model release dates file (maps model names to release dates)
    model_release_dates_path: Path = field(
        default_factory=lambda: Path("data/terminalbench/model_release_dates.json")
    )

    # No pre-computed baseline IRT for TerminalBench (will use oracle or skip)
    baseline_irt_path: Optional[Path] = None

    embeddings_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "embeddings/"
            "embeddings__deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__pool-lasttoken__qs-sol-instr__"
            "__solcap_nocapqs_sol_no_tests_instr_nocap_b7008f2d__idnorm_instance-v2__"
            "json_terminal_bench_tasks.jsonl__test__maxlen8192.npz"
        )
    )
    llm_judge_path: Optional[Path] = field(
        default_factory=lambda: Path(
            "llm_judge_features/defaults/terminalbench/llm_judge_features.csv"
        )
    )

    # Frontier split settings
    # Cutoff 2025-09-01 splits:
    #   Pre-frontier: 37 agents (models released before Sept 2025)
    #   Post-frontier: 46 agents (models from Sept 2025 onwards)
    #   Frontier tasks (zero_pre): 11 tasks
    # Earlier cutoff gives more frontier tasks for reliable Mean Per-Agent AUC
    # with lower SEM (~0.04 vs ~0.2 with later cutoffs)
    cutoff_date: str = "20250901"

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path("output/experiment_b/terminalbench")
    )

    # Cache for loaded model release dates
    _model_release_dates_cache: Optional[Dict[str, str]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        return "TerminalBench"

    def _load_model_release_dates(self) -> Dict[str, str]:
        """Load model release dates from JSON file (cached)."""
        if self._model_release_dates_cache is None:
            object.__setattr__(
                self,
                "_model_release_dates_cache",
                load_model_release_dates(self.model_release_dates_path),
            )
        return self._model_release_dates_cache

    def get_agent_dates(self, agents: List[str]) -> Dict[str, str]:
        """Get dates from model release dates.

        For each agent, extract the underlying model(s) and look up their release
        dates. For multi-model agents, use the latest (max) release date since
        the agent can't exist before all its models are released.

        Args:
            agents: List of agent IDs from response matrix

        Returns:
            Dict mapping agent_id -> date string (YYYYMMDD)

        Raises:
            ValueError: If any model is missing from the release dates JSON
        """
        release_dates = self._load_model_release_dates()
        agent_dates = {}

        for agent in agents:
            models = extract_models_from_agent_id(agent)

            # Look up release date for each model
            model_dates = []
            missing_models = []
            for model in models:
                if model in release_dates:
                    model_dates.append(release_dates[model])
                else:
                    missing_models.append(model)

            if missing_models:
                raise ValueError(
                    f"Models {missing_models} from agent '{agent}' not found in "
                    f"{self.model_release_dates_path}. Please add the release dates."
                )

            # Use the latest (max) date for multi-model agents
            agent_dates[agent] = max(model_dates)

        return agent_dates

    @property
    def llm_judge_feature_cols(self) -> List[str]:
        """TerminalBench LLM judge feature columns (unified 15 features)."""
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
