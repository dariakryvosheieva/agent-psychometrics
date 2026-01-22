"""Model release date lookup for agent date correction.

This module provides a mapping from model identifiers to their actual release dates,
allowing experiments to use model release dates instead of SWE-bench submission dates.

The submission date (extracted from agent name prefix) often lags the actual model
release date by days to months. Using release dates provides a more accurate
representation of when model capability became available.

For agents where the underlying model cannot be identified, the submission date
is used as a fallback.
"""

import re
from typing import Optional

# Model release dates in YYYYMMDD format
# Sources documented in /Users/chrisge/.claude/plans/humming-spinning-wave.md
MODEL_RELEASE_DATES = {
    # Legacy models (2022-2023)
    "gpt35": "20221130",
    "gpt-3.5": "20221130",
    "gpt-35": "20221130",
    "claude2": "20230711",
    "claude-2": "20230711",
    "gpt4": "20230314",
    "gpt-4": "20230314",
    "swellama7b": "20231206",
    "swellama13b": "20231206",
    "swe-llama-7b": "20231206",
    "swe-llama-13b": "20231206",
    # 2024 releases
    "claude3opus": "20240304",
    "claude-3-opus": "20240304",
    "claude3-opus": "20240304",
    "llama3_70b": "20240418",
    "llama-3-70b": "20240418",
    "llama3-70b": "20240418",
    "gpt4o": "20240513",
    "gpt-4o": "20240513",
    "gpt4-o": "20240513",
    # Claude 3.5 Sonnet v1 (June 2024)
    "claude3.5sonnet": "20240620",
    "claude-3.5-sonnet": "20240620",
    "claude-3-5-sonnet-20240620": "20240620",
    "claude35sonnet": "20240620",
    # o1 (September 2024)
    "o1": "20240912",
    "o1-preview": "20240912",
    "o1-mini": "20240912",
    # Qwen 2.5 (September 2024)
    "qwen2.5-72b": "20240919",
    "qwen-2.5-72b": "20240919",
    "qwen2.5-7b": "20240919",
    # Claude 3.5 Sonnet v2/new (October 2024)
    "claude-3-5-sonnet-20241022": "20241022",
    "claude-3-5-sonnet-updated": "20241022",
    "sonnet-20241022": "20241022",
    # Claude 3.5 Haiku (November 2024)
    "claude-3-5-haiku": "20241104",
    "claude3.5haiku": "20241104",
    "claude35haiku": "20241104",
    # Qwen 2.5 Coder (November 2024)
    "qwen2.5-coder-32b": "20241112",
    "qwen-2.5-coder": "20241112",
    # Gemini 2.0 Flash (December 2024)
    "gemini_2.0_flash": "20241211",
    "gemini-2-0-flash": "20241211",
    "gemini-2.0-flash": "20241211",
    "gemini2.0flash": "20241211",
    # DeepSeek V3 (December 2024)
    "deepseek-v3": "20241226",
    "deepseek_v3": "20241226",
    "deepseekvr": "20241226",
    # 2025 releases
    # o3-mini (January 2025)
    "o3-mini": "20250131",
    "o3_mini": "20250131",
    "o3mini": "20250131",
    # Claude 3.7 Sonnet (February 2025)
    "claude-3-7-sonnet": "20250224",
    "claude-3.7-sonnet": "20250224",
    "claude37": "20250224",
    "claude3.7": "20250224",
    # o3 and o4-mini (April 2025)
    "o3": "20250416",
    "o4-mini": "20250416",
    "o4_mini": "20250416",
    "o4mini": "20250416",
    # Qwen3 (April 2025)
    "qwen3": "20250428",
    "qwen-3": "20250428",
    # Amazon Nova Premier (April 2025)
    "nova-premier": "20250430",
    "nova_premier": "20250430",
    # Devstral (May 2025)
    "devstral": "20250521",
    "devstral_small": "20250521",
    # Claude 4 Sonnet and Opus (May 2025)
    "claude-4-sonnet": "20250522",
    "claude_4_sonnet": "20250522",
    "claude4sonnet": "20250522",
    "claude-4-sonnet-20250514": "20250522",
    "claude-4-opus": "20250522",
    "claude_4_opus": "20250522",
    "claude4opus": "20250522",
    # Skywork-SWE 32B (June 2025)
    "skywork-swe-32b": "20250620",
    "skywork_swe_32b": "20250620",
    # Kimi K2 (July 2025)
    "kimi_k2": "20250716",
    "kimi-k2": "20250716",
    "kimi_k2_instruct": "20250716",
    # Qwen3-Coder (July 2025)
    "qwen3-coder": "20250722",
    "qwen3_coder": "20250722",
    "qwen3-coder-480b": "20250722",
    "qwen3-coder-30b": "20250722",
    # GLM-4.5 (July 2025)
    "glm4-5": "20250728",
    "glm-4.5": "20250728",
    "glm-4-5": "20250728",
    "glm45": "20250728",
    # GPT-5 (August 2025)
    "gpt5": "20250807",
    "gpt-5": "20250807",
    # Claude Sonnet 4.5 (September 2025)
    "claude-sonnet-4-5": "20250929",
    "claude-sonnet-4.5": "20250929",
    "claude_sonnet_4_5": "20250929",
    # GLM-4.6 (September 2025)
    "glm4-6": "20250930",
    "glm-4.6": "20250930",
    "glm-4-6": "20250930",
    "glm46": "20250930",
    # Gemini 3 Pro (November 2025)
    "gemini-3-pro": "20251118",
    "gemini_3_pro": "20251118",
    "gemini3pro": "20251118",
    # Claude Opus 4.5 (November 2025)
    "claude-opus-4-5": "20251124",
    "claude-opus-4.5": "20251124",
    # Doubao Seed Code (December 2025)
    "doubao": "20251203",
    "doubao_seed_code": "20251203",
    "doubao-seed-code": "20251203",
}


def _extract_submission_date(agent_name: str) -> Optional[str]:
    """Extract YYYYMMDD submission date prefix from agent name.

    Args:
        agent_name: Agent identifier (e.g., "20240620_sweagent_claude3.5sonnet")

    Returns:
        Date string in YYYYMMDD format, or None if no valid prefix found
    """
    match = re.match(r"^(\d{8})_", agent_name)
    if match:
        return match.group(1)
    return None


def _identify_model_from_agent_name(agent_name: str) -> Optional[str]:
    """Identify the underlying model from an agent name.

    This function uses pattern matching to identify the LLM used by an agent.
    The patterns are ordered from most specific to least specific to avoid
    false matches (e.g., "claude3.5" should match before "claude3").

    Args:
        agent_name: Agent identifier string (lowercase for matching)

    Returns:
        A key from MODEL_RELEASE_DATES if a model is identified, else None
    """
    name_lower = agent_name.lower()

    # Ordered patterns from most specific to least specific
    patterns = [
        # Claude Haiku 4.5 (TerminalBench format)
        (r"claude[-_]?haiku[-_]?4[-_.]?5", "claude-sonnet-4-5"),  # Map to release date
        # Claude 4.5 variants
        (r"claude-sonnet-4-5|claude[-_]sonnet[-_]4[-_.]5", "claude-sonnet-4-5"),
        (r"claude-opus-4-5|claude[-_]opus[-_]4[-_.]5", "claude-opus-4-5"),
        # Claude 4 variants
        (r"claude[-_]?4[-_]?opus", "claude-4-opus"),
        (r"claude[-_]?4[-_]?sonnet|claude-4-sonnet-20250514", "claude-4-sonnet"),
        # Claude 3.7
        (r"claude[-_]?3[-_.]?7[-_]?sonnet|claude37", "claude-3-7-sonnet"),
        # Claude 3.5 Sonnet (check for updated version first)
        (r"claude-3-5-sonnet-20241022|sonnet-20241022|claude-3-5-sonnet-updated", "claude-3-5-sonnet-20241022"),
        (r"claude[-_]?3[-_.]?5[-_]?sonnet|claude3\.5sonnet|claude35sonnet", "claude3.5sonnet"),
        # Claude 3.5 Haiku
        (r"claude[-_]?3[-_.]?5[-_]?haiku", "claude-3-5-haiku"),
        # Claude 3 Opus
        (r"claude[-_]?3[-_]?opus|claude3opus", "claude3opus"),
        # Claude 2
        (r"claude[-_]?2\b", "claude2"),
        # GPT-5 (also match gpt-5 in TerminalBench format)
        (r"gpt[-_]?5(?:\b|_at_)", "gpt5"),
        # GPT-4o
        (r"gpt[-_]?4[-_]?o\b", "gpt4o"),
        # GPT-4 (must come after gpt4o)
        (r"gpt[-_]?4\b", "gpt4"),
        # GPT-3.5
        (r"gpt[-_]?3[-_.]?5|gpt35", "gpt35"),
        # OpenAI o-series (more flexible patterns)
        (r"o4[-_]?mini", "o4-mini"),
        (r"o3[-_]?mini", "o3-mini"),
        (r"_o3\b|^o3\b", "o3"),  # Match _o3 at end or o3 at start
        (r"_o1|o1[-_]|^o1\b", "o1"),  # Match _o1 or o1_ or o1 at start
        # Gemini
        (r"gemini[-_]?3[-_]?pro", "gemini-3-pro"),
        (r"gemini[-_]?2[-_.]?0[-_]?flash", "gemini_2.0_flash"),
        # Chinese models
        (r"kimi[-_]?k2", "kimi_k2"),
        (r"deepseek[-_]?v3", "deepseek-v3"),
        (r"glm[-_]?4[-_.]?6", "glm4-6"),
        (r"glm[-_]?4[-_.]?5", "glm4-5"),
        (r"doubao", "doubao"),
        # Qwen variants (most specific first)
        (r"qwen3[-_]?coder[-_]?480b", "qwen3-coder-480b"),
        (r"qwen3[-_]?coder[-_]?30b", "qwen3-coder-30b"),
        (r"qwen3[-_]?coder", "qwen3-coder"),
        (r"qwencoder[-_]?30b", "qwen3-coder-30b"),
        (r"qwen2\.5[-_]?coder", "qwen2.5-coder-32b"),
        (r"qwen2\.5[-_]?72b", "qwen2.5-72b"),
        (r"qwen2\.5[-_]?7b", "qwen2.5-7b"),
        # Other models
        (r"devstral", "devstral"),
        (r"nova[-_]?premier", "nova-premier"),
        (r"llama[-_]?3[-_]?70b", "llama3_70b"),
        (r"skywork[-_]?swe[-_]?32b", "skywork-swe-32b"),
        (r"swellama[-_]?13b", "swellama13b"),
        (r"swellama[-_]?7b", "swellama7b"),
        # Lingma uses custom Qwen-based models, but we can't determine exact release date
        # so these will fall back to submission date
    ]

    for pattern, model_key in patterns:
        if re.search(pattern, name_lower):
            return model_key

    return None


def get_model_release_date(agent_name: str) -> str:
    """Get the model release date for an agent.

    This function attempts to identify the underlying LLM from the agent name
    and return its release date. If the model cannot be identified, it falls
    back to the submission date extracted from the agent name prefix.

    Args:
        agent_name: Agent identifier (e.g., "20240620_sweagent_claude3.5sonnet")

    Returns:
        Date string in YYYYMMDD format

    Raises:
        ValueError: If neither model release date nor submission date can be determined
    """
    # Try to identify the model and get its release date
    model_key = _identify_model_from_agent_name(agent_name)
    if model_key and model_key in MODEL_RELEASE_DATES:
        return MODEL_RELEASE_DATES[model_key]

    # Fall back to submission date
    submission_date = _extract_submission_date(agent_name)
    if submission_date:
        return submission_date

    raise ValueError(
        f"Cannot determine date for agent '{agent_name}': "
        "no model identified and no submission date prefix found"
    )


def get_agent_dates_with_release_dates(agents: list[str]) -> dict[str, str]:
    """Get dates for a list of agents using model release dates where possible.

    Args:
        agents: List of agent identifiers

    Returns:
        Dict mapping agent_id -> date string (YYYYMMDD)

    Raises:
        ValueError: If any agent's date cannot be determined
    """
    return {agent: get_model_release_date(agent) for agent in agents}
