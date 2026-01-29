"""Keyword feature definitions for trajectory analysis.

These features are extracted from the LLM's reasoning text about agent trajectories.
Each keyword pattern captures a specific behavioral signal that may correlate with
task difficulty.

To add a new keyword feature:
1. Add the pattern to KEYWORD_FEATURES dict below
2. Run the analysis script to check if it's significant
3. If significant (p < 0.05), it will be automatically included in the model
"""

import re
from typing import Dict

# Keyword features extracted from LLM reasoning text
# Key: feature name suffix (will be prefixed with agent name, e.g., "masai_kw_stuck_loop")
# Value: regex pattern to search for in reasoning text (case-insensitive)
KEYWORD_FEATURES: Dict[str, str] = {
    # Struggle indicators
    "kw_stuck_loop": r"stuck|loop|same error|repeating|circular|trying.*again",
    "kw_gave_up": r"gave up|cannot|unable|impossible|no way|failed to",
    "kw_multiple_failed": r"multiple attempts|tried several|many attempts|kept failing",

    # Progress indicators
    "kw_verified": r"verified|confirmed|passes|correct|working",
    "kw_incomplete": r"incomplete|partial|not finished|didn't finish",

    # Environment/infrastructure issues
    "kw_infrastructure": r"environment|setup|install|dependency|config|permission",
    "kw_env_issues": r"environment|setup|dependency|permission|timeout|memory",
    "kw_test_infra": r"test infrastructure|test framework|test setup|setting up test|standalone test",

    # Understanding issues
    "kw_misunderstood": r"misunderstood|wrong approach|incorrect assumption|realized",
    "kw_complex": r"complex|complicated|intricate|difficult",

    # New features from high-residual analysis (2026-01-28)
    "kw_repetitive": r"repetitive|repeat|same.*call|same.*search|without.*progress",
    "kw_backtrack": r"backtrack|revert|constantly revert|same.*error|same.*approach|same.*fail",
    "kw_truncated": r"truncated|incomplete|cut off|partial.*patch",
    "kw_quick_fix": r"quickly identified|quickly.*fix|quick.*diagnos|early.*identif",

    # Second batch from high-residual analysis (2026-01-28)
    "kw_wrong_location": r"wrong file|wrong location|mislocalized|wrong function|incorrect file|wrong.*file",
    "kw_fundamental": r"fundamentally flawed|fundamentally incorrect|fundamental misunderstanding|fundamentally misidentified",
    "kw_never_found": r"never found|never arrived|never discovered|never identified|never recognized",
    "kw_syntax_error": r"syntax error|invalid syntax|indentation error|truncation error|sanitization",
    "kw_constant_pivot": r"constantly pivoted|constant backtracking|keeps trying|kept trying|cycling through",

    # Third batch - task-specific indicators (2026-01-28)
    "kw_efficient": r"exceptional efficiency|efficient|straightforward|quickly solved|immediately identified",
    "kw_extension_code": r"c\+\+|extension|cython|native|cpp",
    "kw_edge_cases": r"edge case|edge-case|boundary|corner case",
    "kw_correct_early": r"correct.*from.*start|correctly.*from.*start|correctly identified.*early|understood.*from.*start",

    # Fourth batch - solution quality indicators (2026-01-28)
    "kw_speculative": r"speculative|guessing|uncertain|may not|unlikely to work",
    "kw_contradicted": r"contradicted|different.*answers|inconsistent.*results|three different",
    "kw_oversimplified": r"oversimplified|too simple|missed.*root cause|didn't.*address",
    "kw_deep_issue": r"precedence|expression.*tree|deeper.*issue|root cause|underlying",

    # Fifth batch - implementation quality (2026-01-28)
    "kw_no_implement": r"never.*implement|no.*fix|without.*implement|didn't.*implement|no.*solution.*attempt",
    "kw_logic_error": r"flawed logic|incorrect logic|wrong logic|logic.*error|double.*normaliz",
    "kw_debugging_only": r"spent.*debugging|investigation.*without|exploring.*without|debugging.*without",
    "kw_correct_location": r"correctly identified.*location|correctly identified.*file|correct.*file.*method",
}


def extract_keyword_features(reasoning_text: str, agent_prefix: str) -> Dict[str, int]:
    """Extract keyword features from reasoning text.

    Args:
        reasoning_text: The LLM's reasoning about the trajectory
        agent_prefix: Prefix for feature names (e.g., "masai")

    Returns:
        Dict mapping feature names to binary values (0 or 1)
    """
    if not reasoning_text or not isinstance(reasoning_text, str):
        return {f"{agent_prefix}_{name}": 0 for name in KEYWORD_FEATURES}

    text_lower = reasoning_text.lower()
    features = {}

    for name, pattern in KEYWORD_FEATURES.items():
        match = bool(re.search(pattern, text_lower))
        features[f"{agent_prefix}_{name}"] = int(match)

    return features
