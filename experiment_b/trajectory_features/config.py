"""Configuration for trajectory feature extraction.

Selected agents span IRT ability from -1.26 to +2.24, all with trajectories < 120K tokens.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class AgentInfo:
    """Information about a selected agent."""
    name: str
    theta: float  # IRT ability
    avg_tokens: int  # Average trajectory tokens
    max_tokens: int  # Maximum trajectory tokens


# Selected 6 pre-frontier agents (date < 20250501) for Experiment B
# Selection criteria:
# 1. Date before cutoff (20250501) to avoid contamination
# 2. 100% task coverage (500/500 tasks in unified_trajs)
# 3. Optimal ability spread from theta=-1.60 to theta=+1.65
# 4. Even gaps (0.53-0.72 points) across ability spectrum
SELECTED_AGENTS: List[AgentInfo] = [
    AgentInfo("20250415_openhands", 1.65, 27419, 77677),
    AgentInfo("20250410_cortexa", 1.03, 3171, 3763),
    AgentInfo("20241029_OpenHands-CodeAct-2.1-sonnet-20241022", 0.50, 26151, 63186),
    AgentInfo("20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022", -0.22, 17947, 81182),
    AgentInfo("20240721_amazon-q-developer-agent-20240719-dev", -0.91, 1946, 5757),
    AgentInfo("20241002_lingma-agent_lingma-swe-gpt-72b", -1.60, 10813, 25451),
]

AGENT_NAMES = [a.name for a in SELECTED_AGENTS]


@dataclass
class FeatureSpec:
    """Specification for a trajectory feature."""
    name: str
    description: str
    scale: str  # "0-5", "bool", "count"
    expected_direction: str  # "positive" (higher = harder), "negative", "unknown"


# Features to extract from trajectories
TRAJECTORY_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        "loop_detection",
        "Did the model repeat similar actions/mistakes?",
        "0-5",
        "positive"  # More loops = harder task
    ),
    FeatureSpec(
        "localization_quality",
        "Did the model correctly identify the problem location?",
        "0-5",
        "negative"  # Better localization = easier task
    ),
    FeatureSpec(
        "debugging_cycles",
        "Number of debug-fix cycles (attempts to fix after errors)",
        "count",
        "positive"  # More cycles = harder task
    ),
    FeatureSpec(
        "error_recovery",
        "Did the model successfully recover from errors?",
        "0-5",
        "negative"  # Better recovery = easier task
    ),
    FeatureSpec(
        "exploration_breadth",
        "How many files/approaches did the model explore?",
        "count",
        "positive"  # More exploration = harder task
    ),
    FeatureSpec(
        "focus_drift",
        "Did the model stay on task or get distracted?",
        "0-5",
        "positive"  # More drift = harder task
    ),
    FeatureSpec(
        "solution_completeness",
        "How complete was the attempted solution?",
        "0-5",
        "negative"  # More complete = easier task
    ),
    FeatureSpec(
        "edge_case_handling",
        "Did the agent consider and handle edge cases?",
        "0-5",
        "negative"  # Better edge case handling = easier task
    ),
    FeatureSpec(
        "test_verification",
        "Did the agent verify their solution works?",
        "0-5",
        "negative"  # Better verification = easier task
    ),
]

FEATURE_NAMES = [f.name for f in TRAJECTORY_FEATURES]


# Model configuration
DEFAULT_MODEL_EXPLORATION = "claude-sonnet-4-5-20250929"
DEFAULT_MODEL_FINAL = "claude-opus-4-5-20251101"
