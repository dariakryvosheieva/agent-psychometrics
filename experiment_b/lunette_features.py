"""Extract trajectory features using Lunette as LLM judge.

This module provides rich semantic features extracted from agent trajectories,
combining insights from:
- AgentDiagnose (EMNLP 2025): Agentic competency ratings
- SWE-bench Failures paper: Failure mode taxonomy
- SWE-bench Pro: Additional trajectory signals
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Feature names for the Lunette-extracted feature vector
# Main output is lunette_difficulty_score (0-1), plus parsed features from explanation
LUNETTE_FEATURE_NAMES = [
    # Primary output: Lunette's difficulty prediction (0=easy, 1=hard)
    "lunette_difficulty_score",
    # Agentic competencies (1-4 scale, from AgentDiagnose)
    "backtracking_exploration",
    "task_decomposition",
    "observation_reading",
    "self_verification",
    # Failure mode indicators (0-1 binary, from SWE-bench Failures)
    "localization_failure",
    "strategy_defect",
    "implementation_defect",
    "incomplete_repair",
    "verification_failure",
    # Trajectory signals (0-1 binary)
    "agent_looping",
    "agent_gave_up_early",
    "agent_wrong_focus",
    "context_overflow",
]

# Grading prompt for Lunette - outputs difficulty as score (0-1)
TRAJECTORY_GRADING_PROMPT = """You are analyzing a SWE-bench agent trajectory to PREDICT TASK DIFFICULTY.

The trajectory shows an agent attempting to solve a software engineering task. Based on how the agent struggled (or didn't), estimate how difficult this task is.

## YOUR TASK: Predict Difficulty (0.0 to 1.0)

Output a **difficulty score** between 0.0 and 1.0:
- 0.0 = Very easy task (agent solved it quickly with no issues)
- 0.3 = Easy task (minor struggles but clear path to solution)
- 0.5 = Medium difficulty (some exploration needed, partial failures)
- 0.7 = Hard task (significant struggles, multiple failure modes)
- 1.0 = Very hard task (agent failed completely despite substantial effort)

## Difficulty Indicators to Consider

### Positive (harder task) indicators:
- Agent tried multiple approaches that failed
- Agent got stuck in loops or repeated similar mistakes
- Agent mislocated the relevant code
- Agent's fix was superficial or incomplete
- Agent lost track of context
- Many back-and-forth iterations without progress

### Negative (easier task) indicators:
- Agent found the right location quickly
- Agent's first approach worked
- Clear error messages guided the agent
- Simple, localized fix was sufficient
- Agent verified solution correctly

## In your explanation, briefly note:

1. **Competencies** (1-4 scale each): backtracking_exploration, task_decomposition, observation_reading, self_verification

2. **Failure modes detected** (if any): localization_failure, strategy_defect, implementation_defect, incomplete_repair, verification_failure

3. **Trajectory signals** (if any): agent_looping, agent_gave_up_early, agent_wrong_focus, context_overflow

## REQUIRED: Structure your explanation as follows:

COMPETENCIES: backtracking=X/4, decomposition=X/4, observation=X/4, verification=X/4
FAILURES: [list any that apply: localization_failure, strategy_defect, implementation_defect, incomplete_repair, verification_failure]
SIGNALS: [list any that apply: agent_looping, agent_gave_up_early, agent_wrong_focus, context_overflow]
REASONING: [1-2 sentences explaining why this difficulty score]

Example for a hard task:
"COMPETENCIES: backtracking=2/4, decomposition=2/4, observation=2/4, verification=1/4
FAILURES: localization_failure, strategy_defect
SIGNALS: agent_looping, context_overflow
REASONING: Agent struggled to find the correct file, tried multiple superficial fixes, and lost track of earlier findings."
"""


@dataclass
class LunetteFeatures:
    """Features extracted from a trajectory using Lunette."""

    # Primary output: Lunette's difficulty prediction (0=easy, 1=hard)
    lunette_difficulty_score: float

    # Agentic competencies (1-4 scale)
    backtracking_exploration: float
    task_decomposition: float
    observation_reading: float
    self_verification: float

    # Failure mode indicators (0-1)
    localization_failure: float
    strategy_defect: float
    implementation_defect: float
    incomplete_repair: float
    verification_failure: float

    # Trajectory signals (0-1)
    agent_looping: float
    agent_gave_up_early: float
    agent_wrong_focus: float
    context_overflow: float

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector in standard order."""
        return np.array([
            self.lunette_difficulty_score,
            self.backtracking_exploration,
            self.task_decomposition,
            self.observation_reading,
            self.self_verification,
            self.localization_failure,
            self.strategy_defect,
            self.implementation_defect,
            self.incomplete_repair,
            self.verification_failure,
            self.agent_looping,
            self.agent_gave_up_early,
            self.agent_wrong_focus,
            self.context_overflow,
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LunetteFeatures":
        """Create from dictionary (e.g., parsed JSON)."""
        return cls(
            lunette_difficulty_score=float(d.get("lunette_difficulty_score", 0.5)),
            backtracking_exploration=float(d.get("backtracking_exploration", 2.5)),
            task_decomposition=float(d.get("task_decomposition", 2.5)),
            observation_reading=float(d.get("observation_reading", 2.5)),
            self_verification=float(d.get("self_verification", 2.5)),
            localization_failure=float(d.get("localization_failure", 0)),
            strategy_defect=float(d.get("strategy_defect", 0)),
            implementation_defect=float(d.get("implementation_defect", 0)),
            incomplete_repair=float(d.get("incomplete_repair", 0)),
            verification_failure=float(d.get("verification_failure", 0)),
            agent_looping=float(d.get("agent_looping", 0)),
            agent_gave_up_early=float(d.get("agent_gave_up_early", 0)),
            agent_wrong_focus=float(d.get("agent_wrong_focus", 0)),
            context_overflow=float(d.get("context_overflow", 0)),
        )

    @classmethod
    def default(cls) -> "LunetteFeatures":
        """Return default/neutral features."""
        return cls(
            lunette_difficulty_score=0.5,
            backtracking_exploration=2.5,
            task_decomposition=2.5,
            observation_reading=2.5,
            self_verification=2.5,
            localization_failure=0.0,
            strategy_defect=0.0,
            implementation_defect=0.0,
            incomplete_repair=0.0,
            verification_failure=0.0,
            agent_looping=0.0,
            agent_gave_up_early=0.0,
            agent_wrong_focus=0.0,
            context_overflow=0.0,
        )


def load_lunette_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LunetteFeatures]:
    """Load pre-computed Lunette features for a task-agent pair.

    Args:
        task_id: Task instance ID
        agent: Agent name
        features_dir: Base directory for features

    Returns:
        LunetteFeatures or None if not found
    """
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LunetteFeatures.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_lunette_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LunetteFeatures]:
    """Load Lunette features for a task across multiple agents.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        features_dir: Base directory for features

    Returns:
        Dict mapping agent -> LunetteFeatures
    """
    result = {}
    for agent in agents:
        features = load_lunette_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_lunette_features(features: Dict[str, LunetteFeatures]) -> np.ndarray:
    """Aggregate Lunette features across multiple trajectories.

    Returns averaged feature vector across all agents.
    """
    if not features:
        return np.zeros(len(LUNETTE_FEATURE_NAMES))

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)


def load_and_aggregate_lunette_features(
    task_ids: List[str],
    agents: List[str],
    features_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load and aggregate Lunette features for multiple tasks.

    Args:
        task_ids: List of task IDs
        agents: List of agent names whose features to use
        features_dir: Base directory for features

    Returns:
        Dict mapping task_id -> aggregated feature vector
    """
    result = {}
    for task_id in task_ids:
        features = load_lunette_features_for_task(task_id, agents, features_dir)
        if features:
            result[task_id] = aggregate_lunette_features(features)
    return result
