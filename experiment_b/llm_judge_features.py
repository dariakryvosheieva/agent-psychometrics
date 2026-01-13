"""Extract trajectory features using direct LLM API calls (no Lunette sandbox).

This module provides semantic features extracted from agent trajectories using
direct LLM API calls (Anthropic/OpenAI). It mirrors the interface of
lunette_features.py for consistent evaluation in Experiment B.

Features are the same 14 dimensions as Lunette for direct comparison:
- Primary output: llm_judge_difficulty_score (0-1)
- Agentic competencies (1-4): backtracking_exploration, task_decomposition,
  observation_reading, self_verification
- Failure modes (0-1): localization_failure, strategy_defect, implementation_defect,
  incomplete_repair, verification_failure
- Trajectory signals (0-1): agent_looping, agent_gave_up_early, agent_wrong_focus,
  context_overflow
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Feature names - same structure as lunette_features.py
LLM_JUDGE_FEATURE_NAMES = [
    # Primary output: LLM judge's difficulty prediction (0=easy, 1=hard)
    "llm_judge_difficulty_score",
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


# Grading prompt - includes both trajectory AND SWE-bench metadata
TRAJECTORY_GRADING_PROMPT = """You are analyzing a SWE-bench agent trajectory to PREDICT TASK DIFFICULTY.

## TASK METADATA (from SWE-bench Verified)

**Task ID:** {instance_id}
**Repository:** {repo}
**Version:** {version}

**Problem Statement:**
{problem_statement}

**Gold Patch (correct solution):**
```diff
{patch}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{fail_to_pass}

{hints_section}

## AGENT TRAJECTORY

The following shows the agent's attempt at solving this task:

{trajectory_text}

**Outcome:** {resolved_status}

## YOUR TASK: Predict Difficulty (0.0 to 1.0)

Based on how the agent struggled (or didn't), estimate how difficult this task is.

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

## In your response, provide:

1. **Competencies** (1-4 scale each):
   - backtracking_exploration: How well did the agent backtrack and explore alternatives?
   - task_decomposition: How well did the agent break down the problem?
   - observation_reading: How well did the agent read and understand outputs?
   - self_verification: How well did the agent verify its own work?

2. **Failure modes detected** (0 or 1 each):
   - localization_failure: Failed to find the right code location
   - strategy_defect: Used wrong approach/strategy
   - implementation_defect: Implementation bugs in the fix
   - incomplete_repair: Fix was partial/incomplete
   - verification_failure: Failed to verify the fix worked

3. **Trajectory signals** (0 or 1 each):
   - agent_looping: Got stuck in repetitive loops
   - agent_gave_up_early: Stopped before fully exploring
   - agent_wrong_focus: Fixated on irrelevant code/issues
   - context_overflow: Lost track of earlier findings

Respond with ONLY a JSON object in this exact format:
{{
    "llm_judge_difficulty_score": <0.0-1.0>,
    "backtracking_exploration": <1-4>,
    "task_decomposition": <1-4>,
    "observation_reading": <1-4>,
    "self_verification": <1-4>,
    "localization_failure": <0 or 1>,
    "strategy_defect": <0 or 1>,
    "implementation_defect": <0 or 1>,
    "incomplete_repair": <0 or 1>,
    "verification_failure": <0 or 1>,
    "agent_looping": <0 or 1>,
    "agent_gave_up_early": <0 or 1>,
    "agent_wrong_focus": <0 or 1>,
    "context_overflow": <0 or 1>,
    "reasoning": "<brief explanation of your difficulty assessment, 2-3 sentences>"
}}
"""


@dataclass
class LLMJudgeFeatures:
    """Features extracted from a trajectory using direct LLM API."""

    # Primary output: LLM judge's difficulty prediction (0=easy, 1=hard)
    llm_judge_difficulty_score: float

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
            self.llm_judge_difficulty_score,
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
    def from_dict(cls, d: Dict) -> "LLMJudgeFeatures":
        """Create from dictionary (e.g., parsed JSON)."""
        return cls(
            llm_judge_difficulty_score=float(d.get("llm_judge_difficulty_score", 0.5)),
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
    def default(cls) -> "LLMJudgeFeatures":
        """Return default/neutral features."""
        return cls(
            llm_judge_difficulty_score=0.5,
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


def load_llm_judge_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeFeatures]:
    """Load pre-computed LLM judge features for a task-agent pair.

    Args:
        task_id: Task instance ID
        agent: Agent name
        features_dir: Base directory for features

    Returns:
        LLMJudgeFeatures or None if not found
    """
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeFeatures.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeFeatures]:
    """Load LLM judge features for a task across multiple agents.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        features_dir: Base directory for features

    Returns:
        Dict mapping agent -> LLMJudgeFeatures
    """
    result = {}
    for agent in agents:
        features = load_llm_judge_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_features(features: Dict[str, LLMJudgeFeatures]) -> np.ndarray:
    """Aggregate LLM judge features across multiple trajectories.

    Returns averaged feature vector across all agents.
    """
    if not features:
        return np.zeros(len(LLM_JUDGE_FEATURE_NAMES))

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)


def load_and_aggregate_llm_judge_features(
    task_ids: List[str],
    agents: List[str],
    features_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load and aggregate LLM judge features for multiple tasks.

    Args:
        task_ids: List of task IDs
        agents: List of agent names whose features to use
        features_dir: Base directory for features

    Returns:
        Dict mapping task_id -> aggregated feature vector
    """
    result = {}
    for task_id in task_ids:
        features = load_llm_judge_features_for_task(task_id, agents, features_dir)
        if features:
            result[task_id] = aggregate_llm_judge_features(features)
    return result
