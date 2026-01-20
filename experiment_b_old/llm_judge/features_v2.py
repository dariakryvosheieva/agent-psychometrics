"""Improved LLM judge features focusing on residual prediction.

This module provides an alternative prompt strategy that focuses on predicting
what the heuristic prior (problem_len, patch_len, repo) gets wrong.

Key insight from residual analysis:
- High positive residuals (harder than expected): Subtle API interactions, edge cases
- High negative residuals (easier than expected): Large patches with simple conceptual fixes

The prompt asks the LLM to identify these patterns from the trajectory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Feature names for V2 - focused on residual prediction
LLM_JUDGE_V2_FEATURE_NAMES = [
    # Primary outputs: residual predictors
    "conceptual_vs_mechanical_complexity",  # +high = conceptually simple but mechanically large
    "hidden_trap_indicator",                 # +high = subtle edge cases/API interactions
    "solution_directness",                   # +high = clear path to solution, low = exploration needed

    # Task characteristics (from problem + trajectory)
    "api_edge_case",                # 0-1: involves subtle API interactions
    "multi_component_interaction",  # 0-1: requires understanding multiple components
    "error_message_helpfulness",    # 1-4: how helpful were error messages
    "fix_is_pattern_change",        # 0-1: fix follows known pattern vs novel approach

    # Trajectory patterns
    "exploration_breadth",          # 1-4: how widely did agent search
    "exploration_efficiency",       # 1-4: how quickly did agent find relevant code
    "agent_confidence_mismatch",    # 0-1: agent was confident but wrong
    "repetitive_failed_attempts",   # 0-1: agent tried same approach multiple times

    # Standard features (for comparison)
    "llm_judge_difficulty_score",   # 0-1: overall difficulty assessment
    "resolved_correctly",           # 0-1: did agent actually solve it
    "trajectory_length_normalized", # normalized by typical length for repo
]


RESIDUAL_FOCUSED_PROMPT = """You are analyzing a SWE-bench agent trajectory to identify what makes this task's difficulty SURPRISING.

## CONTEXT: The Prior Model

A simple prior model predicts task difficulty using only:
- Problem statement length
- Gold patch length
- Repository name

This prior is often WRONG because it misses:
1. **Conceptually simple but mechanically large fixes** - The patch is big but the idea is simple
2. **Conceptually hard but small fixes** - The patch is tiny but requires deep understanding
3. **Hidden traps** - Subtle API edge cases, race conditions, inheritance issues

## TASK METADATA

**Task ID:** {instance_id}
**Repository:** {repo}
**Version:** {version}

**Problem Statement ({problem_len} chars):**
{problem_statement}

**Gold Patch ({patch_len} chars):**
```diff
{patch}
```

**Tests that must pass:** {fail_to_pass}

{hints_section}

## AGENT TRAJECTORY

{trajectory_text}

**Outcome:** {resolved_status}

## YOUR TASK: Identify What Makes Difficulty Surprising

Based on the trajectory, help predict what the prior model gets WRONG:

### 1. Conceptual vs Mechanical Complexity (-1 to +1)
- **+1**: Patch is large but conceptually simple (e.g., changing return value, adding parameter, mechanical refactoring)
- **0**: Patch size matches conceptual complexity
- **-1**: Patch is small but conceptually hard (subtle bug, edge case, requires deep understanding)

Look for:
- Did the agent find the fix quickly despite large patch? → +1
- Did the agent struggle with a small fix? → -1
- Did the agent need to explore many files for a simple change? → +1

### 2. Hidden Trap Indicator (0 to 1)
Does this task have traps that aren't obvious from problem/patch length?
- 0: No hidden complexity
- 1: Significant hidden traps

Hidden traps include:
- Subtle API interactions (e.g., pandas transform + aggregation)
- Multi-model inheritance issues
- Edge cases in data structures
- Order-dependent behavior

### 3. Solution Directness (0 to 1)
- 0: Solution required extensive exploration, trial-and-error
- 1: Clear, direct path from problem to solution

Look for:
- How many false starts did the agent have?
- Did the agent find the right location quickly?
- Was the fix obvious once the location was found?

## Response Format

Respond with ONLY a JSON object:
{{
    "conceptual_vs_mechanical_complexity": <-1.0 to +1.0>,
    "hidden_trap_indicator": <0.0 to 1.0>,
    "solution_directness": <0.0 to 1.0>,

    "api_edge_case": <0 or 1>,
    "multi_component_interaction": <0 or 1>,
    "error_message_helpfulness": <1-4>,
    "fix_is_pattern_change": <0 or 1>,

    "exploration_breadth": <1-4>,
    "exploration_efficiency": <1-4>,
    "agent_confidence_mismatch": <0 or 1>,
    "repetitive_failed_attempts": <0 or 1>,

    "llm_judge_difficulty_score": <0.0 to 1.0>,
    "resolved_correctly": <0 or 1>,
    "trajectory_length_normalized": <0.0 to 2.0, where 1.0 is typical>,

    "reasoning": "<2-3 sentences explaining what makes this task's difficulty surprising>"
}}
"""


@dataclass
class LLMJudgeV2Features:
    """Features extracted from a trajectory using V2 residual-focused prompt."""

    conceptual_vs_mechanical_complexity: float
    hidden_trap_indicator: float
    solution_directness: float

    api_edge_case: int
    multi_component_interaction: int
    error_message_helpfulness: int
    fix_is_pattern_change: int

    exploration_breadth: int
    exploration_efficiency: int
    agent_confidence_mismatch: int
    repetitive_failed_attempts: int

    llm_judge_difficulty_score: float
    resolved_correctly: int
    trajectory_length_normalized: float

    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for Ridge regression."""
        return np.array([
            self.conceptual_vs_mechanical_complexity,
            self.hidden_trap_indicator,
            self.solution_directness,
            float(self.api_edge_case),
            float(self.multi_component_interaction),
            float(self.error_message_helpfulness),
            float(self.fix_is_pattern_change),
            float(self.exploration_breadth),
            float(self.exploration_efficiency),
            float(self.agent_confidence_mismatch),
            float(self.repetitive_failed_attempts),
            self.llm_judge_difficulty_score,
            float(self.resolved_correctly),
            self.trajectory_length_normalized,
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV2Features":
        """Create features from dict (from JSON response)."""
        return cls(
            conceptual_vs_mechanical_complexity=float(d.get("conceptual_vs_mechanical_complexity", 0)),
            hidden_trap_indicator=float(d.get("hidden_trap_indicator", 0.5)),
            solution_directness=float(d.get("solution_directness", 0.5)),
            api_edge_case=int(d.get("api_edge_case", 0)),
            multi_component_interaction=int(d.get("multi_component_interaction", 0)),
            error_message_helpfulness=int(d.get("error_message_helpfulness", 2)),
            fix_is_pattern_change=int(d.get("fix_is_pattern_change", 0)),
            exploration_breadth=int(d.get("exploration_breadth", 2)),
            exploration_efficiency=int(d.get("exploration_efficiency", 2)),
            agent_confidence_mismatch=int(d.get("agent_confidence_mismatch", 0)),
            repetitive_failed_attempts=int(d.get("repetitive_failed_attempts", 0)),
            llm_judge_difficulty_score=float(d.get("llm_judge_difficulty_score", 0.5)),
            resolved_correctly=int(d.get("resolved_correctly", 0)),
            trajectory_length_normalized=float(d.get("trajectory_length_normalized", 1.0)),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV2Features":
        """Return default features when no data available."""
        return cls(
            conceptual_vs_mechanical_complexity=0.0,
            hidden_trap_indicator=0.5,
            solution_directness=0.5,
            api_edge_case=0,
            multi_component_interaction=0,
            error_message_helpfulness=2,
            fix_is_pattern_change=0,
            exploration_breadth=2,
            exploration_efficiency=2,
            agent_confidence_mismatch=0,
            repetitive_failed_attempts=0,
            llm_judge_difficulty_score=0.5,
            resolved_correctly=0,
            trajectory_length_normalized=1.0,
        )


def load_llm_judge_v2_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV2Features]:
    """Load pre-computed V2 features for a specific task/agent pair."""
    features_path = features_dir / agent / f"{task_id}.json"
    if not features_path.exists():
        return None

    try:
        with open(features_path) as f:
            data = json.load(f)
        return LLMJudgeV2Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v2_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV2Features]:
    """Load V2 features for a task across multiple agents."""
    features = {}
    for agent in agents:
        feat = load_llm_judge_v2_features(task_id, agent, features_dir)
        if feat is not None:
            features[agent] = feat
    return features


def aggregate_llm_judge_v2_features(
    features: Dict[str, LLMJudgeV2Features],
) -> np.ndarray:
    """Aggregate V2 features across agents (mean aggregation)."""
    if not features:
        return LLMJudgeV2Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
