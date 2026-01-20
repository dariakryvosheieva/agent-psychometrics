"""LLM judge features V5: Features observable from FAILING trajectories.

V5 focuses on HOW the agent failed, not just effort/complexity ratios.
These features are designed to correlate with embedding prior residuals:
- High residual (harder than predicted) → agent misunderstood design intent
- Low residual (easier than predicted) → agent understood approach but made implementation error

Key insight: M1 agents mostly FAIL on D_train tasks, so features must be
observable from failing trajectories. The key distinguishing factor is whether
the agent understood WHAT needed to change vs just WHERE.

Features use INTEGER scales (1-5) for better LLM reliability.

After initial testing, only location_vs_fix_alignment showed correct correlation.
V5.1 adds design_intent_gap to capture misunderstanding of WHAT to do.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# V5 Features - focused on failure patterns
# After analysis: location_vs_fix_alignment works (+0.42 correlation)
# V5.1: Replaced design_intent_gap with solution_discoverability
# Key insight: measure if solution was DISCOVERABLE, not if agent discovered it
LLM_JUDGE_V5_FEATURE_NAMES = [
    "navigation_efficiency",      # 1-5: did agent find relevant code quickly?
    "reproduction_success",       # 1-5: could agent reproduce the issue?
    "location_vs_fix_alignment",  # 1-5: found right location but couldn't fix?
    "solution_discoverability",   # 1-5: could solution be discovered from code?
]


V5_PROMPT = """You are analyzing a FAILING SWE-bench agent trajectory to understand HOW the agent failed.

## CONTEXT

We want to predict task difficulty from agent failure patterns. Key insight:
- EASY tasks (that seem hard from text): Agent finds right area, understands the approach, but makes implementation mistake
- HARD tasks (that seem easy from text): Agent finds the location but misunderstands WHAT needs to change

The critical distinction is between knowing WHERE vs knowing WHAT.

## TASK METADATA

**Task ID:** {instance_id}
**Repository:** {repo}
**Problem Statement ({problem_len} chars):**
{problem_statement}

**Gold Patch ({patch_len} chars):**
```diff
{patch}
```

{hints_section}

## AGENT TRAJECTORY (FAILED)

{trajectory_text}

**Outcome:** {resolved_status}

## FEATURES TO EXTRACT (all 1-5 integers)

### 1. NAVIGATION EFFICIENCY (1-5)
How efficiently did the agent find relevant code?

- **1**: Agent found right file/function within first few searches
  - Example: Agent goes directly to the right module based on error message
  - This suggests: task is EASY, navigation is straightforward
- **2**: Agent found right area after a few wrong turns
- **3**: Agent found right area after moderate exploration
- **4**: Agent spent significant time searching before finding relevant code
- **5**: Agent wandered extensively without finding relevant code
  - Example: Heavy search-explore loops, never converging on right area
  - This suggests: task is HARD, codebase is confusing or problem is vague

COUNT: Number of searches/file opens before finding the file where gold patch applies.

### 2. REPRODUCTION SUCCESS (1-5)
Could the agent reproduce the issue described in the problem?

- **1**: Agent successfully reproduced exact issue with test script
  - Example: Agent created reproduce.py, saw exact error/warning described
  - This suggests: task is EASY, clear reproduction path
- **2**: Agent reproduced issue but not perfectly (similar but not exact error)
- **3**: Agent partially reproduced or had unclear reproduction
- **4**: Agent attempted reproduction but failed to see expected behavior
- **5**: Agent could not reproduce issue or didn't attempt
  - Example: Visual outputs, environment-specific behavior, complex setup
  - This suggests: task is HARD, problem is hard to observe

LOOK FOR: Did agent create test scripts? Did they see the expected error messages?

### 3. LOCATION VS FIX ALIGNMENT (1-5)
Did agent find the right location but fail to implement the correct fix?

- **1**: Agent never found the right location (wandering failure)
  - Agent failed by not discovering WHERE to fix
- **2**: Agent found same file as gold patch but wrong section
- **3**: Agent found right file but wrong function/class
- **4**: Agent found right function but wrong lines
- **5**: Agent found EXACT right location but couldn't implement fix
  - Example: Agent edited the same lines multiple times without solving
  - This indicates agent knew WHERE but not WHAT

COMPARE: Agent's edits vs gold patch location. Same file? Same function? Same lines?

### 4. SOLUTION DISCOVERABILITY (1-5)
Could the correct solution be DISCOVERED from available information?

This is the KEY feature: Easy tasks have solutions discoverable from context; hard tasks require external knowledge.

- **1**: Solution clearly implied by problem description or error messages
  - Example: Error "missing attribute X" → just add attribute X
  - Example: Traceback shows exact line and obvious fix
  - This suggests: EASY task - anyone could find the solution
- **2**: Solution discoverable by reading the code at the error location
- **3**: Solution requires exploring nearby code to understand patterns
- **4**: Solution requires understanding broader codebase architecture
- **5**: Solution requires domain-specific or "insider" knowledge not in the code
  - Example: Must know to REMOVE code, not add deprecation warning
  - Example: Must know `classes_` is a sklearn convention for estimators
  - Example: Must know MTI parent link ordering is significant in Django
  - Example: Must know the design rationale behind an architectural choice
  - This suggests: HARD task - solution not discoverable from context

KEY QUESTION: Given unlimited time to read the codebase, could a competent developer
discover the correct solution without asking a maintainer?

LOOK FOR signals of non-discoverable solutions:
- Gold patch does something counter-intuitive (remove instead of fix)
- Gold patch relies on conventions not documented in the code
- Gold patch requires understanding design decisions made elsewhere
- Agent tried multiple reasonable approaches that all failed

## Response Format

Respond with ONLY a JSON object (integers 1-5, not floats):
{{
    "navigation_efficiency": <1-5>,
    "reproduction_success": <1-5>,
    "location_vs_fix_alignment": <1-5>,
    "solution_discoverability": <1-5>,
    "reasoning": "<2-3 sentences: Was the correct solution DISCOVERABLE from the codebase, or did it require knowledge not present in the code?>"
}}

IMPORTANT:
- Use integers 1-5 only
- solution_discoverability is about whether the solution could be FOUND, not whether this agent found it
- High value (5) means: even a perfect agent couldn't discover this from the code alone
- Low value (1) means: the solution was obvious from available information
"""


@dataclass
class LLMJudgeV5Features:
    """V5 features - failure pattern analysis.

    After initial testing, location_vs_fix_alignment showed correct correlation (+0.42).
    V5.1 replaces design_intent_gap with solution_discoverability to measure if solution
    was discoverable from code, not just if agent discovered it.
    """

    navigation_efficiency: int       # 1-5 (1=efficient, 5=wandering)
    reproduction_success: int        # 1-5 (1=reproduced, 5=couldn't reproduce)
    location_vs_fix_alignment: int   # 1-5 (1=never found, 5=found but couldn't fix)
    solution_discoverability: int    # 1-5 (1=obvious solution, 5=requires insider knowledge)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (normalized to 0-1 range)."""
        # Normalize from 1-5 to 0-1
        return np.array([
            (self.navigation_efficiency - 1) / 4.0,
            (self.reproduction_success - 1) / 4.0,
            (self.location_vs_fix_alignment - 1) / 4.0,
            (self.solution_discoverability - 1) / 4.0,
        ])

    def to_raw_vector(self) -> np.ndarray:
        """Convert to feature vector (raw 1-5 values)."""
        return np.array([
            float(self.navigation_efficiency),
            float(self.reproduction_success),
            float(self.location_vs_fix_alignment),
            float(self.solution_discoverability),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV5Features":
        """Create from dict (JSON response).

        Handles backwards compatibility with old feature names.
        """
        # Handle backwards compatibility with old feature names
        sol_disc = d.get("solution_discoverability",
                        d.get("design_intent_gap",
                        d.get("exploration_breadth", 3)))
        return cls(
            navigation_efficiency=int(d.get("navigation_efficiency", 3)),
            reproduction_success=int(d.get("reproduction_success", 3)),
            location_vs_fix_alignment=int(d.get("location_vs_fix_alignment", 3)),
            solution_discoverability=int(sol_disc),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV5Features":
        """Return default features (middle values)."""
        return cls(
            navigation_efficiency=3,
            reproduction_success=3,
            location_vs_fix_alignment=3,
            solution_discoverability=3,
        )


def load_llm_judge_v5_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV5Features]:
    """Load V5 features for a task-agent pair."""
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV5Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v5_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV5Features]:
    """Load V5 features for a task across agents."""
    result = {}
    for agent in agents:
        features = load_llm_judge_v5_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v5_features(features: Dict[str, LLMJudgeV5Features]) -> np.ndarray:
    """Aggregate V5 features across agents (mean of normalized features)."""
    if not features:
        return LLMJudgeV5Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
