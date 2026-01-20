"""LLM judge features V4: Only features with correct residual correlation.

Based on V3 testing:
- KEEP: effort_to_solution_ratio (r=+0.44), problem_text_accuracy (r=+0.15), error_misdirection (r=+0.27)
- DROP: location_discoverability (r=-0.60), solution_path_clarity (r=-0.08), api_surprise (r=-0.32)

The dropped features had WRONG direction - they were higher for tasks where embedding
OVERESTIMATED difficulty, not underestimated.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# V4 Features - only those with correct correlation direction
LLM_JUDGE_V4_FEATURE_NAMES = [
    # Core features that predict residual correctly
    "effort_to_solution_ratio",     # -1 to +1: effort vs patch size mismatch
    "problem_text_accuracy",        # -1 to +1: how misleading was problem text?
    "error_misdirection",           # 0 or 1: errors pointed wrong way

    # Additional ratio-based features
    "fix_complexity_vs_description", # -1 to +1: was fix simpler or harder than described?
    "exploration_efficiency",        # 0 to 1: how much wasted effort?

    # Outcome
    "resolved",                     # 0 or 1: did agent solve it?
]


V4_PROMPT = """You are analyzing a SWE-bench agent trajectory to identify MISMATCHES between what the problem text suggested and what the agent actually experienced.

## CONTEXT

An embedding model predicts task difficulty from the problem statement and gold patch text.
Your job is to identify information that ONLY becomes visible from the agent's execution:
- Whether the agent's effort matched what the patch size would suggest
- Whether the problem description was misleading about what was needed
- Whether error messages helped or misled the agent

CRITICAL: Do NOT just rate "how hard was this task". Instead, identify WHERE THE TEXT WAS WRONG.

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

## AGENT TRAJECTORY

{trajectory_text}

**Outcome:** {resolved_status}

## FEATURES TO EXTRACT

### 1. EFFORT-TO-SOLUTION RATIO (-1.0 to +1.0)
Compare agent effort to patch size:
- **+1.0**: Agent struggled extensively for a small/simple patch
  - Many messages, errors, backtracking for what turned out to be a few-line fix
  - Indicates: task was HARDER than patch size suggests
- **0.0**: Effort matched patch complexity
- **-1.0**: Agent solved quickly despite large/complex patch
  - Few messages, direct path despite large diff
  - Indicates: task was EASIER than patch size suggests

KEY: Count messages, errors, backtracks. Compare to patch line count.

### 2. PROBLEM-TEXT ACCURACY (-1.0 to +1.0)
Did the problem statement prepare the agent correctly?
- **+1.0**: Problem was MISLEADING - agent discovered requirements not mentioned
  - Hidden edge cases, undocumented behaviors, misleading suggestions
  - Indicates: task was HARDER than problem suggested
- **0.0**: Problem accurately described the challenge
- **-1.0**: Problem was OVERCOMPLICATED - solution was simpler than described
  - Problem suggested complexity that wasn't there
  - Indicates: task was EASIER than problem suggested

KEY: Did agent discover surprises not in problem text?

### 3. ERROR MISDIRECTION (0 or 1)
Did error messages point the agent in the WRONG direction?
- **0**: Errors were helpful or neutral
- **1**: Errors actively misled the agent (pointed to wrong file, wrong cause)

### 4. FIX COMPLEXITY VS DESCRIPTION (-1.0 to +1.0)
Compare actual fix to how the problem described the needed change:
- **+1.0**: Fix was more intricate than problem implied (multiple interactions, edge cases)
- **0.0**: Fix matched problem's implied complexity
- **-1.0**: Fix was more straightforward than problem implied (simple pattern, one location)

### 5. EXPLORATION EFFICIENCY (0.0 to 1.0)
How much of the agent's effort was wasted on wrong paths?
- **0.0**: Agent went directly to solution (minimal wasted effort)
- **0.5**: Some exploration but mostly productive
- **1.0**: Most effort was wasted on wrong approaches

### 6. RESOLVED (0 or 1)
Did the agent successfully solve the task?

## Response Format

Respond with ONLY a JSON object:
{{
    "effort_to_solution_ratio": <-1.0 to +1.0>,
    "problem_text_accuracy": <-1.0 to +1.0>,
    "error_misdirection": <0 or 1>,
    "fix_complexity_vs_description": <-1.0 to +1.0>,
    "exploration_efficiency": <0.0 to 1.0>,
    "resolved": <0 or 1>,
    "reasoning": "<2-3 sentences: What did execution reveal that the problem text got wrong?>"
}}
"""


@dataclass
class LLMJudgeV4Features:
    """V4 features - only correct-direction correlations."""

    effort_to_solution_ratio: float      # -1 to +1
    problem_text_accuracy: float         # -1 to +1
    error_misdirection: int              # 0 or 1
    fix_complexity_vs_description: float # -1 to +1
    exploration_efficiency: float        # 0 to 1
    resolved: int                        # 0 or 1

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.effort_to_solution_ratio,
            self.problem_text_accuracy,
            float(self.error_misdirection),
            self.fix_complexity_vs_description,
            self.exploration_efficiency,
            float(self.resolved),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV4Features":
        """Create from dict (JSON response)."""
        return cls(
            effort_to_solution_ratio=float(d.get("effort_to_solution_ratio", 0)),
            problem_text_accuracy=float(d.get("problem_text_accuracy", 0)),
            error_misdirection=int(d.get("error_misdirection", 0)),
            fix_complexity_vs_description=float(d.get("fix_complexity_vs_description", 0)),
            exploration_efficiency=float(d.get("exploration_efficiency", 0.5)),
            resolved=int(d.get("resolved", 0)),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV4Features":
        """Return default features."""
        return cls(
            effort_to_solution_ratio=0.0,
            problem_text_accuracy=0.0,
            error_misdirection=0,
            fix_complexity_vs_description=0.0,
            exploration_efficiency=0.5,
            resolved=0,
        )


def load_llm_judge_v4_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV4Features]:
    """Load V4 features for a task-agent pair."""
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV4Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v4_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV4Features]:
    """Load V4 features for a task across agents."""
    result = {}
    for agent in agents:
        features = load_llm_judge_v4_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v4_features(features: Dict[str, LLMJudgeV4Features]) -> np.ndarray:
    """Aggregate V4 features across agents."""
    if not features:
        return LLMJudgeV4Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
