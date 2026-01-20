"""LLM judge features V3: Orthogonal to embeddings.

KEY INSIGHT FROM ANALYSIS:
- Embedding prior captures text-based complexity (problem length, patch size, repo)
- LLM judge V1 features (difficulty_score, failure modes) are CORRELATED with
  what embeddings measure → no new information
- Features need to be ORTHOGONAL to text complexity

WHAT EMBEDDINGS MISS (execution-time information):
1. Whether fix complexity matches problem complexity (ratio, not absolute)
2. How much trial-and-error was needed (normalized by solution complexity)
3. Whether agent found solution location quickly vs slowly
4. Gap between agent confidence and actual success

This V3 focuses on RATIOS and COMPARISONS that embeddings can't see from text alone.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# V3 Features - orthogonal to what embeddings capture
LLM_JUDGE_V3_FEATURE_NAMES = [
    # Core ratio/comparison features (what embeddings miss)
    "effort_to_solution_ratio",     # -1 to +1: effort vs patch size mismatch
    "problem_text_accuracy",        # -1 to +1: how misleading was problem text?
    "location_discoverability",     # 0 to 1: how hard to find fix location?
    "solution_path_clarity",        # 0 to 1: how clear was fix once located?

    # Binary execution-time surprises
    "api_surprise",                 # 0 or 1: unexpected API behavior
    "error_misdirection",           # 0 or 1: errors pointed wrong way
    "test_surprise",                # 0 or 1: unexpected test behavior
    "multi_file_coordination",      # 0 or 1: needed multi-file understanding

    # Outcome
    "resolved",                     # 0 or 1: did agent solve it?
]


RESIDUAL_AWARE_PROMPT = """You are analyzing a SWE-bench agent trajectory to identify EXECUTION-TIME SURPRISES.

## YOUR GOAL

An embedding model predicted this task's difficulty based on the problem text and patch.
Your job is to identify aspects of the task that the embeddings CANNOT see:
- Things that only become visible when code is actually executed
- Gaps between what the problem text suggests and what the agent experienced
- Information about the "shape" of the solution path, not just the endpoint

IMPORTANT: Do NOT just rate "how hard was this task" - the embedding already does that.
Instead, identify WHERE THE TEXT WAS MISLEADING about difficulty.

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
Compare the agent's effort to the patch size:
- **+1.0**: Agent struggled extensively for what turned out to be a small/simple fix
  (patch is small but agent took many steps, made many errors)
- **0.0**: Agent effort was proportional to patch complexity
- **-1.0**: Agent found solution quickly despite a large/complex-looking patch
  (patch is large but agent solved it efficiently)

This is NOT about absolute difficulty - it's about whether effort matched expectations.

### 2. PROBLEM-TEXT ACCURACY (-1.0 to +1.0)
How well did the problem statement predict what the agent needed to do?
- **+1.0**: Problem description was misleading or incomplete - agent discovered
  requirements not mentioned (hidden edge cases, undocumented behaviors)
- **0.0**: Problem accurately described the challenge
- **-1.0**: Problem made task sound harder than it was - solution was more
  straightforward than the description implied

### 3. LOCATION DISCOVERABILITY (0.0 to 1.0)
How hard was it to find WHERE to make the fix?
- **0.0**: Agent found the right file/function immediately (obvious from problem)
- **0.5**: Agent needed some exploration but location was findable
- **1.0**: Agent struggled significantly to locate the relevant code

### 4. SOLUTION PATH CLARITY (0.0 to 1.0)
Once at the right location, how clear was the fix?
- **0.0**: Fix was obvious once location was found
- **0.5**: Some trial and error needed
- **1.0**: Even at right location, agent struggled to formulate correct fix

### 5. BINARY SIGNALS (0 or 1)
- **api_surprise**: Agent was surprised by API behavior (unexpected returns, side effects)
- **error_misdirection**: Error messages pointed to wrong location or cause
- **test_surprise**: Tests failed for unexpected reasons, or passed unexpectedly
- **multi_file_coordination**: Fix required understanding interactions between files

### 6. OUTCOME
- **resolved**: 0 or 1 - did agent succeed?

## Response Format

Respond with ONLY a JSON object:
{{
    "effort_to_solution_ratio": <-1.0 to +1.0>,
    "problem_text_accuracy": <-1.0 to +1.0>,
    "location_discoverability": <0.0 to 1.0>,
    "solution_path_clarity": <0.0 to 1.0>,

    "api_surprise": <0 or 1>,
    "error_misdirection": <0 or 1>,
    "test_surprise": <0 or 1>,
    "multi_file_coordination": <0 or 1>,

    "resolved": <0 or 1>,

    "reasoning": "<2-3 sentences explaining what the trajectory revealed that wasn't obvious from the problem text>"
}}
"""


@dataclass
class LLMJudgeV3Features:
    """V3 features - orthogonal to embeddings."""

    # Core ratio/comparison features
    effort_to_solution_ratio: float      # -1 to +1
    problem_text_accuracy: float         # -1 to +1
    location_discoverability: float      # 0 to 1
    solution_path_clarity: float         # 0 to 1

    # Binary execution-time surprises
    api_surprise: int                    # 0 or 1
    error_misdirection: int              # 0 or 1
    test_surprise: int                   # 0 or 1
    multi_file_coordination: int         # 0 or 1

    # Outcome
    resolved: int                        # 0 or 1

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.effort_to_solution_ratio,
            self.problem_text_accuracy,
            self.location_discoverability,
            self.solution_path_clarity,
            float(self.api_surprise),
            float(self.error_misdirection),
            float(self.test_surprise),
            float(self.multi_file_coordination),
            float(self.resolved),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV3Features":
        """Create from dict (JSON response)."""
        return cls(
            effort_to_solution_ratio=float(d.get("effort_to_solution_ratio", 0)),
            problem_text_accuracy=float(d.get("problem_text_accuracy", 0)),
            location_discoverability=float(d.get("location_discoverability", 0.5)),
            solution_path_clarity=float(d.get("solution_path_clarity", 0.5)),
            api_surprise=int(d.get("api_surprise", 0)),
            error_misdirection=int(d.get("error_misdirection", 0)),
            test_surprise=int(d.get("test_surprise", 0)),
            multi_file_coordination=int(d.get("multi_file_coordination", 0)),
            resolved=int(d.get("resolved", 0)),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV3Features":
        """Return default features."""
        return cls(
            effort_to_solution_ratio=0.0,
            problem_text_accuracy=0.0,
            location_discoverability=0.5,
            solution_path_clarity=0.5,
            api_surprise=0,
            error_misdirection=0,
            test_surprise=0,
            multi_file_coordination=0,
            resolved=0,
        )


def load_llm_judge_v3_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV3Features]:
    """Load V3 features for a task-agent pair."""
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV3Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v3_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV3Features]:
    """Load V3 features for a task across agents."""
    result = {}
    for agent in agents:
        features = load_llm_judge_v3_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v3_features(features: Dict[str, LLMJudgeV3Features]) -> np.ndarray:
    """Aggregate V3 features across agents."""
    if not features:
        return LLMJudgeV3Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
