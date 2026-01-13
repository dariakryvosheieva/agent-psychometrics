"""LLM judge features V5 - Single feature version.

This version extracts only location_vs_fix_alignment to avoid feature interference.
Based on testing, this is the only feature that showed correct positive correlation
with residual difficulty (r=+0.794, p=0.006 on 10 extreme tasks).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Single feature - the only one that works
LLM_JUDGE_V5_SINGLE_FEATURE_NAMES = [
    "location_vs_fix_alignment",  # 1-5: found right location but couldn't fix?
]


V5_SINGLE_PROMPT = """You are analyzing a SWE-bench agent trajectory to measure one thing:
Did the agent find the RIGHT LOCATION but fail to implement the correct fix?

## TASK

**Task ID:** {instance_id}
**Repository:** {repo}

**Problem Statement:**
{problem_statement}

**Gold Patch (the correct fix):**
```diff
{patch}
```

## AGENT TRAJECTORY

{trajectory_text}

**Outcome:** {resolved_status}

## YOUR TASK

Compare WHERE the agent made edits vs WHERE the gold patch applies.

Rate on a scale of 1-5:

- **1**: Agent never found the right location
  - Agent edited completely different files than the gold patch
  - Agent wandered without finding relevant code

- **2**: Agent found the same FILE as gold patch but wrong section
  - Right file, but different class or unrelated functions

- **3**: Agent found the right FILE and general AREA
  - Same file, possibly same class, but different functions

- **4**: Agent found the right FUNCTION but wrong lines
  - Agent edited the correct function but targeted wrong lines

- **5**: Agent found the EXACT right location but couldn't fix it
  - Agent edited the same lines as the gold patch
  - Agent may have edited these lines multiple times without success
  - This indicates the agent knew WHERE but not WHAT to change

## HOW TO COMPARE

1. Look at which files the agent edited (search for "edit" commands in trajectory)
2. Compare to the files in the gold patch diff
3. If same file, check if same function/class
4. If same function, check if same lines

## Response Format

Respond with ONLY a JSON object:
{{
    "location_vs_fix_alignment": <1-5>,
    "agent_files_edited": "<list main files agent edited>",
    "gold_patch_files": "<list files in gold patch>",
    "reasoning": "<1-2 sentences: How close was the agent to the right location?>"
}}

Use integers 1-5 only.
"""


@dataclass
class LLMJudgeV5SingleFeatures:
    """V5 single feature - location vs fix alignment only."""

    location_vs_fix_alignment: int   # 1-5 (1=never found, 5=found but couldn't fix)

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (normalized to 0-1 range)."""
        return np.array([
            (self.location_vs_fix_alignment - 1) / 4.0,
        ])

    def to_raw_vector(self) -> np.ndarray:
        """Convert to feature vector (raw 1-5 values)."""
        return np.array([
            float(self.location_vs_fix_alignment),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV5SingleFeatures":
        """Create from dict (JSON response)."""
        return cls(
            location_vs_fix_alignment=int(d.get("location_vs_fix_alignment", 3)),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV5SingleFeatures":
        """Return default features (middle value)."""
        return cls(
            location_vs_fix_alignment=3,
        )


def load_llm_judge_v5_single_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV5SingleFeatures]:
    """Load V5 single features for a task-agent pair."""
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV5SingleFeatures.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v5_single_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV5SingleFeatures]:
    """Load V5 single features for a task across agents."""
    result = {}
    for agent in agents:
        features = load_llm_judge_v5_single_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v5_single_features(features: Dict[str, LLMJudgeV5SingleFeatures]) -> np.ndarray:
    """Aggregate V5 single features across agents (mean of normalized features)."""
    if not features:
        return LLMJudgeV5SingleFeatures.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
