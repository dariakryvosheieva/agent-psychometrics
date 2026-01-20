"""LLM judge features V7 - Unified trajectory analysis.

This version combines semantic trajectory analysis with task discoverability
into a single LLM call. Features are designed to capture signal orthogonal
to what the embedding prior can see.

Features extracted:
1. error_misdirection (1-5): Did errors/tracebacks point to wrong location?
2. bug_reproduction_quality (1-5): How well did agent reproduce the bug?
3. location_vs_fix_gap (1-5): Found right location but couldn't fix? (from v5)
4. solution_discoverability (1-5): Can solution be found from reading code?

All features use 1-5 scale where higher = harder task (positive correlation with residual).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Feature names - all on 1-5 scale, higher = harder
LLM_JUDGE_V7_FEATURE_NAMES = [
    "error_misdirection",        # 1=errors helpful, 5=errors misleading
    "bug_reproduction_quality",  # 1=easily reproduced, 5=couldn't reproduce
    "location_vs_fix_gap",       # 1=wrong location, 5=right location but couldn't fix
    "solution_discoverability",  # 1=obvious from code, 5=requires insider knowledge
]


V7_UNIFIED_PROMPT = """You are analyzing a SWE-bench agent trajectory to extract features that predict task difficulty.

## TASK CONTEXT

**Task ID:** {instance_id}
**Repository:** {repo}

**Problem Statement:**
{problem_statement}

**Gold Patch (the correct fix):**
```diff
{patch}
```

**Gold patch files:** {gold_patch_files}

## AGENT TRAJECTORY

The agent {resolved_status} this task.

{trajectory_text}

## FEATURES TO EXTRACT

Rate each feature on a 1-5 scale. Higher scores indicate a HARDER task.

### 1. Error Misdirection (1-5)
Did the errors, tracebacks, and test failures point the agent toward or away from the correct fix location?

- **1 = Errors were helpful**: Errors clearly pointed to the files/functions in the gold patch
- **2 = Mostly helpful**: Errors mentioned relevant areas with some noise
- **3 = Neutral**: Errors provided little directional signal
- **4 = Somewhat misleading**: Errors pointed to related but wrong areas
- **5 = Actively misleading**: Errors pointed to completely wrong files/areas, or the real issue produced no errors

Look at: Which files appear in tracebacks vs gold patch files? Did test failures indicate the right problem?

### 2. Bug Reproduction Quality (1-5)
How well did the agent reproduce/understand the original bug before attempting to fix it?

- **1 = Perfect reproduction**: Agent clearly demonstrated the bug with a test case
- **2 = Good reproduction**: Agent showed they understood the bug behavior
- **3 = Partial**: Agent had rough understanding but missed nuances
- **4 = Poor**: Agent misunderstood the bug or couldn't trigger it
- **5 = No reproduction**: Agent never verified the bug existed, jumped straight to fixing

### 3. Location vs Fix Gap (1-5)
Did the agent find the right location but fail to implement the correct fix?

- **1 = Wrong location entirely**: Agent edited completely different files than gold patch
- **2 = Wrong area**: Same repo area but different files/classes
- **3 = Close**: Right file, possibly wrong function
- **4 = Very close**: Right function, wrong specific lines
- **5 = Exact location, wrong fix**: Agent edited the exact same lines as gold patch but couldn't get the fix right

Compare: Files/functions the agent edited vs files/functions in gold patch.

### 4. Solution Discoverability (1-5)
Could a competent developer discover this solution by reading the codebase?

- **1 = Obvious**: Fix is straightforward from reading the problem and code
- **2 = Findable**: Requires some exploration but clearly derivable
- **3 = Moderate**: Needs significant code reading and understanding
- **4 = Hidden**: Requires understanding undocumented patterns or framework internals
- **5 = Insider knowledge**: Requires knowing conventions/behaviors not in the code (e.g., sklearn's `classes_` naming, Django's MTI ordering)

Examples of undiscoverable (score 5):
- Problem says "add deprecation warning" but fix is to DELETE code
- Must know a naming convention like `classes_` that isn't documented
- Fix requires knowing implicit framework behavior

## OUTPUT FORMAT

Respond with ONLY a JSON object:
{{
    "error_misdirection": <1-5>,
    "bug_reproduction_quality": <1-5>,
    "location_vs_fix_gap": <1-5>,
    "solution_discoverability": <1-5>,
    "reasoning": {{
        "error_misdirection": "<1 sentence>",
        "bug_reproduction_quality": "<1 sentence>",
        "location_vs_fix_gap": "<1 sentence>",
        "solution_discoverability": "<1 sentence>"
    }}
}}

Use integers 1-5 only. Be conservative - when uncertain, rate toward HARDER (higher scores).
"""


@dataclass
class LLMJudgeV7Features:
    """V7 unified features from trajectory analysis."""

    error_misdirection: int        # 1-5: higher = more misleading errors
    bug_reproduction_quality: int  # 1-5: higher = harder to reproduce
    location_vs_fix_gap: int       # 1-5: higher = found location but couldn't fix
    solution_discoverability: int  # 1-5: higher = needs insider knowledge
    reasoning: Optional[Dict[str, str]] = None

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (normalized to 0-1 range).

        All features already have higher = harder, so we normalize directly.
        """
        return np.array([
            (self.error_misdirection - 1) / 4.0,
            (self.bug_reproduction_quality - 1) / 4.0,
            (self.location_vs_fix_gap - 1) / 4.0,
            (self.solution_discoverability - 1) / 4.0,
        ])

    def to_raw_vector(self) -> np.ndarray:
        """Convert to feature vector (raw 1-5 values)."""
        return np.array([
            float(self.error_misdirection),
            float(self.bug_reproduction_quality),
            float(self.location_vs_fix_gap),
            float(self.solution_discoverability),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV7Features":
        """Create from dict (JSON response)."""
        return cls(
            error_misdirection=int(d.get("error_misdirection", 3)),
            bug_reproduction_quality=int(d.get("bug_reproduction_quality", 3)),
            location_vs_fix_gap=int(d.get("location_vs_fix_gap", 3)),
            solution_discoverability=int(d.get("solution_discoverability", 3)),
            reasoning=d.get("reasoning"),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV7Features":
        """Return default features (middle values)."""
        return cls(
            error_misdirection=3,
            bug_reproduction_quality=3,
            location_vs_fix_gap=3,
            solution_discoverability=3,
        )


def format_trajectory_for_prompt(
    trajectory: dict,
    max_chars: int = 15000,
) -> str:
    """Format trajectory messages for the prompt.

    Args:
        trajectory: Loaded trajectory JSON
        max_chars: Maximum characters to include

    Returns:
        Formatted trajectory text
    """
    messages = trajectory.get("messages", [])

    # Skip system messages, format user/assistant
    formatted = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue

        # Truncate very long messages
        if len(content) > 2000:
            content = content[:1000] + "\n... [truncated] ...\n" + content[-500:]

        prefix = "AGENT:" if role == "assistant" else "OUTPUT:"
        text = f"{prefix}\n{content}\n"

        if total_chars + len(text) > max_chars:
            formatted.append("... [trajectory truncated for length] ...")
            break

        formatted.append(text)
        total_chars += len(text)

    return "\n".join(formatted)


def build_v7_prompt(
    instance_id: str,
    repo: str,
    problem_statement: str,
    patch: str,
    trajectory: dict,
    gold_patch_files: List[str],
) -> str:
    """Build the V7 unified prompt.

    Args:
        instance_id: Task ID
        repo: Repository name
        problem_statement: Problem description
        patch: Gold patch diff
        trajectory: Loaded trajectory
        gold_patch_files: List of files in gold patch

    Returns:
        Formatted prompt string
    """
    resolved = trajectory.get("resolved", False)
    trajectory_text = format_trajectory_for_prompt(trajectory)

    return V7_UNIFIED_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        problem_statement=problem_statement[:6000],
        patch=patch[:4000],
        gold_patch_files=", ".join(gold_patch_files) or "unknown",
        resolved_status="RESOLVED" if resolved else "FAILED to resolve",
        trajectory_text=trajectory_text,
    )


def load_llm_judge_v7_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV7Features]:
    """Load V7 features for a task-agent pair.

    Args:
        task_id: Task instance ID
        agent: Agent name
        features_dir: Directory containing feature JSON files

    Returns:
        LLMJudgeV7Features or None if not found
    """
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV7Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v7_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV7Features]:
    """Load V7 features for a task across agents.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        features_dir: Directory containing feature files

    Returns:
        Dict mapping agent -> LLMJudgeV7Features
    """
    result = {}
    for agent in agents:
        features = load_llm_judge_v7_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v7_features(
    features: Dict[str, LLMJudgeV7Features],
) -> np.ndarray:
    """Aggregate V7 features across agents (mean of normalized features).

    Args:
        features: Dict mapping agent -> LLMJudgeV7Features

    Returns:
        Aggregated feature vector
    """
    if not features:
        return LLMJudgeV7Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
