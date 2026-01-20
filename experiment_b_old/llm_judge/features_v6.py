"""LLM judge features V6 - Solution Discoverability.

This version focuses on a single high-signal feature: whether the solution
could be discovered from reading the codebase alone, or requires "insider
knowledge" (undocumented conventions, implicit requirements, etc.).

Based on analysis of high-residual tasks, undiscoverable solutions are
systematically harder than the prior predicts because:
1. The problem text doesn't reveal hidden requirements
2. Error messages don't point to the actual fix
3. Reading the code doesn't surface the necessary knowledge

Examples of undiscoverable solutions:
- Must know to REMOVE code entirely (not add deprecation warning)
- Must know `classes_` is a sklearn naming convention
- Must know Django MTI parent link ordering rules
- Must know a specific API quirk not documented in comments
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Single feature - solution discoverability
LLM_JUDGE_V6_FEATURE_NAMES = [
    "solution_discoverability",  # 1-5: can solution be discovered from code?
]


V6_DISCOVERABILITY_PROMPT = """You are analyzing a SWE-bench task to assess: Can a competent developer discover the correct solution by reading the codebase?

## TASK

**Task ID:** {instance_id}
**Repository:** {repo}

**Problem Statement:**
{problem_statement}

**Gold Patch (the correct fix):**
```diff
{patch}
```

## AGENT TRAJECTORY SUMMARY

The agent {resolved_status} this task.

Files the agent explored: {files_explored}
Files the agent edited: {files_edited}
Key errors encountered: {key_errors}

## YOUR TASK

Rate how DISCOVERABLE the solution is on a 1-5 scale:

**1 = Completely Undiscoverable**
- Solution requires insider knowledge not in the codebase
- Must know undocumented conventions (e.g., sklearn's `classes_` attribute naming)
- Must know to do the OPPOSITE of what the problem statement suggests
- Solution is counter-intuitive and not hinted at in code comments

**2 = Mostly Undiscoverable**
- Key insight is buried or implicit
- Code comments are misleading about the correct approach
- Must understand complex framework internals not obviously relevant
- Error messages actively misdirect toward wrong fix

**3 = Partially Discoverable**
- Solution can be found but requires significant exploration
- The relevant code area is not obvious from the problem
- Some implicit knowledge helps but isn't strictly required
- A good developer might find it after extensive search

**4 = Mostly Discoverable**
- Solution is findable with careful code reading
- The problem points to the general area
- Code comments or tests hint at the fix
- A competent developer should find it

**5 = Fully Discoverable**
- Solution is straightforward from reading the code
- Problem statement accurately describes what's needed
- The fix location is obvious from the error/description
- Any competent developer would find this

## EXAMPLES OF UNDISCOVERABLE PATTERNS

1. **Problem text mismatch**: Issue says "add deprecation warning" but fix is to DELETE code entirely
2. **Implicit conventions**: Must know `classes_` is how sklearn exposes fitted class labels
3. **Hidden dependencies**: Fix requires understanding psycopg2 import behavior
4. **Framework magic**: Must know Django's metaclass processes parent_links in specific order
5. **Counter-intuitive fix**: Fix is to NOT set an index when DataFrame already has one

## Response Format

Respond with ONLY a JSON object:
{{
    "solution_discoverability": <1-5>,
    "undiscoverable_elements": "<list any insider knowledge required, or 'none'>",
    "reasoning": "<2-3 sentences explaining your rating>"
}}

Use integers 1-5 only. Be conservative - if in doubt, rate LOWER (harder to discover).
"""


@dataclass
class LLMJudgeV6Features:
    """V6 feature - solution discoverability only."""

    solution_discoverability: int  # 1-5 (1=undiscoverable, 5=obvious)
    undiscoverable_elements: str   # Description of insider knowledge needed
    reasoning: str                 # Explanation

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (normalized to 0-1 range)."""
        # Invert so higher = harder (matches residual direction)
        # 1 (undiscoverable) -> 1.0, 5 (obvious) -> 0.0
        return np.array([
            1.0 - (self.solution_discoverability - 1) / 4.0,
        ])

    def to_raw_vector(self) -> np.ndarray:
        """Convert to feature vector (raw 1-5 values)."""
        return np.array([
            float(self.solution_discoverability),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeV6Features":
        """Create from dict (JSON response)."""
        return cls(
            solution_discoverability=int(d.get("solution_discoverability", 3)),
            undiscoverable_elements=str(d.get("undiscoverable_elements", "")),
            reasoning=str(d.get("reasoning", "")),
        )

    @classmethod
    def default(cls) -> "LLMJudgeV6Features":
        """Return default features (middle value)."""
        return cls(
            solution_discoverability=3,
            undiscoverable_elements="unknown",
            reasoning="default",
        )


def format_trajectory_summary(trajectory: dict) -> Dict[str, str]:
    """Extract summary info from trajectory for the prompt.

    Args:
        trajectory: Loaded trajectory JSON

    Returns:
        Dict with files_explored, files_edited, key_errors
    """
    messages = trajectory.get("messages", [])
    resolved = trajectory.get("resolved", False)

    # Extract files explored (from file views/opens)
    files_explored = set()
    files_edited = set()
    key_errors = []

    import re

    for msg in messages:
        content = msg.get("content", "")

        # Files opened/viewed
        for match in re.finditer(r'\[File: ([^\]]+)\]', content):
            files_explored.add(match.group(1).split('/')[-1])

        # Files edited
        if "File updated" in content or "edit" in content.lower():
            for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*\.py)', content):
                files_edited.add(match.group(1))

        # Key errors (first line of each error)
        for match in re.finditer(r'((?:Error|Exception|Traceback)[^\n]{0,100})', content):
            error = match.group(1)[:80]
            if error not in key_errors and len(key_errors) < 5:
                key_errors.append(error)

    return {
        "resolved_status": "RESOLVED" if resolved else "FAILED to resolve",
        "files_explored": ", ".join(list(files_explored)[:10]) or "none recorded",
        "files_edited": ", ".join(list(files_edited)[:10]) or "none",
        "key_errors": "; ".join(key_errors) or "none recorded",
    }


def build_v6_prompt(
    instance_id: str,
    repo: str,
    problem_statement: str,
    patch: str,
    trajectory: dict,
) -> str:
    """Build the V6 discoverability prompt.

    Args:
        instance_id: Task ID
        repo: Repository name
        problem_statement: Problem description
        patch: Gold patch diff
        trajectory: Loaded trajectory

    Returns:
        Formatted prompt string
    """
    summary = format_trajectory_summary(trajectory)

    return V6_DISCOVERABILITY_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        problem_statement=problem_statement[:6000],
        patch=patch[:4000],
        **summary,
    )


def load_llm_judge_v6_features(
    task_id: str,
    agent: str,
    features_dir: Path,
) -> Optional[LLMJudgeV6Features]:
    """Load V6 features for a task-agent pair.

    Args:
        task_id: Task instance ID
        agent: Agent name
        features_dir: Directory containing feature JSON files

    Returns:
        LLMJudgeV6Features or None if not found
    """
    feature_file = features_dir / agent / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeV6Features.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_v6_features_for_task(
    task_id: str,
    agents: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeV6Features]:
    """Load V6 features for a task across agents.

    Args:
        task_id: Task instance ID
        agents: List of agent names
        features_dir: Directory containing feature files

    Returns:
        Dict mapping agent -> LLMJudgeV6Features
    """
    result = {}
    for agent in agents:
        features = load_llm_judge_v6_features(task_id, agent, features_dir)
        if features is not None:
            result[agent] = features
    return result


def aggregate_llm_judge_v6_features(
    features: Dict[str, LLMJudgeV6Features],
) -> np.ndarray:
    """Aggregate V6 features across agents (mean of normalized features).

    Args:
        features: Dict mapping agent -> LLMJudgeV6Features

    Returns:
        Aggregated feature vector
    """
    if not features:
        return LLMJudgeV6Features.default().to_vector()

    vectors = [f.to_vector() for f in features.values()]
    return np.mean(vectors, axis=0)
