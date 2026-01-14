"""LLM judge features for prior model (problem features only, no trajectory).

This module extracts the 9 semantic features from Experiment A's prompt,
allowing direct comparison between:
1. Prior model (problem features only)
2. Combined model (problem + trajectory features)

Features are extracted from static task information only:
- Problem statement
- Gold patch
- Tests (FAIL_TO_PASS, PASS_TO_PASS)
- Hints

This establishes the baseline for testing whether trajectory data adds signal.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# 9 problem-level features from Experiment A
LLM_JUDGE_PRIOR_FEATURE_NAMES = [
    "fix_in_description",          # 0-3: Does problem hint at solution?
    "problem_clarity",             # 1-5: How clear is the problem?
    "error_message_provided",      # 0/1: Is error message included?
    "reproduction_steps",          # 0/1: Are reproduction steps provided?
    "fix_locality",                # 1-3: How localized is the fix?
    "domain_knowledge_required",   # 1-5: Specialized knowledge needed?
    "fix_complexity",              # 1-5: How complex is the fix?
    "logical_reasoning_required",  # 1-5: Reasoning required to find fix?
    "atypicality",                 # 1-5: How unusual is the bug?
]


LLM_JUDGE_PRIOR_PROMPT = """You are analyzing a SWE-bench coding task to predict its difficulty.
You will analyze ONLY the static task information (no agent trajectories).

## TASK INFORMATION

**Instance ID:** {instance_id}
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

**Regression tests (PASS_TO_PASS):**
{pass_to_pass}

{hints_section}

## FEATURES TO EVALUATE

Analyze the problem statement and gold patch to evaluate these 9 semantic features.
Be precise and consistent with your ratings.

### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

### 2. Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the problem?
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior

### 3. Error Message/Traceback (error_message_provided: 0/1)
Does the problem include an error message or traceback?
- 0: No error message provided
- 1: Error message, traceback, or exception shown

### 4. Reproduction Steps (reproduction_steps: 0/1)
Are concrete reproduction steps provided?
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)

### 5. Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

### 6. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

### 7. Fix Complexity (fix_complexity: 1-5)
How complex is the actual fix?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

### 8. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed to arrive at the fix?
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior

### 9. Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all features. No markdown, no extra text.

{{
    "fix_in_description": <0-3>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}
"""


@dataclass
class LLMJudgePriorFeatures:
    """Prior model features - 9 problem-level features from Experiment A."""

    fix_in_description: int        # 0-3
    problem_clarity: int           # 1-5
    error_message_provided: int    # 0/1
    reproduction_steps: int        # 0/1
    fix_locality: int              # 1-3
    domain_knowledge_required: int # 1-5
    fix_complexity: int            # 1-5
    logical_reasoning_required: int # 1-5
    atypicality: int               # 1-5
    reasoning: Optional[str] = None

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (normalized to 0-1 range)."""
        return np.array([
            self.fix_in_description / 3.0,
            (self.problem_clarity - 1) / 4.0,
            float(self.error_message_provided),
            float(self.reproduction_steps),
            (self.fix_locality - 1) / 2.0,
            (self.domain_knowledge_required - 1) / 4.0,
            (self.fix_complexity - 1) / 4.0,
            (self.logical_reasoning_required - 1) / 4.0,
            (self.atypicality - 1) / 4.0,
        ])

    def to_raw_vector(self) -> np.ndarray:
        """Convert to feature vector (raw values)."""
        return np.array([
            float(self.fix_in_description),
            float(self.problem_clarity),
            float(self.error_message_provided),
            float(self.reproduction_steps),
            float(self.fix_locality),
            float(self.domain_knowledge_required),
            float(self.fix_complexity),
            float(self.logical_reasoning_required),
            float(self.atypicality),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgePriorFeatures":
        """Create from dict (JSON response)."""
        return cls(
            fix_in_description=int(d.get("fix_in_description", 1)),
            problem_clarity=int(d.get("problem_clarity", 3)),
            error_message_provided=int(d.get("error_message_provided", 0)),
            reproduction_steps=int(d.get("reproduction_steps", 0)),
            fix_locality=int(d.get("fix_locality", 2)),
            domain_knowledge_required=int(d.get("domain_knowledge_required", 3)),
            fix_complexity=int(d.get("fix_complexity", 3)),
            logical_reasoning_required=int(d.get("logical_reasoning_required", 3)),
            atypicality=int(d.get("atypicality", 3)),
            reasoning=d.get("reasoning"),
        )

    @classmethod
    def default(cls) -> "LLMJudgePriorFeatures":
        """Return default features (middle values)."""
        return cls(
            fix_in_description=1,
            problem_clarity=3,
            error_message_provided=0,
            reproduction_steps=0,
            fix_locality=2,
            domain_knowledge_required=3,
            fix_complexity=3,
            logical_reasoning_required=3,
            atypicality=3,
        )


def format_prior_prompt(
    instance_id: str,
    repo: str,
    version: str,
    problem_statement: str,
    patch: str,
    fail_to_pass: str,
    pass_to_pass: str,
    hints_text: str = "",
) -> str:
    """Format the prior prompt with task-specific information.

    Args:
        instance_id: SWE-bench instance ID
        repo: Repository name (e.g., "django/django")
        version: Version string
        problem_statement: The issue description
        patch: The gold solution patch
        fail_to_pass: Tests that should pass after fix
        pass_to_pass: Regression tests
        hints_text: Optional hints

    Returns:
        Formatted prompt string
    """
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate very long fields to avoid context overflow
    problem_statement = (
        problem_statement[:12000] if len(problem_statement) > 12000 else problem_statement
    )
    patch = patch[:8000] if len(patch) > 8000 else patch

    return LLM_JUDGE_PRIOR_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        version=version,
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        hints_section=hints_section,
    )


def load_llm_judge_prior_features(
    task_id: str,
    features_dir: Path,
) -> Optional[LLMJudgePriorFeatures]:
    """Load prior features for a task.

    Args:
        task_id: Task instance ID
        features_dir: Directory containing feature JSON files

    Returns:
        LLMJudgePriorFeatures or None if not found
    """
    feature_file = features_dir / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgePriorFeatures.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_prior_features_batch(
    task_ids: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgePriorFeatures]:
    """Load prior features for multiple tasks.

    Args:
        task_ids: List of task instance IDs
        features_dir: Directory containing feature files

    Returns:
        Dict mapping task_id -> LLMJudgePriorFeatures
    """
    result = {}
    for task_id in task_ids:
        features = load_llm_judge_prior_features(task_id, features_dir)
        if features is not None:
            result[task_id] = features
    return result
