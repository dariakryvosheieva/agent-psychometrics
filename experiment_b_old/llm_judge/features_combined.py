"""LLM judge combined features (problem + trajectory features).

This module extracts 13 features total:
- 9 problem-level features from the prior model (Experiment A)
- 4 trajectory-level features from V5 (designed for single failing agent)

The combined prompt analyzes both static task information AND a single agent's
trajectory to extract all features in one LLM call. This allows direct comparison
between:
1. Prior model (9 problem features only)
2. Combined model (9 problem + 4 trajectory features)

If Combined > Prior, trajectory data contains additional signal for difficulty prediction.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# All 13 features (9 problem + 4 trajectory)
LLM_JUDGE_COMBINED_FEATURE_NAMES = [
    # Problem features (9)
    "fix_in_description",
    "problem_clarity",
    "error_message_provided",
    "reproduction_steps",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
    # Trajectory features (4 from V5)
    "navigation_efficiency",
    "reproduction_success",
    "location_vs_fix_alignment",
    "solution_discoverability",
]

PROBLEM_FEATURE_NAMES = LLM_JUDGE_COMBINED_FEATURE_NAMES[:9]
TRAJECTORY_FEATURE_NAMES = LLM_JUDGE_COMBINED_FEATURE_NAMES[9:]


LLM_JUDGE_COMBINED_PROMPT = """You are analyzing a SWE-bench coding task AND an agent's trajectory to predict task difficulty.
You will analyze both the static task information AND how the agent approached the problem.

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

{hints_section}

## AGENT TRAJECTORY

The agent {resolved_status} this task.

{trajectory_text}

## FEATURES TO EVALUATE

You will evaluate 13 features total: 9 about the problem itself, and 4 about how the agent approached it.

### PROBLEM FEATURES (from static task information)

#### 1. Fix Information in Description (fix_in_description: 0-3)
Does the problem statement contain or hint at the solution?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of what needs to change
- 3: Exact code fix or detailed solution provided

#### 2. Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the problem?
- 1: Very vague, unclear what's actually wrong
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with reproduction steps and expected behavior

#### 3. Error Message/Traceback (error_message_provided: 0/1)
Does the problem include an error message or traceback?
- 0: No error message provided
- 1: Error message, traceback, or exception shown

#### 4. Reproduction Steps (reproduction_steps: 0/1)
Are concrete reproduction steps provided?
- 0: No concrete reproduction steps
- 1: Has reproduction steps (code snippet, test case, or commands)

#### 5. Fix Locality (fix_locality: 1-3)
How localized is the fix based on the patch?
- 1: Single location, few lines changed (1-5 lines)
- 2: Multiple locations in same file, or moderate changes (6-20 lines)
- 3: Multiple files or significant changes (>20 lines)

#### 6. Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed to understand and fix this?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge

#### 7. Fix Complexity (fix_complexity: 1-5)
How complex is the actual fix?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases, tricky bugs)

#### 8. Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed to arrive at the fix?
- 1: Mechanical fix, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior

#### 9. Atypicality (atypicality: 1-5)
How unusual is this bug pattern?
- 1: Very common bug pattern (typo, off-by-one, missing null check)
- 2: Common pattern (incorrect condition, wrong default)
- 3: Moderately unusual
- 4: Unusual bug pattern
- 5: Rare or novel bug pattern

### TRAJECTORY FEATURES (from agent behavior)

#### 10. Navigation Efficiency (navigation_efficiency: 1-5)
How efficiently did the agent find relevant code?
- 1: Agent found right file/function within first few searches (EASY navigation)
- 2: Agent found right area after a few wrong turns
- 3: Agent found right area after moderate exploration
- 4: Agent spent significant time searching before finding relevant code
- 5: Agent wandered extensively without finding relevant code (HARD navigation)

Count: How many file opens/searches before finding the file where gold patch applies?

#### 11. Reproduction Success (reproduction_success: 1-5)
Could the agent reproduce the issue described in the problem?
- 1: Agent successfully reproduced exact issue with test script (EASY)
- 2: Agent reproduced issue but not perfectly
- 3: Agent partially reproduced or had unclear reproduction
- 4: Agent attempted reproduction but failed to see expected behavior
- 5: Agent could not reproduce issue or didn't attempt (HARD)

Look for: Did agent create test scripts? Did they see the expected error messages?

#### 12. Location vs Fix Alignment (location_vs_fix_alignment: 1-5)
Did agent find the right location but fail to implement the correct fix?
- 1: Agent never found the right location (wandering failure)
- 2: Agent found same file as gold patch but wrong section
- 3: Agent found right file but wrong function/class
- 4: Agent found right function but wrong lines
- 5: Agent found EXACT right location but couldn't implement fix (knew WHERE but not WHAT)

Compare: Agent's edits vs gold patch location. Same file? Same function? Same lines?

#### 13. Solution Discoverability (solution_discoverability: 1-5)
Could the correct solution be DISCOVERED from available information?
- 1: Solution clearly implied by problem description or error messages (EASY)
- 2: Solution discoverable by reading the code at the error location
- 3: Solution requires exploring nearby code to understand patterns
- 4: Solution requires understanding broader codebase architecture
- 5: Solution requires domain-specific or "insider" knowledge not in the code (HARD)

Examples of undiscoverable (score 5):
- Must know to REMOVE code, not add deprecation warning
- Must know `classes_` is a sklearn convention for estimators
- Must know MTI parent link ordering is significant in Django

## OUTPUT FORMAT

Respond with ONLY a JSON object containing all 13 features. No markdown, no extra text.

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
    "navigation_efficiency": <1-5>,
    "reproduction_success": <1-5>,
    "location_vs_fix_alignment": <1-5>,
    "solution_discoverability": <1-5>,
    "reasoning": "<2-3 sentence summary of key difficulty factors from both problem and trajectory>"
}}
"""


@dataclass
class LLMJudgeCombinedFeatures:
    """Combined features - 9 problem + 4 trajectory features."""

    # Problem features (9)
    fix_in_description: int
    problem_clarity: int
    error_message_provided: int
    reproduction_steps: int
    fix_locality: int
    domain_knowledge_required: int
    fix_complexity: int
    logical_reasoning_required: int
    atypicality: int
    # Trajectory features (4)
    navigation_efficiency: int
    reproduction_success: int
    location_vs_fix_alignment: int
    solution_discoverability: int
    # Optional
    reasoning: Optional[str] = None

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector (all 13 features, normalized to 0-1 range)."""
        return np.array([
            # Problem features
            self.fix_in_description / 3.0,
            (self.problem_clarity - 1) / 4.0,
            float(self.error_message_provided),
            float(self.reproduction_steps),
            (self.fix_locality - 1) / 2.0,
            (self.domain_knowledge_required - 1) / 4.0,
            (self.fix_complexity - 1) / 4.0,
            (self.logical_reasoning_required - 1) / 4.0,
            (self.atypicality - 1) / 4.0,
            # Trajectory features
            (self.navigation_efficiency - 1) / 4.0,
            (self.reproduction_success - 1) / 4.0,
            (self.location_vs_fix_alignment - 1) / 4.0,
            (self.solution_discoverability - 1) / 4.0,
        ])

    def to_problem_vector(self) -> np.ndarray:
        """Get only the 9 problem features (for comparison with prior)."""
        return self.to_vector()[:9]

    def to_trajectory_vector(self) -> np.ndarray:
        """Get only the 4 trajectory features."""
        return self.to_vector()[9:]

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
            float(self.navigation_efficiency),
            float(self.reproduction_success),
            float(self.location_vs_fix_alignment),
            float(self.solution_discoverability),
        ])

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMJudgeCombinedFeatures":
        """Create from dict (JSON response)."""
        return cls(
            # Problem features
            fix_in_description=int(d.get("fix_in_description", 1)),
            problem_clarity=int(d.get("problem_clarity", 3)),
            error_message_provided=int(d.get("error_message_provided", 0)),
            reproduction_steps=int(d.get("reproduction_steps", 0)),
            fix_locality=int(d.get("fix_locality", 2)),
            domain_knowledge_required=int(d.get("domain_knowledge_required", 3)),
            fix_complexity=int(d.get("fix_complexity", 3)),
            logical_reasoning_required=int(d.get("logical_reasoning_required", 3)),
            atypicality=int(d.get("atypicality", 3)),
            # Trajectory features
            navigation_efficiency=int(d.get("navigation_efficiency", 3)),
            reproduction_success=int(d.get("reproduction_success", 3)),
            location_vs_fix_alignment=int(d.get("location_vs_fix_alignment", 3)),
            solution_discoverability=int(d.get("solution_discoverability", 3)),
            reasoning=d.get("reasoning"),
        )

    @classmethod
    def default(cls) -> "LLMJudgeCombinedFeatures":
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
            navigation_efficiency=3,
            reproduction_success=3,
            location_vs_fix_alignment=3,
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


def format_combined_prompt(
    instance_id: str,
    repo: str,
    version: str,
    problem_statement: str,
    patch: str,
    fail_to_pass: str,
    hints_text: str,
    trajectory: dict,
) -> str:
    """Format the combined prompt with task and trajectory information.

    Args:
        instance_id: SWE-bench instance ID
        repo: Repository name
        version: Version string
        problem_statement: The issue description
        patch: The gold solution patch
        fail_to_pass: Tests that should pass after fix
        hints_text: Optional hints
        trajectory: Loaded trajectory JSON

    Returns:
        Formatted prompt string
    """
    hints_section = ""
    if hints_text and hints_text.strip():
        hints_section = f"**Hints:**\n{hints_text}"

    # Truncate long fields
    problem_statement = (
        problem_statement[:12000] if len(problem_statement) > 12000 else problem_statement
    )
    patch = patch[:8000] if len(patch) > 8000 else patch

    resolved = trajectory.get("resolved", False)
    trajectory_text = format_trajectory_for_prompt(trajectory)

    return LLM_JUDGE_COMBINED_PROMPT.format(
        instance_id=instance_id,
        repo=repo,
        version=version,
        problem_statement=problem_statement,
        patch=patch,
        fail_to_pass=fail_to_pass,
        hints_section=hints_section,
        resolved_status="RESOLVED" if resolved else "FAILED to resolve",
        trajectory_text=trajectory_text,
    )


def load_llm_judge_combined_features(
    task_id: str,
    features_dir: Path,
) -> Optional[LLMJudgeCombinedFeatures]:
    """Load combined features for a task.

    Args:
        task_id: Task instance ID
        features_dir: Directory containing feature JSON files

    Returns:
        LLMJudgeCombinedFeatures or None if not found
    """
    feature_file = features_dir / f"{task_id}.json"
    if not feature_file.exists():
        return None

    try:
        with open(feature_file) as f:
            data = json.load(f)
        return LLMJudgeCombinedFeatures.from_dict(data)
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def load_llm_judge_combined_features_batch(
    task_ids: List[str],
    features_dir: Path,
) -> Dict[str, LLMJudgeCombinedFeatures]:
    """Load combined features for multiple tasks.

    Args:
        task_ids: List of task instance IDs
        features_dir: Directory containing feature files

    Returns:
        Dict mapping task_id -> LLMJudgeCombinedFeatures
    """
    result = {}
    for task_id in task_ids:
        features = load_llm_judge_combined_features(task_id, features_dir)
        if features is not None:
            result[task_id] = features
    return result
