"""Extended problem-only features for SWE-bench ablation study.

This extracts 8 ADDITIONAL features from the problem statement alone,
to be combined with the existing 7 problem-only features for a total of 15.

These features are designed to capture aspects of difficulty that can be
assessed purely from the problem description, without seeing the solution,
test patch, or exploring the environment.
"""

from typing import Any, Dict, List

from experiment_ab_shared.llm_judge.prompt_config import PromptConfig, FeatureDefinition


# =============================================================================
# Extended Problem-Only Features (8 new features)
# =============================================================================

ERROR_SPECIFICITY = FeatureDefinition(
    name="error_specificity",
    min_value=1,
    max_value=5,
    description="How specific is the error/bug description? (1=vague symptoms, 5=exact error with stack trace)",
)

REPRODUCTION_CLARITY = FeatureDefinition(
    name="reproduction_clarity",
    min_value=1,
    max_value=5,
    description="How clear are the reproduction steps? (1=unclear, 5=exact steps given)",
)

EXPECTED_BEHAVIOR_CLARITY = FeatureDefinition(
    name="expected_behavior_clarity",
    min_value=1,
    max_value=5,
    description="How clear is what the correct behavior should be? (1=ambiguous, 5=precisely specified)",
)

DEBUGGING_COMPLEXITY = FeatureDefinition(
    name="debugging_complexity",
    min_value=1,
    max_value=5,
    description="How complex would debugging/root cause analysis be? (1=obvious cause, 5=deep investigation needed)",
)

CODEBASE_SCOPE = FeatureDefinition(
    name="codebase_scope",
    min_value=1,
    max_value=5,
    description="How much of the codebase might be involved? (1=single file, 5=system-wide)",
)

INFORMATION_COMPLETENESS = FeatureDefinition(
    name="information_completeness",
    min_value=1,
    max_value=5,
    description="How complete is the information provided? (1=missing key info, 5=all context given)",
)

SIMILAR_ISSUE_LIKELIHOOD = FeatureDefinition(
    name="similar_issue_likelihood",
    min_value=0,
    max_value=1,
    description="Is this likely a common issue type with known solutions? (0=novel, 1=common pattern)",
)

BACKWARDS_COMPATIBILITY_RISK = FeatureDefinition(
    name="backwards_compatibility_risk",
    min_value=1,
    max_value=5,
    description="How much backwards compatibility consideration is needed? (1=none, 5=critical)",
)


EXTENDED_PROBLEM_FEATURES = [
    ERROR_SPECIFICITY,
    REPRODUCTION_CLARITY,
    EXPECTED_BEHAVIOR_CLARITY,
    DEBUGGING_COMPLEXITY,
    CODEBASE_SCOPE,
    INFORMATION_COMPLETENESS,
    SIMILAR_ISSUE_LIKELIHOOD,
    BACKWARDS_COMPATIBILITY_RISK,
]


# =============================================================================
# Prompt Template
# =============================================================================

SWEBENCH_PROBLEM_EXTENDED_PROMPT = """You are analyzing a SWE-bench coding task to predict its difficulty.
This is a BUG FIX task in a Python repository. You only have access to the problem statement.

CRITICAL: You MUST provide a value for EVERY feature listed below.
Do not skip any features. If uncertain, provide your best estimate.

## TASK INFORMATION

**Instance ID:** {instance_id}

**Problem Statement:**
{problem_statement}

## FEATURES TO EVALUATE

Analyze ONLY the problem statement to evaluate these 8 features.
Focus on what information is available and what would be needed to solve this.

### Error Specificity (error_specificity: 1-5)
How specific is the error or bug description?
- 1: Very vague symptoms, unclear what's actually broken
- 2: General description of misbehavior
- 3: Specific behavior described but no error details
- 4: Clear error description with some context
- 5: Exact error message, stack trace, or precise failure mode

### Reproduction Clarity (reproduction_clarity: 1-5)
How clear are the steps to reproduce the issue?
- 1: No reproduction steps, unclear how to trigger
- 2: Vague conditions mentioned
- 3: General scenario described
- 4: Clear steps but some setup unclear
- 5: Exact reproduction steps with code/commands provided

### Expected Behavior Clarity (expected_behavior_clarity: 1-5)
How clear is what the correct behavior should be?
- 1: Very ambiguous, multiple interpretations possible
- 2: General expectation but details unclear
- 3: Reasonably clear expected outcome
- 4: Clear expected behavior with examples
- 5: Precisely specified with exact expected output/behavior

### Debugging Complexity (debugging_complexity: 1-5)
Based on the problem description, how complex would root cause analysis be?
- 1: Obvious cause stated or implied
- 2: Straightforward to identify cause
- 3: Moderate investigation needed
- 4: Complex debugging likely required
- 5: Deep investigation into internals needed

### Codebase Scope (codebase_scope: 1-5)
How much of the codebase might need to be understood or modified?
- 1: Likely isolated to single file/function
- 2: Few related files
- 3: Multiple components involved
- 4: Cross-cutting concern affecting many areas
- 5: System-wide implications

### Information Completeness (information_completeness: 1-5)
How complete is the information provided in the problem statement?
- 1: Missing critical information, many unknowns
- 2: Key details missing
- 3: Adequate information but gaps exist
- 4: Good context provided
- 5: Comprehensive information including versions, configs, examples

### Similar Issue Likelihood (similar_issue_likelihood: 0/1)
Is this likely a common issue type that has known solutions?
- 0: Novel or unusual issue, unlikely to find similar cases
- 1: Common pattern (e.g., null check, encoding, off-by-one, race condition)

### Backwards Compatibility Risk (backwards_compatibility_risk: 1-5)
How much backwards compatibility consideration is needed?
- 1: No compatibility concerns (internal change)
- 2: Minor API implications
- 3: Some compatibility considerations
- 4: Significant API/behavior changes
- 5: Critical compatibility concerns, deprecation needed

## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "error_specificity": <1-5>,
    "reproduction_clarity": <1-5>,
    "expected_behavior_clarity": <1-5>,
    "debugging_complexity": <1-5>,
    "codebase_scope": <1-5>,
    "information_completeness": <1-5>,
    "similar_issue_likelihood": <0 or 1>,
    "backwards_compatibility_risk": <1-5>,
    "reasoning": "<2-3 sentence summary of the key factors>"
}}"""


def format_swebench_problem_extended_prompt(task: Dict[str, Any]) -> str:
    """Format the extended problem-only prompt."""
    return SWEBENCH_PROBLEM_EXTENDED_PROMPT.format(
        instance_id=task.get("instance_id", ""),
        problem_statement=task.get("problem_statement", ""),
    )


SWEBENCH_PROBLEM_EXTENDED_CONFIG = PromptConfig(
    name="swebench_problem_extended",
    features=EXTENDED_PROBLEM_FEATURES,
    prompt_template=SWEBENCH_PROBLEM_EXTENDED_PROMPT,
    task_id_field="instance_id",
    truncation_limits={},
    format_prompt_fn=format_swebench_problem_extended_prompt,
)
