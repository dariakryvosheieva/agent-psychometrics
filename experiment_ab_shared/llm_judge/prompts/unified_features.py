"""Unified feature definitions for LLM judge prompts.

This module defines the standardized set of features used across all datasets
in Experiment A. The goal is to enable fair comparison across:
- SWE-bench Verified (bug fixes)
- SWE-bench Pro (bug fixes)
- TerminalBench (terminal/shell tasks)
- GSO (performance optimization)

Features are divided into:
1. CORE_FEATURES: 8 features that apply to ALL datasets
2. Dataset-specific features: integration_complexity (code) or tooling_complexity (terminal)
"""

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition


# =============================================================================
# Core Features (ALL 4 datasets)
# =============================================================================

SOLUTION_HINT = FeatureDefinition(
    name="solution_hint",
    min_value=0,
    max_value=3,
    description="Does the task description hint at the solution? (0=none, 3=exact solution)",
)

PROBLEM_CLARITY = FeatureDefinition(
    name="problem_clarity",
    min_value=1,
    max_value=5,
    description="How clear and well-specified is the task? (1=vague, 5=crystal clear)",
)

SOLUTION_COMPLEXITY = FeatureDefinition(
    name="solution_complexity",
    min_value=1,
    max_value=5,
    description="How complex is the actual solution? (1=trivial, 5=very complex)",
)

DOMAIN_KNOWLEDGE_REQUIRED = FeatureDefinition(
    name="domain_knowledge_required",
    min_value=1,
    max_value=5,
    description="How much specialized knowledge is needed? (1=basic, 5=expert)",
)

LOGICAL_REASONING_REQUIRED = FeatureDefinition(
    name="logical_reasoning_required",
    min_value=1,
    max_value=5,
    description="How much logical reasoning is needed? (1=mechanical, 5=deep reasoning)",
)

ATYPICALITY = FeatureDefinition(
    name="atypicality",
    min_value=1,
    max_value=5,
    description="How unusual is this task pattern? (1=very common, 5=rare/novel)",
)

VERIFICATION_DIFFICULTY = FeatureDefinition(
    name="verification_difficulty",
    min_value=1,
    max_value=5,
    description="How hard to verify the solution is correct? (1=trivial, 5=very hard)",
)

STANDARD_PATTERN_AVAILABLE = FeatureDefinition(
    name="standard_pattern_available",
    min_value=0,
    max_value=1,
    description="Is this a well-documented pattern? (0=novel solution needed, 1=known pattern)",
)

# List of all 8 core features (same for all datasets)
CORE_FEATURES = [
    SOLUTION_HINT,
    PROBLEM_CLARITY,
    SOLUTION_COMPLEXITY,
    DOMAIN_KNOWLEDGE_REQUIRED,
    LOGICAL_REASONING_REQUIRED,
    ATYPICALITY,
    VERIFICATION_DIFFICULTY,
    STANDARD_PATTERN_AVAILABLE,
]

# Feature names for validation
CORE_FEATURE_NAMES = [f.name for f in CORE_FEATURES]


# =============================================================================
# Dataset-Specific Features
# =============================================================================

# For code-based datasets (SWE-bench, SWE-bench Pro, GSO)
INTEGRATION_COMPLEXITY = FeatureDefinition(
    name="integration_complexity",
    min_value=1,
    max_value=5,
    description="How tightly integrated with existing code? (1=isolated, 5=system-wide)",
)

# For terminal-based datasets (TerminalBench)
TOOLING_COMPLEXITY = FeatureDefinition(
    name="tooling_complexity",
    min_value=1,
    max_value=5,
    description="How complex is the tooling/environment? (1=basic shell, 5=exotic toolchain)",
)


# =============================================================================
# Combined Feature Sets
# =============================================================================

# For SWE-bench, SWE-bench Pro, GSO (9 features)
CODE_DATASET_FEATURES = CORE_FEATURES + [INTEGRATION_COMPLEXITY]
CODE_DATASET_FEATURE_NAMES = [f.name for f in CODE_DATASET_FEATURES]

# For TerminalBench (9 features)
TERMINAL_DATASET_FEATURES = CORE_FEATURES + [TOOLING_COMPLEXITY]
TERMINAL_DATASET_FEATURE_NAMES = [f.name for f in TERMINAL_DATASET_FEATURES]


# =============================================================================
# Shared Scale Descriptions (can be customized per dataset)
# =============================================================================

SOLUTION_HINT_SCALE = """### Solution Hint (solution_hint: 0-3)
Does the task description contain or hint at the solution approach?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of approach needed
- 3: Exact solution or detailed steps provided"""

PROBLEM_CLARITY_SCALE = """### Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the task?
- 1: Very vague, unclear what's actually required
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with explicit steps and expected behavior"""

SOLUTION_COMPLEXITY_SCALE_CODE = """### Solution Complexity (solution_complexity: 1-5)
How complex is the actual code change?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases)"""

SOLUTION_COMPLEXITY_SCALE_TERMINAL = """### Solution Complexity (solution_complexity: 1-5)
How complex is the actual solution?
- 1: Trivial (single command, simple file operation)
- 2: Simple (straightforward multi-step task)
- 3: Moderate (requires understanding context, multiple tools)
- 4: Complex (multiple interdependent steps, debugging needed)
- 5: Very complex (multi-stage pipeline, cross-system integration)"""

SOLUTION_COMPLEXITY_SCALE_OPTIMIZATION = """### Solution Complexity (solution_complexity: 1-5)
How complex is the optimization logic?
- 1: Simple (add caching, use built-in function)
- 2: Standard (vectorization, batch processing)
- 3: Moderate (algorithm improvements, memory optimization)
- 4: Complex (significant algorithmic changes)
- 5: Very complex (architectural redesign, low-level optimization)"""

DOMAIN_KNOWLEDGE_SCALE_CODE = """### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge"""

DOMAIN_KNOWLEDGE_SCALE_TERMINAL = """### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic shell commands anyone could use (ls, cd, cat, echo)
- 2: Standard Unix tools (grep, sed, awk, find)
- 3: Specialized tools or configurations (cmake, git internals, network tools)
- 4: Deep understanding of systems (kernel, filesystems, protocols)
- 5: Obscure tools, APIs, or highly specialized domain knowledge"""

DOMAIN_KNOWLEDGE_SCALE_OPTIMIZATION = """### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic Python performance (list comprehensions, generators)
- 2: Standard library optimization patterns
- 3: Library-specific knowledge (numpy, pandas internals)
- 4: Deep understanding of library implementation
- 5: Expert knowledge (SIMD, memory layout, CPU caches)"""

LOGICAL_REASONING_SCALE = """### Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed?
- 1: Mechanical execution, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior"""

ATYPICALITY_SCALE = """### Atypicality (atypicality: 1-5)
How unusual is this task pattern?
- 1: Very common pattern (well-known solution approach)
- 2: Common pattern in this domain
- 3: Moderately unusual
- 4: Unusual pattern
- 5: Rare or novel pattern"""

VERIFICATION_DIFFICULTY_SCALE = """### Verification Difficulty (verification_difficulty: 1-5)
How hard is it to verify the solution is correct?
- 1: Trivial (obvious pass/fail)
- 2: Easy (straightforward test cases)
- 3: Moderate (some edge cases to consider)
- 4: Hard (subtle correctness issues, complex setup)
- 5: Very hard (rare edge cases, hard to reproduce, timing-sensitive)"""

STANDARD_PATTERN_SCALE = """### Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented pattern with existing examples?
- 0: Novel solution needed, no clear pattern to follow
- 1: Well-documented pattern (e.g., common idiom, StackOverflow answer available)"""

INTEGRATION_COMPLEXITY_SCALE = """### Integration Complexity (integration_complexity: 1-5)
How tightly must the changes integrate with existing code?
- 1: Self-contained/greenfield - new code with clear boundaries
- 2: Simple extension - adds to existing code with clear interface
- 3: Moderate integration - changes interact with several existing components
- 4: Deep integration - requires understanding multiple subsystems
- 5: Pervasive integration - affects system-wide behavior, many touchpoints"""

TOOLING_COMPLEXITY_SCALE = """### Tooling Complexity (tooling_complexity: 1-5)
How complex is the tooling/environment setup?
- 1: No special tools needed (basic shell)
- 2: Standard development tools (git, make, pip)
- 3: Multiple specialized tools or complex configuration
- 4: Uncommon tools or complex build systems
- 5: Exotic toolchain, legacy systems, or cross-compilation"""


# =============================================================================
# Prompt Components
# =============================================================================

COMPLETENESS_INSTRUCTION = """
CRITICAL: You MUST provide a value for EVERY feature listed below.
Do not skip any features. If uncertain, provide your best estimate.
Missing values will cause extraction to fail.
"""

OUTPUT_FORMAT_9_FEATURES_CODE = """## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "solution_hint": <0-3>,
    "problem_clarity": <1-5>,
    "solution_complexity": <1-5>,
    "domain_knowledge_required": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "integration_complexity": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}"""

OUTPUT_FORMAT_9_FEATURES_TERMINAL = """## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "solution_hint": <0-3>,
    "problem_clarity": <1-5>,
    "solution_complexity": <1-5>,
    "domain_knowledge_required": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "tooling_complexity": <1-5>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}"""


# =============================================================================
# No-Solution Features (for ablation study)
# =============================================================================

# 7 features: CORE_FEATURES minus solution_complexity
# (solution_complexity and dataset-specific features require the gold patch)
NO_SOLUTION_FEATURES = [f for f in CORE_FEATURES if f.name != "solution_complexity"]
NO_SOLUTION_FEATURE_NAMES = [f.name for f in NO_SOLUTION_FEATURES]

OUTPUT_FORMAT_7_FEATURES = """## OUTPUT FORMAT

Respond with ONLY a JSON object. No markdown, no extra text.

{{
    "solution_hint": <0-3>,
    "problem_clarity": <1-5>,
    "domain_knowledge_required": <1-5>,
    "logical_reasoning_required": <1-5>,
    "atypicality": <1-5>,
    "verification_difficulty": <1-5>,
    "standard_pattern_available": <0 or 1>,
    "reasoning": "<2-3 sentence summary of the key difficulty factors>"
}}"""
