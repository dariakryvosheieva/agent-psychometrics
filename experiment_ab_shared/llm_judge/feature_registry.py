"""Central registry of all LLM judge features.

Single source of truth for every feature used in Experiment A, across all
datasets and information levels. Each feature carries its own scale text
(rubric) and knows what task information it needs (InfoLevel).

Usage:
    from experiment_ab_shared.llm_judge.feature_registry import (
        ALL_FEATURES, get_features, get_features_by_level,
    )

    # Look up specific features
    feats = get_features(["solution_hint", "problem_clarity", "test_comprehensiveness"])

    # Get all features at a given info level
    problem_feats = get_features_by_level(InfoLevel.PROBLEM)
"""

from typing import Dict, List

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, InfoLevel


# =============================================================================
# Helper to build auditor-style scale text from a dict
# =============================================================================

def _auditor_scale(title: str, field_name: str, description: str, scale: Dict[int, str]) -> str:
    """Convert auditor-style {1: 'desc', ...} to markdown rubric text."""
    lines = [f"### {title} ({field_name}: 1-5)"]
    lines.append(description)
    for score, desc in sorted(scale.items()):
        lines.append(f"- {score}: {desc}")
    return "\n".join(lines)


# =============================================================================
# PROBLEM level features (15) — sees only problem statement
# =============================================================================

SOLUTION_HINT = FeatureDefinition(
    name="solution_hint",
    min_value=0,
    max_value=3,
    description="Does the task description hint at the solution? (0=none, 3=exact solution)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Solution Hint (solution_hint: 0-3)
Does the task description contain or hint at the solution approach?
- 0: No hint at the solution at all
- 1: Vague hint or general direction
- 2: Clear description of approach needed
- 3: Exact solution or detailed steps provided""",
    },
)

PROBLEM_CLARITY = FeatureDefinition(
    name="problem_clarity",
    min_value=1,
    max_value=5,
    description="How clear and well-specified is the task? (1=vague, 5=crystal clear)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Problem Clarity (problem_clarity: 1-5)
How clear and well-specified is the task?
- 1: Very vague, unclear what's actually required
- 2: Somewhat clear but missing key details
- 3: Reasonably clear, some ambiguity
- 4: Clear with good context
- 5: Crystal clear with explicit steps and expected behavior""",
    },
)

DOMAIN_KNOWLEDGE_REQUIRED = FeatureDefinition(
    name="domain_knowledge_required",
    min_value=1,
    max_value=5,
    description="How much specialized knowledge is needed? (1=basic, 5=expert)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic Python, obvious fix anyone could make
- 2: Standard library knowledge needed
- 3: Framework-specific knowledge (Django, pytest, numpy, etc.)
- 4: Deep understanding of the library's internals
- 5: Obscure APIs, protocols, or highly specialized domain knowledge""",
        "terminal": """### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic shell commands anyone could use (ls, cd, cat, echo)
- 2: Standard Unix tools (grep, sed, awk, find)
- 3: Specialized tools or configurations (cmake, git internals, network tools)
- 4: Deep understanding of systems (kernel, filesystems, protocols)
- 5: Obscure tools, APIs, or highly specialized domain knowledge""",
        "optimization": """### Domain Knowledge Required (domain_knowledge_required: 1-5)
How much specialized knowledge is needed?
- 1: Basic Python performance (list comprehensions, generators)
- 2: Standard library optimization patterns
- 3: Library-specific knowledge (numpy, pandas internals)
- 4: Deep understanding of library implementation
- 5: Expert knowledge (SIMD, memory layout, CPU caches)""",
    },
)

LOGICAL_REASONING_REQUIRED = FeatureDefinition(
    name="logical_reasoning_required",
    min_value=1,
    max_value=5,
    description="How much logical reasoning is needed? (1=mechanical, 5=deep reasoning)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Logical Reasoning Required (logical_reasoning_required: 1-5)
How much logical reasoning is needed?
- 1: Mechanical execution, no reasoning needed
- 2: Simple cause-effect reasoning
- 3: Multi-step reasoning required
- 4: Complex reasoning with multiple factors
- 5: Deep reasoning about edge cases, invariants, or system behavior""",
    },
)

ATYPICALITY = FeatureDefinition(
    name="atypicality",
    min_value=1,
    max_value=5,
    description="How unusual is this task pattern? (1=very common, 5=rare/novel)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Atypicality (atypicality: 1-5)
How unusual is this task pattern?
- 1: Very common pattern (well-known solution approach)
- 2: Common pattern in this domain
- 3: Moderately unusual
- 4: Unusual pattern
- 5: Rare or novel pattern""",
    },
)

VERIFICATION_DIFFICULTY = FeatureDefinition(
    name="verification_difficulty",
    min_value=1,
    max_value=5,
    description="How hard to verify the solution is correct? (1=trivial, 5=very hard)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Verification Difficulty (verification_difficulty: 1-5)
How hard is it to verify the solution is correct?
- 1: Trivial (obvious pass/fail)
- 2: Easy (straightforward test cases)
- 3: Moderate (some edge cases to consider)
- 4: Hard (subtle correctness issues, complex setup)
- 5: Very hard (rare edge cases, hard to reproduce, timing-sensitive)""",
    },
)

STANDARD_PATTERN_AVAILABLE = FeatureDefinition(
    name="standard_pattern_available",
    min_value=0,
    max_value=1,
    description="Is this a well-documented pattern? (0=novel solution needed, 1=known pattern)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Standard Pattern Available (standard_pattern_available: 0/1)
Is this a well-documented pattern with existing examples?
- 0: Novel solution needed, no clear pattern to follow
- 1: Well-documented pattern (e.g., common idiom, StackOverflow answer available)""",
    },
)

ERROR_SPECIFICITY = FeatureDefinition(
    name="error_specificity",
    min_value=1,
    max_value=5,
    description="How specific is the error/bug description? (1=vague symptoms, 5=exact error with stack trace)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Error Specificity (error_specificity: 1-5)
How specific is the error or bug description?
- 1: Very vague symptoms, unclear what's actually broken
- 2: General description of misbehavior
- 3: Specific behavior described but no error details
- 4: Clear error description with some context
- 5: Exact error message, stack trace, or precise failure mode""",
        "terminal": """### Error Specificity (error_specificity: 1-5)
How specific is the problem description?
- 1: Very vague goal, unclear what success looks like
- 2: General description of desired outcome
- 3: Specific outcome described but details missing
- 4: Clear description with context about expected behavior
- 5: Exact specification with precise success criteria""",
        "optimization": """### Error Specificity (error_specificity: 1-5)
How specific is the performance problem description?
- 1: Vague "make it faster" with no specifics
- 2: General description of slowness
- 3: Specific function/API identified as slow
- 4: Clear performance issue with some context
- 5: Exact bottleneck identified with profiling data or benchmarks""",
    },
)

REPRODUCTION_CLARITY = FeatureDefinition(
    name="reproduction_clarity",
    min_value=1,
    max_value=5,
    description="How clear are the reproduction steps? (1=unclear, 5=exact steps given)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Reproduction Clarity (reproduction_clarity: 1-5)
How clear are the steps to reproduce the issue?
- 1: No reproduction steps, unclear how to trigger
- 2: Vague conditions mentioned
- 3: General scenario described
- 4: Clear steps but some setup unclear
- 5: Exact reproduction steps with code/commands provided""",
        "terminal": """### Reproduction Clarity (reproduction_clarity: 1-5)
How clear are the steps to set up and attempt the task?
- 1: No setup steps, unclear how to begin
- 2: Vague environment requirements mentioned
- 3: General setup described
- 4: Clear steps but some prerequisites unclear
- 5: Exact setup and execution steps with commands provided""",
        "optimization": """### Reproduction Clarity (reproduction_clarity: 1-5)
How clear is the performance scenario to reproduce?
- 1: No benchmark or test scenario provided
- 2: Vague description of slow use case
- 3: General performance scenario described
- 4: Clear benchmark but some parameters unclear
- 5: Exact benchmark with input sizes and expected speedup""",
    },
)

EXPECTED_BEHAVIOR_CLARITY = FeatureDefinition(
    name="expected_behavior_clarity",
    min_value=1,
    max_value=5,
    description="How clear is what the correct behavior should be? (1=ambiguous, 5=precisely specified)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Expected Behavior Clarity (expected_behavior_clarity: 1-5)
How clear is what the correct behavior should be?
- 1: Very ambiguous, multiple interpretations possible
- 2: General expectation but details unclear
- 3: Reasonably clear expected outcome
- 4: Clear expected behavior with examples
- 5: Precisely specified with exact expected output/behavior""",
        "terminal": """### Expected Behavior Clarity (expected_behavior_clarity: 1-5)
How clear is what a successful outcome looks like?
- 1: Very ambiguous, multiple valid interpretations
- 2: General goal but details unclear
- 3: Reasonably clear target outcome
- 4: Clear success criteria with examples
- 5: Precisely specified with exact expected output/state""",
        "optimization": """### Expected Behavior Clarity (expected_behavior_clarity: 1-5)
How clear is the optimization target?
- 1: Very ambiguous, unclear what "faster" means in context
- 2: General speedup goal but no specifics
- 3: Target function/API clear but speedup threshold unclear
- 4: Clear optimization target with approximate goals
- 5: Precisely specified with exact performance requirements""",
    },
)

DEBUGGING_COMPLEXITY = FeatureDefinition(
    name="debugging_complexity",
    min_value=1,
    max_value=5,
    description="How complex would debugging/root cause analysis be? (1=obvious cause, 5=deep investigation needed)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Debugging Complexity (debugging_complexity: 1-5)
Based on the problem description, how complex would root cause analysis be?
- 1: Obvious cause stated or implied
- 2: Straightforward to identify cause
- 3: Moderate investigation needed
- 4: Complex debugging likely required
- 5: Deep investigation into internals needed""",
        "terminal": """### Debugging Complexity (debugging_complexity: 1-5)
How complex would it be to figure out the right approach?
- 1: Obvious approach stated or implied
- 2: Straightforward to determine approach
- 3: Moderate exploration/research needed
- 4: Complex problem-solving likely required
- 5: Deep investigation into tools/systems needed""",
        "optimization": """### Debugging Complexity (debugging_complexity: 1-5)
How complex would profiling and identifying the bottleneck be?
- 1: Obvious bottleneck stated or implied
- 2: Straightforward to profile and identify
- 3: Moderate profiling/analysis needed
- 4: Complex performance analysis likely required
- 5: Deep investigation into runtime behavior needed""",
    },
)

CODEBASE_SCOPE = FeatureDefinition(
    name="codebase_scope",
    min_value=1,
    max_value=5,
    description="How much of the codebase might be involved? (1=single file, 5=system-wide)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Codebase Scope (codebase_scope: 1-5)
How much of the codebase might need to be understood or modified?
- 1: Likely isolated to single file/function
- 2: Few related files
- 3: Multiple components involved
- 4: Cross-cutting concern affecting many areas
- 5: System-wide implications""",
    },
)

INFORMATION_COMPLETENESS = FeatureDefinition(
    name="information_completeness",
    min_value=1,
    max_value=5,
    description="How complete is the information provided? (1=missing key info, 5=all context given)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "default": """### Information Completeness (information_completeness: 1-5)
How complete is the information provided in the problem statement?
- 1: Missing critical information, many unknowns
- 2: Key details missing
- 3: Adequate information but gaps exist
- 4: Good context provided
- 5: Comprehensive information including versions, configs, examples""",
    },
)

SIMILAR_ISSUE_LIKELIHOOD = FeatureDefinition(
    name="similar_issue_likelihood",
    min_value=0,
    max_value=1,
    description="Is this likely a common issue type with known solutions? (0=novel, 1=common pattern)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Similar Issue Likelihood (similar_issue_likelihood: 0/1)
Is this likely a common bug type that has known solutions?
- 0: Novel or unusual issue, unlikely to find similar cases
- 1: Common bug pattern (e.g., null check, encoding, off-by-one, race condition)""",
        "terminal": """### Similar Issue Likelihood (similar_issue_likelihood: 0/1)
Is this likely a common shell/automation task with known solutions?
- 0: Novel or unusual task, unlikely to find similar examples online
- 1: Common pattern (e.g., file processing, service configuration, text extraction)""",
        "optimization": """### Similar Issue Likelihood (similar_issue_likelihood: 0/1)
Is this likely a common optimization pattern with known approaches?
- 0: Novel bottleneck requiring creative optimization strategy
- 1: Common optimization pattern (e.g., vectorization, caching, batch processing, algorithmic improvement)""",
    },
)

SIDE_EFFECT_RISK = FeatureDefinition(
    name="side_effect_risk",
    min_value=1,
    max_value=5,
    description="How likely are unintended side effects from changes? (1=none, 5=critical risk)",
    info_level=InfoLevel.PROBLEM,
    scale_text={
        "code": """### Side Effect Risk (side_effect_risk: 1-5)
How likely are unintended side effects from the fix?
- 1: No risk (internal change, no API implications)
- 2: Minor API implications
- 3: Some compatibility or behavioral side effects possible
- 4: Significant API/behavior changes likely
- 5: Critical risk — backwards compatibility, deprecation, or wide-reaching behavioral changes""",
        "terminal": """### Side Effect Risk (side_effect_risk: 1-5)
How likely are unintended side effects from the solution?
- 1: No risk (self-contained operation, no system state changes)
- 2: Minor filesystem or config changes
- 3: Some risk of affecting other services or system state
- 4: Significant risk of breaking other processes or configurations
- 5: Critical risk — system-wide changes, network/security implications""",
        "optimization": """### Side Effect Risk (side_effect_risk: 1-5)
How likely is the optimization to introduce correctness regressions?
- 1: No risk (simple speedup, identical behavior guaranteed)
- 2: Minor numerical precision differences possible
- 3: Some edge cases might behave differently
- 4: Significant behavioral changes in corner cases likely
- 5: Critical risk — optimization fundamentally changes semantics or data flow""",
    },
)


# =============================================================================
# ENVIRONMENT level features (8) — problem + shell exploration, NO tests/solution
# Extracted via auditor agent (AWS/Inspect/Docker pipeline)
# =============================================================================

FIX_LOCALIZATION = FeatureDefinition(
    name="fix_localization",
    min_value=1,
    max_value=5,
    description="How spread out is the likely solution? (1=many modules, 5=single function)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Fix Localization", "fix_localization",
            "How spread out is the likely solution?",
            {
                1: "Solution requires changes across many modules/packages",
                2: "Solution spans multiple files across different directories",
                3: "Solution spans 2-3 files in the same module",
                4: "Solution is in 1-2 closely related files",
                5: "Solution is contained to a single function/method",
            },
        ),
    },
)

ENTRY_POINT_CLARITY = FeatureDefinition(
    name="entry_point_clarity",
    min_value=1,
    max_value=5,
    description="How easy is it to find where the problem manifests? (1=no clear entry point, 5=obvious)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Entry Point Clarity", "entry_point_clarity",
            "How easy is it to find where the problem manifests?",
            {
                1: "No clear entry point, requires deep architecture knowledge",
                2: "Entry point exists but buried in abstraction layers",
                3: "Entry point findable with moderate searching",
                4: "Problem statement or tests hint at the location",
                5: "Clear from problem statement exactly which file/function",
            },
        ),
    },
)

CHANGE_BLAST_RADIUS = FeatureDefinition(
    name="change_blast_radius",
    min_value=1,
    max_value=5,
    description="How many components would be affected by changes? (1=isolated, 5=entire codebase)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Change Blast Radius", "change_blast_radius",
            "How many components would be affected by changes? (Higher = harder)",
            {
                1: "Isolated change, no downstream effects",
                2: "Minor coupling, 1-2 related files to consider",
                3: "Moderate coupling, changes affect a subsystem",
                4: "High coupling, changes ripple across modules",
                5: "Core/shared code, changes affect entire codebase",
            },
        ),
    },
)

ENVIRONMENT_SETUP_COMPLEXITY = FeatureDefinition(
    name="environment_setup_complexity",
    min_value=1,
    max_value=5,
    description="How complex is the runtime/tooling environment? (1=ready to run, 5=exotic setup)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Environment Setup Complexity", "environment_setup_complexity",
            "How complex is the runtime/tooling environment?",
            {
                1: "Standard single-directory project, ready to run out of the box",
                2: "Minor configuration needed, clear project structure",
                3: "Multiple services or components, custom configurations",
                4: "Complex orchestration, specialized dependencies, non-trivial build steps",
                5: "Exotic environment, multi-container setup, hardware-specific requirements",
            },
        ),
    },
)

IMPLEMENTATION_LANGUAGE_COMPLEXITY = FeatureDefinition(
    name="implementation_language_complexity",
    min_value=1,
    max_value=5,
    description="How complex is the primary language/tech stack? (1=pure Python/shell, 5=multi-language)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Implementation Language Complexity", "implementation_language_complexity",
            "How complex is the primary language/tech stack for the solution?",
            {
                1: "Pure Python or simple shell commands",
                2: "Python with standard libraries, basic scripting",
                3: "Mixed languages (Python + build tools), moderately complex shell",
                4: "Compiled languages (C/C++), complex build systems, framework-specific patterns",
                5: "Multi-language (C/Rust + Python bindings), SIMD/assembly, exotic toolchains",
            },
        ),
    },
)

TESTING_INFRASTRUCTURE_QUALITY = FeatureDefinition(
    name="testing_infrastructure_quality",
    min_value=1,
    max_value=5,
    description="How good is the testing/validation setup? (1=no tests, 5=comprehensive suite)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Testing Infrastructure Quality", "testing_infrastructure_quality",
            "How good is the testing/validation setup for verifying a solution?",
            {
                1: "No test framework, no way to validate changes",
                2: "Basic tests exist but hard to run or incomplete",
                3: "Standard test framework, moderate coverage",
                4: "Good test coverage, easy to run tests, clear pass/fail signals",
                5: "Comprehensive test suite, fast feedback loops, detailed error messages",
            },
        ),
    },
)

DEPENDENCY_COMPLEXITY = FeatureDefinition(
    name="dependency_complexity",
    min_value=1,
    max_value=5,
    description="How complex are the project dependencies? (1=standard library only, 5=complex tree)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Dependency Complexity", "dependency_complexity",
            "How complex are the project dependencies?",
            {
                1: "No external dependencies, standard library only",
                2: "Few well-known dependencies (e.g., requests, numpy)",
                3: "Moderate number of standard packages",
                4: "Many dependencies, some specialized or version-sensitive",
                5: "Complex dependency tree, C extensions, system-level deps, version conflicts",
            },
        ),
    },
)

CODEBASE_SCALE = FeatureDefinition(
    name="codebase_scale",
    min_value=1,
    max_value=5,
    description="How large/complex is the codebase? (1=tiny, 5=massive)",
    info_level=InfoLevel.ENVIRONMENT,
    scale_text={
        "default": _auditor_scale(
            "Codebase Scale", "codebase_scale",
            "How large/complex is the codebase the agent needs to work with?",
            {
                1: "Tiny project (<100 files, <5K lines)",
                2: "Small project (100-500 files)",
                3: "Medium project (500-2000 files)",
                4: "Large project (2000-10000 files)",
                5: "Massive project (10000+ files, complex module structure)",
            },
        ),
    },
)


# =============================================================================
# TEST level features (3) — problem + test/evaluation artifact, NO solution
# =============================================================================

TEST_COMPREHENSIVENESS = FeatureDefinition(
    name="test_comprehensiveness",
    min_value=1,
    max_value=5,
    description="How thoroughly does the test cover expected behavior? (1=minimal, 5=exhaustive)",
    info_level=InfoLevel.TEST,
    scale_text={
        "code": """### Test Comprehensiveness (test_comprehensiveness: 1-5)
How thoroughly does the test patch cover the expected behavior?
- 1: Minimal - tests only one basic case
- 2: Limited - tests a few cases but misses important scenarios
- 3: Moderate - covers main functionality with some gaps
- 4: Good - covers most expected behaviors and variations
- 5: Exhaustive - comprehensive coverage including corner cases""",
        "terminal": """### Test Comprehensiveness (test_comprehensiveness: 1-5)
How thoroughly does the evaluation harness cover the expected behavior?
- 1: Minimal - checks only one basic output
- 2: Limited - checks a few conditions but misses important scenarios
- 3: Moderate - covers main success criteria with some gaps
- 4: Good - covers most expected outcomes and edge cases
- 5: Exhaustive - comprehensive coverage including corner cases and error handling""",
        "optimization": """### Test Comprehensiveness (test_comprehensiveness: 1-5)
How thoroughly does the benchmark cover the optimization scenario?
- 1: Minimal - tests only one basic case with trivial input
- 2: Limited - tests a few input sizes but misses important scenarios
- 3: Moderate - covers main use case with some size variations
- 4: Good - covers multiple input sizes, shapes, and data types
- 5: Exhaustive - comprehensive coverage including edge cases and realistic workloads""",
    },
)

TEST_ASSERTION_COMPLEXITY = FeatureDefinition(
    name="test_assertion_complexity",
    min_value=1,
    max_value=5,
    description="How complex are the test assertions? (1=simple equality, 5=complex logic/mocking)",
    info_level=InfoLevel.TEST,
    scale_text={
        "code": """### Test Assertion Complexity (test_assertion_complexity: 1-5)
How complex are the assertions and test setup in the test patch?
- 1: Simple - basic equality checks (assertEqual, assertTrue)
- 2: Standard - uses common assertion patterns
- 3: Moderate - multiple assertions, some setup required
- 4: Complex - requires mocking, fixtures, or intricate setup
- 5: Very complex - extensive mocking, async testing, or multi-step verification""",
        "terminal": """### Test Assertion Complexity (test_assertion_complexity: 1-5)
How complex is the evaluation logic in the test harness?
- 1: Simple - basic file existence or string match check
- 2: Standard - checks output format or simple numeric comparison
- 3: Moderate - multiple checks, some parsing of output required
- 4: Complex - statistical validation, multi-step verification, or custom scoring
- 5: Very complex - cross-referencing multiple outputs, timing-sensitive checks""",
        "optimization": """### Test Assertion Complexity (test_assertion_complexity: 1-5)
How complex is the correctness verification in the benchmark?
- 1: Simple - basic equality check (reference.equals(current))
- 2: Standard - checks a few output fields individually
- 3: Moderate - multiple assertions, type/shape checking, tolerance-based comparison
- 4: Complex - custom equivalence logic, statistical validation, or multi-step verification
- 5: Very complex - domain-specific correctness checks, numerical stability verification""",
    },
)

TEST_EDGE_CASE_COVERAGE = FeatureDefinition(
    name="test_edge_case_coverage",
    min_value=1,
    max_value=5,
    description="Does the test cover edge cases and boundary conditions? (1=happy path only, 5=thorough)",
    info_level=InfoLevel.TEST,
    scale_text={
        "code": """### Test Edge Case Coverage (test_edge_case_coverage: 1-5)
Does the test cover edge cases, boundary conditions, and error scenarios?
- 1: Happy path only - no edge cases tested
- 2: Minimal - one or two edge cases
- 3: Moderate - some boundary conditions checked
- 4: Good - most edge cases and error conditions tested
- 5: Thorough - comprehensive edge case and error handling coverage""",
        "terminal": """### Test Edge Case Coverage (test_edge_case_coverage: 1-5)
Does the evaluation cover edge cases, boundary conditions, and error scenarios?
- 1: Happy path only - no edge cases tested
- 2: Minimal - one or two edge cases
- 3: Moderate - some boundary conditions checked
- 4: Good - most edge cases and failure modes tested
- 5: Thorough - comprehensive edge case, error handling, and adversarial input coverage""",
        "optimization": """### Test Edge Case Coverage (test_edge_case_coverage: 1-5)
Does the benchmark test data include edge cases and degenerate inputs?
- 1: Happy path only - typical input sizes only
- 2: Minimal - one or two boundary sizes
- 3: Moderate - some degenerate inputs (empty, very large)
- 4: Good - includes unusual shapes, dtypes, memory layouts, special values
- 5: Thorough - comprehensive degenerate cases, adversarial inputs, and stress tests""",
    },
)


# =============================================================================
# SOLUTION level features (2) — problem + tests + gold solution
# =============================================================================

SOLUTION_COMPLEXITY = FeatureDefinition(
    name="solution_complexity",
    min_value=1,
    max_value=5,
    description="How complex is the actual solution? (1=trivial, 5=very complex)",
    info_level=InfoLevel.SOLUTION,
    scale_text={
        "code": """### Solution Complexity (solution_complexity: 1-5)
How complex is the actual code change?
- 1: Trivial (add parameter, change value, simple one-liner)
- 2: Simple (straightforward logic change)
- 3: Moderate (requires understanding context, multiple changes)
- 4: Complex (algorithmic changes, multiple interdependent fixes)
- 5: Very complex (architectural changes, subtle edge cases)""",
        "terminal": """### Solution Complexity (solution_complexity: 1-5)
How complex is the actual solution?
- 1: Trivial (single command, simple file operation)
- 2: Simple (straightforward multi-step task)
- 3: Moderate (requires understanding context, multiple tools)
- 4: Complex (multiple interdependent steps, debugging needed)
- 5: Very complex (multi-stage pipeline, cross-system integration)""",
        "optimization": """### Solution Complexity (solution_complexity: 1-5)
How complex is the optimization logic?
- 1: Simple (add caching, use built-in function)
- 2: Standard (vectorization, batch processing)
- 3: Moderate (algorithm improvements, memory optimization)
- 4: Complex (significant algorithmic changes)
- 5: Very complex (architectural redesign, low-level optimization)""",
    },
)

INTEGRATION_COMPLEXITY = FeatureDefinition(
    name="integration_complexity",
    min_value=1,
    max_value=5,
    description="How tightly must changes integrate with existing code? (1=isolated, 5=system-wide)",
    info_level=InfoLevel.SOLUTION,
    scale_text={
        "code": """### Integration Complexity (integration_complexity: 1-5)
How tightly must the changes integrate with existing code?
- 1: Self-contained/greenfield - new code with clear boundaries
- 2: Simple extension - adds to existing code with clear interface
- 3: Moderate integration - changes interact with several existing components
- 4: Deep integration - requires understanding multiple subsystems
- 5: Pervasive integration - affects system-wide behavior, many touchpoints""",
        "terminal": """### Integration Complexity (integration_complexity: 1-5)
How complex is the tooling/environment integration?
- 1: No special tools needed (basic shell)
- 2: Standard development tools (git, make, pip)
- 3: Multiple specialized tools or complex configuration
- 4: Uncommon tools or complex build systems
- 5: Exotic toolchain, legacy systems, or cross-compilation""",
        "optimization": """### Integration Complexity (integration_complexity: 1-5)
How tightly must the optimization integrate with existing code?
- 1: Self-contained optimization with clear boundaries
- 2: Simple drop-in replacement for existing function
- 3: Moderate integration - optimization touches several components
- 4: Deep integration - requires understanding data flow across subsystems
- 5: Pervasive changes - optimization affects system-wide architecture""",
    },
)


# =============================================================================
# Feature collections and lookup
# =============================================================================

# Ordered list of all features
_ALL_FEATURES_LIST: list[FeatureDefinition] = [
    # Problem level (15)
    SOLUTION_HINT,
    PROBLEM_CLARITY,
    DOMAIN_KNOWLEDGE_REQUIRED,
    LOGICAL_REASONING_REQUIRED,
    ATYPICALITY,
    VERIFICATION_DIFFICULTY,
    STANDARD_PATTERN_AVAILABLE,
    ERROR_SPECIFICITY,
    REPRODUCTION_CLARITY,
    EXPECTED_BEHAVIOR_CLARITY,
    DEBUGGING_COMPLEXITY,
    CODEBASE_SCOPE,
    INFORMATION_COMPLETENESS,
    SIMILAR_ISSUE_LIKELIHOOD,
    SIDE_EFFECT_RISK,
    # Environment level (8)
    FIX_LOCALIZATION,
    ENTRY_POINT_CLARITY,
    CHANGE_BLAST_RADIUS,
    ENVIRONMENT_SETUP_COMPLEXITY,
    IMPLEMENTATION_LANGUAGE_COMPLEXITY,
    TESTING_INFRASTRUCTURE_QUALITY,
    DEPENDENCY_COMPLEXITY,
    CODEBASE_SCALE,
    # Test level (3)
    TEST_COMPREHENSIVENESS,
    TEST_ASSERTION_COMPLEXITY,
    TEST_EDGE_CASE_COVERAGE,
    # Solution level (2)
    SOLUTION_COMPLEXITY,
    INTEGRATION_COMPLEXITY,
]

ALL_FEATURES: Dict[str, FeatureDefinition] = {f.name: f for f in _ALL_FEATURES_LIST}


def get_features(names: list[str]) -> list[FeatureDefinition]:
    """Look up features by name, preserving order.

    Raises:
        KeyError: If any name is not in the registry.
    """
    result = []
    for name in names:
        if name not in ALL_FEATURES:
            raise KeyError(
                f"Unknown feature '{name}'. "
                f"Available: {sorted(ALL_FEATURES.keys())}"
            )
        result.append(ALL_FEATURES[name])
    return result


def get_features_by_level(level: InfoLevel) -> list[FeatureDefinition]:
    """Return all features at a given info level, in registry order."""
    return [f for f in _ALL_FEATURES_LIST if f.info_level == level]


def get_all_feature_names() -> list[str]:
    """Return all feature names in registry order."""
    return [f.name for f in _ALL_FEATURES_LIST]
