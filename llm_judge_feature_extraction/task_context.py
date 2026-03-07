"""Task context definitions for batched feature extraction.

Each TaskContext defines how to render task information at each InfoLevel
for a specific dataset. The formatters produce the cacheable prefix of the
prompt — everything except the feature scale descriptions and output format.

Usage:
    from llm_judge_feature_extraction.task_context import get_task_context

    ctx = get_task_context("swebench_verified")
    prefix = ctx.build_prefix(task, InfoLevel.PROBLEM)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llm_judge_feature_extraction.feature_registry import get_features_by_level
from llm_judge_feature_extraction.prompt_config import InfoLevel


COMPLETENESS_INSTRUCTION = (
    "CRITICAL: You MUST provide a value for EVERY feature listed below. "
    "Do not skip any features. If uncertain, provide your best estimate. "
    "Missing values will cause extraction to fail."
)


# =============================================================================
# Field access helpers
# =============================================================================

def _require(task: Dict[str, Any], field: str, dataset: str) -> Any:
    """Get a required field from a task dict, raising if missing or empty."""
    value = task.get(field)
    if value is None:
        raise ValueError(
            f"Task is missing required field '{field}' for dataset '{dataset}'. "
            f"Available fields: {sorted(task.keys())}"
        )
    if isinstance(value, str) and not value.strip():
        raise ValueError(
            f"Task field '{field}' is empty for dataset '{dataset}', "
            f"task_id={task.get('instance_id', task.get('task_id', '?'))}"
        )
    return value


def _optional(task: Dict[str, Any], field: str, default: str = "") -> str:
    """Get an optional field from a task dict."""
    value = task.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


# Maximum chars for fields that can blow up the prompt.
# Budget: ~400K chars ≈ 100K tokens, well within 90% of 128K context.
_MAX_PATCH_CHARS = 300_000
_MAX_TEST_PATCH_CHARS = 200_000
_MAX_PASS_TO_PASS_CHARS = 50_000


def _truncate(text: str, max_chars: int, label: str = "field") -> str:
    """Truncate text to max_chars, appending a notice if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... {label} truncated from {len(text):,} to {max_chars:,} chars ...]"


class _TaskFields:
    """Lazy field accessor that validates required fields on access."""

    def __init__(self, task: Dict[str, Any], dataset: str):
        self._task = task
        self._dataset = dataset

    def require(self, field: str) -> Any:
        return _require(self._task, field, self._dataset)

    def optional(self, field: str, default: str = "") -> str:
        return _optional(self._task, field, default)

    def optional_list(self, field: str) -> List[str]:
        return self._task.get(field) or []


# =============================================================================
# TaskContext
# =============================================================================

@dataclass
class TaskContext:
    """Dataset-specific context for prompt construction.

    Attributes:
        name: Dataset identifier (e.g., "swebench_verified", "gso")
        task_id_field: Field name for task IDs (e.g., "instance_id", "task_id")
        scale_variant: Which scale text variant to use ("code", "terminal", "optimization")
        system_intros: Per-level intro text (first line of prompt)
        format_task_info_fns: Per-level functions that render the ## TASK INFORMATION block
    """

    name: str
    task_id_field: str
    scale_variant: str
    system_intros: Dict[InfoLevel, str]
    format_task_info_fns: Dict[InfoLevel, Callable[[Dict[str, Any]], str]]

    def build_prefix(self, task: Dict[str, Any], level: InfoLevel) -> str:
        """Build the cacheable prefix for a given task and info level.

        Returns the system intro + completeness instruction + task info block.
        This prefix is shared across all feature batches at the same level.

        Raises:
            KeyError: If the info level is not supported for this dataset.
            ValueError: If required task fields are missing.
        """
        if level not in self.system_intros:
            raise KeyError(
                f"Info level {level.value} not supported for dataset '{self.name}'. "
                f"Available: {[l.value for l in self.system_intros]}"
            )
        intro = self.system_intros[level]
        task_info = self.format_task_info_fns[level](task)
        return f"{intro}\n\n{COMPLETENESS_INSTRUCTION}\n\n{task_info}"

    def get_task_id(self, task: Dict[str, Any]) -> str:
        """Extract task ID from a task dict."""
        return _require(task, self.task_id_field, self.name)


# =============================================================================
# SWE-bench formatter factory (shared by Verified and Pro)
# =============================================================================

def _make_swebench_formatters(dataset: str) -> Dict[InfoLevel, Callable]:
    """Create SWE-bench task info formatters for a given dataset name."""

    def problem_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, dataset)
        return f"""## TASK INFORMATION

**Instance ID:** {f.require("instance_id")}

**Problem Statement:**
{f.require("problem_statement")}"""

    def test_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, dataset)
        hints = f.optional("hints_text")
        hints_section = f"\n**Hints:**\n{hints}" if hints else ""
        # SWE-bench Pro uses lowercase field names in some sources
        fail_to_pass = task.get("fail_to_pass") or f.require("FAIL_TO_PASS")
        return f"""## TASK INFORMATION

**Instance ID:** {f.require("instance_id")}
**Repository:** {f.require("repo")}
**Version:** {f.optional("version", "unknown")}

**Problem Statement:**
{f.require("problem_statement")}

**Test Patch (tests that verify the fix):**
```diff
{_truncate(str(f.require("test_patch")), _MAX_TEST_PATCH_CHARS, "test_patch")}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{fail_to_pass}
{hints_section}"""

    def solution_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, dataset)
        hints = f.optional("hints_text")
        hints_section = f"\n**Hints:**\n{hints}" if hints else ""
        fail_to_pass = task.get("fail_to_pass") or f.require("FAIL_TO_PASS")
        pass_to_pass = task.get("pass_to_pass") or f.optional("PASS_TO_PASS", "[]")
        patch = _truncate(str(f.require("patch")), _MAX_PATCH_CHARS, "patch")
        pass_to_pass = _truncate(str(pass_to_pass), _MAX_PASS_TO_PASS_CHARS, "PASS_TO_PASS")
        return f"""## TASK INFORMATION

**Instance ID:** {f.require("instance_id")}
**Repository:** {f.require("repo")}
**Version:** {f.optional("version", "unknown")}

**Problem Statement:**
{f.require("problem_statement")}

**Gold Patch (correct solution):**
```diff
{patch}
```

**Test Patch (tests that verify the fix):**
```diff
{_truncate(str(f.require("test_patch")), _MAX_TEST_PATCH_CHARS, "test_patch")}
```

**Tests that should pass after fix (FAIL_TO_PASS):**
{fail_to_pass}

**Regression tests (PASS_TO_PASS):**
{pass_to_pass}
{hints_section}"""

    return {
        InfoLevel.PROBLEM: problem_info,
        InfoLevel.TEST: test_info,
        InfoLevel.SOLUTION: solution_info,
    }


# =============================================================================
# SWE-bench Verified
# =============================================================================

SWEBENCH_VERIFIED_CONTEXT = TaskContext(
    name="swebench_verified",
    task_id_field="instance_id",
    scale_variant="code",
    system_intros={
        InfoLevel.PROBLEM: (
            "You are analyzing a SWE-bench Verified coding task to predict its difficulty. "
            "This is a BUG FIX task in a Python repository. "
            "You only have access to the problem statement."
        ),
        InfoLevel.ENVIRONMENT: (
            "You are auditing a **bug-fix** task in a Python repository. "
            "The /testbed directory contains the project codebase with a known bug. "
            "The problem statement describes the bug, and there are failing tests "
            "(FAIL_TO_PASS) that should pass after the fix."
        ),
        InfoLevel.TEST: (
            "You are analyzing a SWE-bench Verified coding task to predict its difficulty. "
            "This is a BUG FIX task in a Python repository. "
            "You have access to the problem statement and the test patch, but NOT the solution."
        ),
        InfoLevel.SOLUTION: (
            "You are analyzing a SWE-bench Verified coding task to predict its difficulty. "
            "This is a BUG FIX task in a Python repository. "
            "You have access to the full task information including the gold solution patch."
        ),
    },
    format_task_info_fns=_make_swebench_formatters("swebench_verified"),
)


# =============================================================================
# SWE-bench Pro
# =============================================================================

SWEBENCH_PRO_CONTEXT = TaskContext(
    name="swebench_pro",
    task_id_field="instance_id",
    scale_variant="code",
    system_intros={
        InfoLevel.PROBLEM: (
            "You are analyzing a SWE-bench Pro coding task to predict its difficulty. "
            "This is a BUG FIX task in a Python repository. "
            "SWE-bench Pro contains more challenging tasks than standard SWE-bench. "
            "You only have access to the problem statement."
        ),
        InfoLevel.ENVIRONMENT: (
            "You are auditing a **bug-fix** task in a Python repository. "
            "The /testbed directory contains the project codebase with a known bug. "
            "The problem statement describes the bug, and there are failing tests "
            "(FAIL_TO_PASS) that should pass after the fix. "
            "These are more challenging tasks than standard SWE-bench."
        ),
        InfoLevel.TEST: (
            "You are analyzing a SWE-bench Pro coding task to predict its difficulty. "
            "This is a BUG FIX task in a Python repository. "
            "SWE-bench Pro contains more challenging tasks than standard SWE-bench. "
            "You have access to the problem statement and the test patch, but NOT the solution."
        ),
        InfoLevel.SOLUTION: (
            "You are analyzing a SWE-bench Pro coding task to predict its difficulty. "
            "This is a BUG FIX task in a Python repository. "
            "SWE-bench Pro contains more challenging tasks than standard SWE-bench. "
            "You have access to the full task information including the gold solution patch."
        ),
    },
    format_task_info_fns=_make_swebench_formatters("swebench_pro"),
)


# =============================================================================
# TerminalBench
# =============================================================================

def _make_terminalbench_formatters() -> Dict[InfoLevel, Callable]:
    ds = "terminalbench"

    def problem_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, ds)
        return f"""## TASK INFORMATION

**Task ID:** {f.require("task_id")}

**Task Instruction:**
{f.require("problem_statement")}"""

    def test_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, ds)
        tags = f.optional_list("tags")
        return f"""## TASK INFORMATION

**Task ID:** {f.require("task_id")}
**Category:** {f.optional("category", "N/A")}
**Tags:** {", ".join(tags) if tags else "N/A"}

**Task Instruction:**
{f.require("problem_statement")}

**Evaluation Test Harness:**
```
{f.require("tests")}
```"""

    def solution_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, ds)
        tags = f.optional_list("tags")
        return f"""## TASK INFORMATION

**Task ID:** {f.require("task_id")}
**Category:** {f.optional("category", "N/A")}
**Tags:** {", ".join(tags) if tags else "N/A"}
**Claimed Difficulty:** {f.optional("difficulty", "N/A")}

**Task Instruction:**
{f.require("problem_statement")}

**Evaluation Test Harness:**
```
{f.require("tests")}
```

**Reference Solution (solution.sh):**
```bash
{_truncate(str(f.require("patch")), _MAX_PATCH_CHARS, "solution patch")}
```"""

    return {
        InfoLevel.PROBLEM: problem_info,
        InfoLevel.TEST: test_info,
        InfoLevel.SOLUTION: solution_info,
    }


TERMINALBENCH_CONTEXT = TaskContext(
    name="terminalbench",
    task_id_field="task_id",
    scale_variant="terminal",
    system_intros={
        InfoLevel.PROBLEM: (
            "You are analyzing a TerminalBench terminal/shell task to predict its difficulty. "
            "This task requires writing shell commands or scripts to accomplish a goal. "
            "You only have access to the task instruction."
        ),
        InfoLevel.ENVIRONMENT: (
            "You are auditing a **terminal automation** task in a sandboxed environment. "
            "The task requires using command-line tools to accomplish a goal. "
            "The environment may include pre-installed tools, configuration files, "
            "and multi-service Docker setups."
        ),
        InfoLevel.TEST: (
            "You are analyzing a TerminalBench terminal/shell task to predict its difficulty. "
            "This task requires writing shell commands or scripts to accomplish a goal. "
            "You have access to the task instruction and evaluation test harness, but NOT the reference solution."
        ),
        InfoLevel.SOLUTION: (
            "You are analyzing a TerminalBench terminal/shell task to predict its difficulty. "
            "This task requires writing shell commands or scripts to accomplish a goal. "
            "You have access to the full task information including the reference solution."
        ),
    },
    format_task_info_fns=_make_terminalbench_formatters(),
)


# =============================================================================
# GSO
# =============================================================================

def _make_gso_formatters() -> Dict[InfoLevel, Callable]:
    ds = "gso"

    def problem_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, ds)
        return f"""## TASK INFORMATION

**Instance ID:** {f.require("instance_id")}
**Repository:** {f.require("repo")}
**API/Function being optimized:** {f.require("api")}"""

    def test_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, ds)
        return f"""## TASK INFORMATION

**Instance ID:** {f.require("instance_id")}
**Repository:** {f.require("repo")}
**API/Function being optimized:** {f.require("api")}

**Performance Benchmark Script:**
```python
{f.require("prob_script")}
```"""

    def solution_info(task: Dict[str, Any]) -> str:
        f = _TaskFields(task, ds)
        hints = f.optional("hints_text")
        hints_section = f"\n**Hints:**\n{hints}" if hints else ""
        return f"""## TASK INFORMATION

**Instance ID:** {f.require("instance_id")}
**Repository:** {f.require("repo")}
**API/Function being optimized:** {f.require("api")}

**Performance Benchmark Script:**
```python
{f.require("prob_script")}
```

**Gold Patch (optimization solution):**
```diff
{f.require("gt_diff")}
```
{hints_section}"""

    return {
        InfoLevel.PROBLEM: problem_info,
        InfoLevel.TEST: test_info,
        InfoLevel.SOLUTION: solution_info,
    }


GSO_CONTEXT = TaskContext(
    name="gso",
    task_id_field="instance_id",
    scale_variant="optimization",
    system_intros={
        InfoLevel.PROBLEM: (
            "You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty. "
            "This is a PERFORMANCE OPTIMIZATION task, NOT a bug fix. "
            "The goal is to make code run faster while maintaining correctness. "
            "You only have access to the task description."
        ),
        InfoLevel.ENVIRONMENT: (
            "You are auditing a **performance optimization** task in a software project. "
            "The /testbed directory contains a codebase where specific code needs to be "
            "optimized for speed. The task includes performance tests that measure "
            "execution time before and after changes."
        ),
        InfoLevel.TEST: (
            "You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty. "
            "This is a PERFORMANCE OPTIMIZATION task, NOT a bug fix. "
            "The goal is to make code run faster while maintaining correctness. "
            "You have access to the task description and performance benchmark, but NOT the optimization solution."
        ),
        InfoLevel.SOLUTION: (
            "You are analyzing a GSO (Software Optimization Benchmark) task to predict its difficulty. "
            "This is a PERFORMANCE OPTIMIZATION task, NOT a bug fix. "
            "The goal is to make code run faster while maintaining correctness. "
            "You have access to the full task information including the gold optimization patch."
        ),
    },
    format_task_info_fns=_make_gso_formatters(),
)


# =============================================================================
# Registry
# =============================================================================

TASK_CONTEXTS: Dict[str, TaskContext] = {
    "swebench_verified": SWEBENCH_VERIFIED_CONTEXT,
    "swebench_pro": SWEBENCH_PRO_CONTEXT,
    "terminalbench": TERMINALBENCH_CONTEXT,
    "gso": GSO_CONTEXT,
}


def get_task_context(dataset: str) -> TaskContext:
    """Look up a task context by dataset name.

    Raises:
        KeyError: If the dataset is not registered.
    """
    if dataset not in TASK_CONTEXTS:
        raise KeyError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {sorted(TASK_CONTEXTS.keys())}"
        )
    return TASK_CONTEXTS[dataset]


# =============================================================================
# Auditor agent prompt builder (ENVIRONMENT level)
# =============================================================================

def build_auditor_system_prompt(task_type: str = "swebench_verified") -> str:
    """Build the system prompt for the auditor agent.

    Reads ENVIRONMENT-level feature definitions from the feature registry and
    the dataset-specific intro from the task context.

    Args:
        task_type: One of the registered dataset names.

    Returns:
        Complete system prompt string.

    Raises:
        KeyError: If task_type is not registered or lacks an ENVIRONMENT intro.
    """
    ctx = get_task_context(task_type)
    task_context = ctx.system_intros[InfoLevel.ENVIRONMENT]
    env_features = get_features_by_level(InfoLevel.ENVIRONMENT)

    # Build feature descriptions from registry
    features_text = "\n\n".join(
        feat.get_scale_text("default") for feat in env_features
    )

    # Build example JSON
    example_features = [
        f'  "{feat.name}": {{"value": 3, "reasoning": "Brief explanation"}}'
        for feat in env_features
    ]
    example_json = "{\n" + ",\n".join(example_features) + "\n}"

    num_features = len(env_features)

    return f"""You are a codebase auditor evaluating task environments for difficulty prediction. Your job is to explore the environment and rate it on {num_features} difficulty-related axes.

## Task Context

{task_context}

## Your Task

1. Explore the working directory to understand the project structure
2. Read the problem statement (provided as input)
3. Try to understand the scope and complexity of the task
4. Check available tools, tests, and dependencies
5. Rate the environment on the {num_features} axes below

## Features to Assess (1-5 scale)

{features_text}

## Output Format

After your exploration (use 3-8 tool calls), output your final assessment as a JSON object with exactly {num_features} features. Each feature should be an object with "value" (1-5 integer) and "reasoning" (brief explanation):

```json
{example_json}
```

**CRITICAL**: Your final message MUST contain a valid JSON object with all {num_features} features. Do not forget any features.

## Tips

- Start with `ls` or `find` to understand the project structure
- Check for test files, configuration files, and dependency files
- Look at file extensions to understand the tech stack
- Use `wc -l` or `find . -type f | wc -l` to gauge codebase size
- Check `requirements.txt`, `setup.py`, `Cargo.toml`, etc. for dependencies
- Keep your exploration focused - aim for 3-8 tool calls

## IMPORTANT: How to Complete

After 3-8 exploration commands, you MUST call the `submit()` function with your JSON report.
Do NOT try to solve the task - just audit and rate the environment.

Now begin your audit. Start by exploring the working directory structure, then submit your ratings.
"""
