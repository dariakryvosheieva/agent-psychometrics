"""Parse test output from trajectory message content.

Handles multiple test frameworks:
- pytest (verbose and summary formats)
- unittest/Django test runner
- Custom test scripts (best effort)
"""

import re
from typing import List, Optional

from .types import TestRun, TestStatus


# Regex patterns for different test frameworks
PATTERNS = {
    # Pytest verbose: test_file.py::TestClass::test_method PASSED/FAILED [XX%]
    "pytest_verbose": re.compile(
        r"^([\w/\.\-]+\.py::[^\s]+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
        re.MULTILINE,
    ),
    # Pytest summary: "X passed, Y failed in Z.ZZs" or "X passed in Z.ZZs"
    # Handles various formats: "1 passed", "1 passed, 2 failed", "1 passed, 1 warning"
    "pytest_summary": re.compile(
        r"=+\s*"
        r"(?:(\d+)\s+passed)?"
        r"[,\s]*(?:(\d+)\s+failed)?"
        r"[,\s]*(?:(\d+)\s+error(?:s|ed)?)?"
        r"[,\s]*(?:(\d+)\s+skipped)?"
        r"[,\s]*(?:(\d+)\s+warnings?)?"
        r"[,\s]*(?:(\d+)\s+deselected)?"
        r"\s*in\s+([\d\.]+)s?"
        r"\s*=+",
        re.IGNORECASE,
    ),
    # Django/unittest: "Ran X tests in Y.YYYs"
    "unittest_ran": re.compile(
        r"Ran\s+(\d+)\s+tests?\s+in\s+([\d\.]+)s",
        re.IGNORECASE,
    ),
    # Unittest OK line
    "unittest_ok": re.compile(r"^OK\s*$", re.MULTILINE),
    # Unittest FAILED line: "FAILED (failures=N, errors=M)"
    "unittest_failed": re.compile(
        r"FAILED\s*\((?:failures=(\d+))?[,\s]*(?:errors=(\d+))?\)",
        re.IGNORECASE,
    ),
    # Test function names in unittest format: test_method (TestClass) ... ok/FAIL
    "unittest_verbose": re.compile(
        r"^(test_\w+)\s*\((\w+)\)\s*\.\.\.\s*(ok|FAIL|ERROR|skip)",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Short pytest format: FAILED test_file.py::test_name - Error message
    "pytest_short_failed": re.compile(
        r"^FAILED\s+([\w/\.\-]+\.py::[^\s]+)",
        re.MULTILINE,
    ),
    # Collected items: "collected X items" or "collected X item"
    "pytest_collected": re.compile(
        r"collected\s+(\d+)\s+items?",
        re.IGNORECASE,
    ),
}


def detect_framework(content: str) -> str:
    """Detect which test framework produced this output.

    Args:
        content: Message content to analyze

    Returns:
        Framework name: "pytest", "unittest", or "unknown"
    """
    # Strong pytest indicators
    if "::test_" in content or "::Test" in content:
        return "pytest"
    if "pytest" in content.lower():
        return "pytest"
    if PATTERNS["pytest_summary"].search(content):
        return "pytest"

    # Unittest indicators
    if PATTERNS["unittest_ran"].search(content):
        return "unittest"
    if "Ran " in content and " tests in " in content:
        return "unittest"

    return "unknown"


def _parse_status(status_str: str) -> TestStatus:
    """Convert status string to TestStatus enum."""
    status_upper = status_str.upper()
    if status_upper in ("PASSED", "OK"):
        return TestStatus.PASSED
    elif status_upper == "FAILED" or status_upper == "FAIL":
        return TestStatus.FAILED
    elif status_upper == "ERROR":
        return TestStatus.ERROR
    elif status_upper in ("SKIPPED", "SKIP"):
        return TestStatus.SKIPPED
    return TestStatus.UNKNOWN


def parse_pytest_output(content: str) -> Optional[TestRun]:
    """Parse pytest-style test output.

    Args:
        content: Message content containing pytest output

    Returns:
        TestRun if test output found, None otherwise
    """
    run = TestRun(run_index=0, message_index=0, framework="pytest")

    # Try to parse verbose results first (test_file.py::TestClass::test PASSED)
    for match in PATTERNS["pytest_verbose"].finditer(content):
        test_id = match.group(1)
        status = _parse_status(match.group(2))
        run.individual_results[test_id] = status

    # Also check for FAILED lines without PASSED (some formats)
    for match in PATTERNS["pytest_short_failed"].finditer(content):
        test_id = match.group(1)
        if test_id not in run.individual_results:
            run.individual_results[test_id] = TestStatus.FAILED

    # Parse summary line
    summary_match = PATTERNS["pytest_summary"].search(content)
    if summary_match:
        run.passed_count = int(summary_match.group(1) or 0)
        run.failed_count = int(summary_match.group(2) or 0)
        run.error_count = int(summary_match.group(3) or 0)
        run.skipped_count = int(summary_match.group(4) or 0)
        run.duration_seconds = float(summary_match.group(7))
        run.total_count = (
            run.passed_count + run.failed_count + run.error_count + run.skipped_count
        )
        run.summary_line = summary_match.group(0)
        return run

    # If we have individual results but no summary, compute summary from them
    if run.individual_results:
        for status in run.individual_results.values():
            if status == TestStatus.PASSED:
                run.passed_count += 1
            elif status == TestStatus.FAILED:
                run.failed_count += 1
            elif status == TestStatus.ERROR:
                run.error_count += 1
            elif status == TestStatus.SKIPPED:
                run.skipped_count += 1
        run.total_count = len(run.individual_results)
        return run

    # Check if we have collected items without results (test setup phase)
    collected_match = PATTERNS["pytest_collected"].search(content)
    if collected_match:
        # We have a test run starting but maybe no results yet
        run.total_count = int(collected_match.group(1))
        # Don't return this as a valid run since we don't have results

    return None


def parse_unittest_output(content: str) -> Optional[TestRun]:
    """Parse unittest/Django-style test output.

    Args:
        content: Message content containing unittest output

    Returns:
        TestRun if test output found, None otherwise
    """
    run = TestRun(run_index=0, message_index=0, framework="unittest")

    # Parse "Ran X tests in Y.YYYs"
    ran_match = PATTERNS["unittest_ran"].search(content)
    if not ran_match:
        return None

    run.total_count = int(ran_match.group(1))
    run.duration_seconds = float(ran_match.group(2))

    # Check for OK or FAILED
    if PATTERNS["unittest_ok"].search(content):
        run.passed_count = run.total_count
        run.failed_count = 0
        run.error_count = 0
    else:
        failed_match = PATTERNS["unittest_failed"].search(content)
        if failed_match:
            run.failed_count = int(failed_match.group(1) or 0)
            run.error_count = int(failed_match.group(2) or 0)
            run.passed_count = run.total_count - run.failed_count - run.error_count
        else:
            # No explicit OK or FAILED - might be truncated
            # Assume passed if no failures mentioned
            run.passed_count = run.total_count

    # Try to get verbose individual results
    for match in PATTERNS["unittest_verbose"].finditer(content):
        test_method = match.group(1)
        test_class = match.group(2)
        status = _parse_status(match.group(3))
        test_id = f"{test_class}::{test_method}"
        run.individual_results[test_id] = status

    run.summary_line = ran_match.group(0)
    return run


def parse_test_output(content: str) -> Optional[TestRun]:
    """Parse test output from any framework.

    Args:
        content: Message content to parse

    Returns:
        TestRun if test output found, None otherwise
    """
    framework = detect_framework(content)

    if framework == "pytest":
        result = parse_pytest_output(content)
        if result and result.total_count > 0:
            return result

    if framework == "unittest":
        result = parse_unittest_output(content)
        if result and result.total_count > 0:
            return result

    # Try both parsers as fallback for unknown framework
    result = parse_pytest_output(content)
    if result and result.total_count > 0:
        return result

    result = parse_unittest_output(content)
    if result and result.total_count > 0:
        return result

    return None


def extract_all_test_runs(messages: List[dict]) -> List[TestRun]:
    """Extract all test runs from trajectory messages.

    Args:
        messages: List of message dicts from trajectory

    Returns:
        List of TestRun objects in chronological order
    """
    runs = []

    for msg_idx, msg in enumerate(messages):
        # Test output typically appears in "user" role messages (tool output)
        # but check all messages to be safe
        role = msg.get("role", "")
        if role == "system":
            continue

        content = msg.get("content", "")
        if not content:
            continue

        # Skip very short content (unlikely to have test output)
        if len(content) < 50:
            continue

        # Check if this message contains test output
        test_run = parse_test_output(content)
        if test_run and test_run.total_count > 0:
            test_run.run_index = len(runs)
            test_run.message_index = msg_idx
            runs.append(test_run)

    return runs
