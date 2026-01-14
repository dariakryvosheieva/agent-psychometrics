"""Data types for test progression feature extraction.

Defines dataclasses for:
- TestStatus: Enum for test outcomes
- TestRun: Single test execution within a trajectory
- TestProgression: Complete test progression for a trajectory
- TestProgressionFeatures: Computed features from test progression
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TestStatus(Enum):
    """Status of a single test."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


@dataclass
class TestRun:
    """A single test execution within a trajectory.

    Represents one pytest/unittest execution with its results.
    """

    run_index: int  # 0-indexed position in trajectory's test runs
    message_index: int  # Which message contains this output
    framework: str  # "pytest", "unittest", "django", "unknown"

    # Granular results (if available - ~22% of trajectories)
    # Maps test_id -> status
    individual_results: Dict[str, TestStatus] = field(default_factory=dict)

    # Summary stats (always available from summary line parsing)
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    total_count: int = 0

    # Timing info
    duration_seconds: Optional[float] = None

    # Raw summary line for debugging
    summary_line: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        """Compute pass rate for this test run."""
        if self.total_count == 0:
            return 0.0
        return self.passed_count / self.total_count


@dataclass
class TestProgression:
    """Complete test progression for a trajectory.

    Contains all test runs from a single agent's attempt at a task.
    """

    task_id: str
    agent: str
    resolved: bool

    # All test runs in chronological order
    runs: List[TestRun] = field(default_factory=list)

    # Metadata
    has_test_output: bool = False
    has_granular_results: bool = False
    framework_detected: str = "unknown"


@dataclass
class TestProgressionFeatures:
    """Computed features from test progression (task-level).

    These features capture how agents iterate on test failures.
    """

    # Basic counts
    num_test_runs: int = 0

    # Pass rate progression
    initial_pass_rate: float = 0.0  # Pass rate on first run
    final_pass_rate: float = 0.0  # Pass rate on last run
    pass_rate_improvement: float = 0.0  # final - initial
    max_pass_rate: float = 0.0  # Best pass rate achieved

    # Improvement dynamics
    runs_until_first_improvement: int = 0  # 0 if no improvement
    runs_until_max_pass_rate: int = 0
    improvement_slope: float = 0.0  # Linear regression slope

    # Blocking tests (tests that fail longest)
    num_blocking_tests: int = 0  # Tests that fail on all runs
    blocking_test_ids: List[str] = field(default_factory=list)

    # Test stability
    test_churn_rate: float = 0.0  # Fraction of tests that flip status

    # Timing between runs
    avg_messages_between_runs: float = 0.0

    # Framework
    framework: str = "unknown"

    # Confidence flags
    has_granular_data: bool = False  # If we have per-test results
    has_test_output: bool = False  # If we have any test output
