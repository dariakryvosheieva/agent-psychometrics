"""Core data types for LLM judge feature extraction.

- InfoLevel: what task information a feature requires
- FeatureDefinition: a single extractable feature with scale text and info level
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class InfoLevel(Enum):
    """What task information a feature needs access to.

    Features are grouped by info level during extraction so that each group's
    prompt contains only the appropriate task data. Levels are cumulative:
    - PROBLEM: only problem statement
    - ENVIRONMENT: problem + shell exploration (auditor agent, no tests/solution)
    - TEST: problem + test/evaluation artifact (no solution)
    - SOLUTION: problem + tests + gold solution
    """

    PROBLEM = "problem"
    ENVIRONMENT = "environment"
    TEST = "test"
    SOLUTION = "solution"


@dataclass
class FeatureDefinition:
    """Definition of a single feature to extract.

    Attributes:
        name: Feature name (e.g., "solution_complexity")
        min_value: Minimum valid value (e.g., 1)
        max_value: Maximum valid value (e.g., 5)
        description: Human-readable description for documentation
        info_level: What task information this feature requires
        scale_text: Per-variant rubric text for prompts. Keys are variant names
            ("default", "code", "terminal", "optimization"). The variant is
            selected based on the dataset being processed.
    """

    name: str
    min_value: int
    max_value: int
    description: str = ""
    info_level: InfoLevel = InfoLevel.PROBLEM
    scale_text: Dict[str, str] = field(default_factory=dict)

    def get_scale_text(self, variant: str = "default") -> str:
        """Get the rubric text for a given dataset variant.

        Falls back to "default" if the requested variant is not found.

        Raises:
            KeyError: If neither the requested variant nor "default" exists.
        """
        if variant in self.scale_text:
            return self.scale_text[variant]
        if "default" in self.scale_text:
            return self.scale_text["default"]
        raise KeyError(
            f"No scale text for variant '{variant}' in feature '{self.name}'. "
            f"Available: {list(self.scale_text.keys())}"
        )

    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this feature."""
        if not isinstance(value, (int, float)):
            return False
        return self.min_value <= value <= self.max_value
