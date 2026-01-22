"""Prompt configuration for LLM judge feature extraction.

This module defines the PromptConfig and FeatureDefinition dataclasses that
configure how features are extracted from different datasets.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FeatureDefinition:
    """Definition of a single feature to extract.

    Attributes:
        name: Feature name (e.g., "fix_complexity")
        min_value: Minimum valid value (e.g., 1)
        max_value: Maximum valid value (e.g., 5)
        description: Human-readable description for documentation
    """

    name: str
    min_value: int
    max_value: int
    description: str = ""

    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this feature."""
        if not isinstance(value, (int, float)):
            return False
        return self.min_value <= value <= self.max_value


@dataclass
class PromptConfig:
    """Configuration for LLM judge feature extraction.

    This dataclass encapsulates all dataset-specific configuration:
    - Feature definitions (names, scales)
    - Prompt template with placeholders
    - Task ID field name
    - Truncation limits for long fields

    Attributes:
        name: Dataset name (e.g., "swebench", "terminalbench")
        features: List of feature definitions
        prompt_template: The prompt string with {placeholders}
        task_id_field: Field name for task ID (e.g., "instance_id", "task_id")
        truncation_limits: Max characters for each field (e.g., {"problem_statement": 12000})
        format_prompt_fn: Optional custom formatting function
    """

    name: str
    features: List[FeatureDefinition]
    prompt_template: str
    task_id_field: str
    truncation_limits: Dict[str, int] = field(default_factory=dict)
    format_prompt_fn: Optional[Callable[[Dict[str, Any]], str]] = None

    def get_feature_names(self) -> List[str]:
        """Return list of feature names for CSV columns."""
        return [f.name for f in self.features]

    def format_prompt(self, task: Dict[str, Any]) -> str:
        """Format the prompt template with task data.

        If a custom format_prompt_fn is provided, it will be used.
        Otherwise, the prompt_template is formatted with task fields,
        applying truncation limits.

        Args:
            task: Task dictionary with fields matching template placeholders

        Returns:
            Formatted prompt string
        """
        if self.format_prompt_fn is not None:
            return self.format_prompt_fn(task)

        # Apply truncation limits
        truncated_task = {}
        for key, value in task.items():
            if isinstance(value, str) and key in self.truncation_limits:
                limit = self.truncation_limits[key]
                truncated_task[key] = value[:limit] if len(value) > limit else value
            else:
                truncated_task[key] = value

        return self.prompt_template.format(**truncated_task)

    def validate_response(self, data: Dict[str, Any]) -> bool:
        """Validate that extracted features match the schema.

        Args:
            data: Parsed JSON response from LLM

        Returns:
            True if at least one expected feature is present and valid
        """
        feature_names = self.get_feature_names()
        has_valid_feature = False

        for feature in self.features:
            if feature.name in data:
                if feature.validate(data[feature.name]):
                    has_valid_feature = True

        return has_valid_feature

    def get_metadata_fields(self) -> List[str]:
        """Return list of metadata field names to include in output."""
        return [
            f"_{self.task_id_field}",  # Internal task ID field
            "_model",
            "_provider",
            "_extracted_at",
        ]
