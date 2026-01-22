"""Response parsing utilities for LLM judge feature extraction.

Handles JSON extraction from LLM responses, including various output formats
(raw JSON, markdown code blocks, etc.).
"""

import json
import re
from typing import Any, Dict, List, Optional


def parse_llm_response(
    text: str,
    expected_features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Parse the LLM response to extract semantic features.

    Handles various output formats:
    1. Raw JSON object
    2. JSON wrapped in ```json ... ``` code block
    3. JSON embedded in other text

    Args:
        text: Raw response text from the LLM
        expected_features: Optional list of feature names to look for when
            searching for embedded JSON. If provided, will search for a JSON
            object containing at least one of these features.

    Returns:
        Parsed dictionary with features, or None if parsing failed
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Try parsing as raw JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Look for ```json block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: Look for any ``` block (sometimes models omit 'json')
    code_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: Look for JSON object containing expected features
    if expected_features:
        for feature in expected_features:
            # Escape feature name for regex
            escaped_feature = re.escape(feature)
            # Look for a JSON object containing this feature
            # Match from { to } but handle nested braces
            pattern = r'\{[^{}]*"' + escaped_feature + r'"[^{}]*\}'
            json_match = re.search(pattern, text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass

    # Strategy 5: Find the first { and last } and try to parse
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(text[first_brace : last_brace + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def validate_features(
    data: Dict[str, Any],
    expected_features: List[str],
    require_all: bool = False,
) -> bool:
    """Validate that extracted data contains expected features.

    Args:
        data: Parsed response dictionary
        expected_features: List of feature names that should be present
        require_all: If True, all features must be present. If False (default),
            at least one feature must be present.

    Returns:
        True if validation passes, False otherwise
    """
    if not data or not isinstance(data, dict):
        return False

    if require_all:
        return all(f in data for f in expected_features)
    else:
        return any(f in data for f in expected_features)
