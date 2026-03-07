"""Parse auditor agent outputs and extract features to CSV.

This module reads Inspect AI log files from auditor runs and extracts
the 8 difficulty-related features into a CSV file for use in Experiment A.

Usage:
    # Parse logs and create CSV
    python -m llm_judge_feature_extraction.auditor_agent.parse_outputs --log_dir chris_output/auditor_runs

    # Validate parsed results
    python -m llm_judge_feature_extraction.auditor_agent.parse_outputs --log_dir chris_output/auditor_runs --validate
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.log import read_eval_log

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from llm_judge_feature_extraction.feature_registry import get_features_by_level
from llm_judge_feature_extraction.prompt_config import InfoLevel

EXPECTED_FEATURES_V4 = [f.name for f in get_features_by_level(InfoLevel.ENVIRONMENT)]

# Default to V4 (current version)
EXPECTED_FEATURES = EXPECTED_FEATURES_V4


def parse_completion(
    completion: str,
    expected_features: list[str] | None = None,
) -> dict[str, Any] | None:
    """Parse the completion to extract feature values.

    Args:
        completion: The model's completion text (should be JSON)
        expected_features: List of feature names to extract

    Returns:
        Dict mapping feature names to values, or None if parsing failed
    """
    if not completion:
        return None

    # Try to parse as JSON directly
    try:
        data = json.loads(completion)
        return extract_features_from_json(data, expected_features)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    code_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', completion, re.DOTALL)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            return extract_features_from_json(data, expected_features)
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object - look for first feature name as anchor
    first_feature = (expected_features or EXPECTED_FEATURES)[0]
    json_match = re.search(rf'\{{[^{{}}]*"{first_feature}"[^{{}}]*\}}', completion, re.DOTALL)
    if json_match:
        try:
            # This might be a partial match, try to find complete JSON
            start = completion.find('{')
            if start >= 0:
                # Count braces to find matching end
                depth = 0
                for i, c in enumerate(completion[start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = completion[start:start + i + 1]
                            data = json.loads(json_str)
                            return extract_features_from_json(data, expected_features)
        except json.JSONDecodeError:
            pass

    return None


def extract_features_from_json(
    data: dict[str, Any],
    expected_features: list[str] | None = None,
) -> dict[str, Any]:
    """Extract feature values from parsed JSON.

    Handles two formats:
    1. {"feature": {"value": N, "reasoning": "..."}}
    2. {"feature": N}

    Args:
        data: Parsed JSON object
        expected_features: List of feature names to extract (default: EXPECTED_FEATURES)

    Returns:
        Dict mapping feature names to integer values
    """
    if expected_features is None:
        expected_features = EXPECTED_FEATURES

    features = {}

    for feature_name in expected_features:
        if feature_name not in data:
            features[feature_name] = None
            features[f"{feature_name}_reasoning"] = None
            continue

        value = data[feature_name]

        if isinstance(value, dict):
            # Format: {"value": N, "reasoning": "..."}
            features[feature_name] = value.get("value")
            features[f"{feature_name}_reasoning"] = value.get("reasoning")
        elif isinstance(value, (int, float)):
            # Format: N
            features[feature_name] = int(value)
            features[f"{feature_name}_reasoning"] = None
        else:
            features[feature_name] = None
            features[f"{feature_name}_reasoning"] = None

    return features


def parse_log_file(
    log_path: Path,
    expected_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse a single log file and extract features for all samples.

    Args:
        log_path: Path to .eval log file
        expected_features: List of feature names to extract

    Returns:
        List of dicts, each containing instance_id and feature values
    """
    results = []

    try:
        log = read_eval_log(str(log_path))

        for sample in log.samples or []:
            instance_id = str(sample.id) if sample.id else None
            if not instance_id:
                continue

            result = {"instance_id": instance_id}

            # Get completion
            completion = sample.output.completion if sample.output else None

            if completion:
                features = parse_completion(completion, expected_features)
                if features:
                    result.update(features)
                else:
                    result["_parse_error"] = "Could not parse completion"
            else:
                result["_parse_error"] = "No completion"

            results.append(result)

    except Exception as e:
        print(f"Error reading {log_path}: {e}")

    return results


def parse_all_logs(
    log_dir: Path,
    expected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Parse all log files in a directory.

    Args:
        log_dir: Directory containing .eval log files
        expected_features: List of feature names to extract

    Returns:
        DataFrame with instance_id and feature columns
    """
    all_results = []

    # Sort by modification time so later runs override earlier ones
    log_files = sorted(log_dir.rglob("*.eval"), key=lambda p: p.stat().st_mtime)
    print(f"Found {len(log_files)} log files in {log_dir}")

    for log_path in log_files:
        results = parse_log_file(log_path, expected_features)
        all_results.extend(results)

    if not all_results:
        print("No results found!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Remove duplicates (keep latest by file order)
    if "instance_id" in df.columns:
        df = df.drop_duplicates(subset=["instance_id"], keep="last")

    print(f"Parsed {len(df)} unique samples")

    return df


def validate_results(
    df: pd.DataFrame,
    expected_features: list[str] | None = None,
) -> bool:
    """Validate parsed results.

    Checks:
    - All expected features are present
    - Values are in valid range (1-5)
    - No missing values

    Args:
        df: DataFrame with parsed features
        expected_features: List of feature names to validate

    Returns:
        True if validation passes, False otherwise
    """
    if expected_features is None:
        expected_features = EXPECTED_FEATURES

    is_valid = True

    # Check for expected feature columns
    for feature in expected_features:
        if feature not in df.columns:
            print(f"FAIL: Missing feature column: {feature}")
            is_valid = False
        else:
            # Check value range
            valid_values = df[feature].dropna()
            if len(valid_values) > 0:
                out_of_range = valid_values[(valid_values < 1) | (valid_values > 5)]
                if len(out_of_range) > 0:
                    print(f"WARN: {feature} has {len(out_of_range)} values out of range [1-5]")

            # Check missing values
            missing = df[feature].isna().sum()
            if missing > 0:
                print(f"WARN: {feature} has {missing} missing values ({missing / len(df) * 100:.1f}%)")

    # Check for parse errors
    if "_parse_error" in df.columns:
        errors = df["_parse_error"].notna().sum()
        if errors > 0:
            print(f"WARN: {errors} samples had parse errors")
            print("  Errors:", df[df["_parse_error"].notna()]["_parse_error"].value_counts().to_dict())

    # Print summary statistics
    print("\nFeature statistics:")
    for feature in expected_features:
        if feature in df.columns:
            valid = df[feature].dropna()
            if len(valid) > 0:
                print(f"  {feature}: mean={valid.mean():.2f}, std={valid.std():.2f}, range=[{valid.min()}, {valid.max()}]")

    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Parse auditor agent outputs and extract features"
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        required=True,
        help="Directory containing .eval log files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: log_dir/auditor_features.csv)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate parsed results",
    )
    args = parser.parse_args()

    expected_features = EXPECTED_FEATURES_V4
    print(f"Using v4 features: {expected_features}")

    # Parse logs
    df = parse_all_logs(args.log_dir, expected_features)

    if df.empty:
        print("No data to save!")
        sys.exit(1)

    # Validate if requested
    if args.validate:
        valid = validate_results(df, expected_features)
        if not valid:
            print("\nValidation failed!")
            sys.exit(1)

    # Save to CSV
    output_path = args.output or (args.log_dir / "auditor_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select columns for output (instance_id + features + reasoning)
    output_cols = ["instance_id"]
    for feature in expected_features:
        if feature in df.columns:
            output_cols.append(feature)
        reasoning_col = f"{feature}_reasoning"
        if reasoning_col in df.columns:
            output_cols.append(reasoning_col)

    df[output_cols].to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
