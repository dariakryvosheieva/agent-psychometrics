"""Post-process Lunette raw responses to extract features from free-form text.

The Lunette GradingPlan sometimes returns free-form text explanations instead of
structured JSON. This script extracts feature values from those text responses
using regex patterns and heuristics.

Usage:
    python -m experiment_a.postprocess_lunette_features
    python -m experiment_a.postprocess_lunette_features --dry_run
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Feature definitions with expected ranges and extraction patterns
FEATURE_SPECS = {
    # Environment-based features (integers)
    "repo_file_count": {"type": "int", "min": 0, "max": 100000},
    "repo_line_count": {"type": "int", "min": 0, "max": 10000000},
    "patch_file_count": {"type": "int", "min": 1, "max": 50},
    "patch_line_count": {"type": "int", "min": 1, "max": 1000},
    "test_file_count": {"type": "int", "min": 0, "max": 10000},
    "related_file_count": {"type": "int", "min": 0, "max": 1000},
    "import_count": {"type": "int", "min": 0, "max": 100},
    "class_count_in_file": {"type": "int", "min": 0, "max": 50},
    "function_count_in_file": {"type": "int", "min": 0, "max": 200},
    "test_count_fail_to_pass": {"type": "int", "min": 0, "max": 100},
    "test_count_pass_to_pass": {"type": "int", "min": 0, "max": 1000},
    "git_commit_count": {"type": "int", "min": 0, "max": 100000},
    "directory_depth": {"type": "int", "min": 1, "max": 15},
    "has_conftest": {"type": "binary", "min": 0, "max": 1},
    "has_init": {"type": "binary", "min": 0, "max": 1},
    # Semantic features (scales)
    "fix_in_description": {"type": "scale", "min": 0, "max": 3},
    "problem_clarity": {"type": "scale", "min": 1, "max": 5},
    "error_message_provided": {"type": "binary", "min": 0, "max": 1},
    "reproduction_steps": {"type": "binary", "min": 0, "max": 1},
    "fix_locality": {"type": "scale", "min": 1, "max": 3},
    "domain_knowledge_required": {"type": "scale", "min": 1, "max": 5},
    "fix_complexity": {"type": "scale", "min": 1, "max": 5},
    "logical_reasoning_required": {"type": "scale", "min": 1, "max": 5},
    "atypicality": {"type": "scale", "min": 1, "max": 5},
}


def extract_number_after_pattern(text: str, patterns: List[str]) -> Optional[int]:
    """Extract a number following any of the given patterns."""
    for pattern in patterns:
        # Try pattern: value format (e.g., "fix_in_description: 3")
        match = re.search(rf"{pattern}[:\s=]+(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try pattern=value format (e.g., "fix_in_description=3")
        match = re.search(rf"{pattern}\s*=\s*(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try (pattern: value) format (e.g., "(fix_in_description: 3)")
        match = re.search(rf"\({pattern}[:\s=]+(\d+)\)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try "pattern X/Y" format (e.g., "domain knowledge: 4/5")
        match = re.search(rf"{pattern}[:\s]+(\d+)\s*/\s*\d+", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try "score: X" after pattern mention
        match = re.search(rf"{pattern}[^.]*?score[:\s]+(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def extract_feature_value(text: str, feature_name: str, spec: dict) -> Optional[int]:
    """Extract a single feature value from text."""
    # Normalize feature name for pattern matching
    patterns = [
        feature_name,
        feature_name.replace("_", " "),
        feature_name.replace("_", "-"),
    ]

    # Add alternative names for some features
    alt_names = {
        "fix_in_description": ["fix in description", "fix_in_desc", "hint in description"],
        "domain_knowledge_required": ["domain knowledge", "domain expertise", "domain_knowledge"],
        "fix_complexity": ["fix complexity", "complexity", "implementation complexity"],
        "logical_reasoning_required": ["logical reasoning", "reasoning required", "reasoning"],
        "problem_clarity": ["problem clarity", "clarity", "clear problem"],
        "fix_locality": ["fix locality", "locality", "localized fix"],
        "atypicality": ["atypicality", "atypical", "unusual"],
        "patch_file_count": ["patch files", "files changed", "modified files"],
        "patch_line_count": ["patch lines", "lines changed", "loc changed"],
        "test_count_fail_to_pass": ["fail to pass", "failing tests", "tests to fix"],
        "test_count_pass_to_pass": ["pass to pass", "regression tests", "existing tests"],
    }

    if feature_name in alt_names:
        patterns.extend(alt_names[feature_name])

    value = extract_number_after_pattern(text, patterns)

    if value is not None:
        # Validate range
        if spec["min"] <= value <= spec["max"]:
            return value
        # If out of range, try to clamp for scale features
        if spec["type"] == "scale":
            return max(spec["min"], min(spec["max"], value))

    return None


def extract_from_score_field(score: float, feature_name: str) -> Optional[Dict[str, int]]:
    """Extract difficulty estimate from Lunette's score field (0-1 scale)."""
    # The score field is sometimes a 0-1 difficulty estimate
    # Map it to fix_complexity or overall difficulty
    if 0 <= score <= 1:
        # Convert to 1-5 scale
        difficulty_1_5 = int(round(1 + score * 4))
        return {"estimated_difficulty_from_score": difficulty_1_5}
    return None


def extract_features_from_dict(source: dict, features: dict) -> int:
    """Extract features from a dict into the features dict.

    Returns count of features extracted.
    """
    count = 0
    for feature_name, spec in FEATURE_SPECS.items():
        if feature_name in source:
            value = source[feature_name]
            if isinstance(value, (int, float)):
                # Validate and add
                if spec["min"] <= value <= spec["max"]:
                    features[feature_name] = int(value) if spec["type"] != "float" else value
                    count += 1
    return count


def infer_features_from_text(text: str, score: Optional[float]) -> Dict:
    """Infer feature values from descriptive text.

    Maps qualitative descriptions to numeric scales.
    """
    features = {}
    text_lower = text.lower()

    # Infer fix_complexity from descriptions
    if any(w in text_lower for w in ["trivial", "extremely easy", "very easy", "simple typo", "one-liner", "single line", "1 line"]):
        features["fix_complexity"] = 1
    elif any(w in text_lower for w in ["easy", "simple", "straightforward"]) and "not" not in text_lower[:text_lower.find("easy") if "easy" in text_lower else 0]:
        features["fix_complexity"] = 2
    elif any(w in text_lower for w in ["moderate", "moderately"]) and "above" not in text_lower:
        features["fix_complexity"] = 3
    elif any(w in text_lower for w in ["above average difficulty", "significantly above average", "not a straightforward"]):
        features["fix_complexity"] = 4
    elif any(w in text_lower for w in ["complex", "difficult", "hard", "challenging"]) and "not" not in text_lower[:20]:
        features["fix_complexity"] = 4
    elif any(w in text_lower for w in ["very complex", "extremely difficult", "highly complex"]):
        features["fix_complexity"] = 5

    # Infer domain_knowledge from descriptions
    if any(w in text_lower for w in ["no domain knowledge", "basic knowledge", "no expertise", "zero domain"]):
        features["domain_knowledge_required"] = 1
    elif any(w in text_lower for w in ["basic-to-intermediate", "basic understanding", "minimal domain"]):
        features["domain_knowledge_required"] = 2
    elif any(w in text_lower for w in ["moderate domain", "framework-specific", "some domain"]):
        features["domain_knowledge_required"] = 3
    elif any(w in text_lower for w in ["deep domain", "significant domain", "considerable domain", "deep understanding"]):
        features["domain_knowledge_required"] = 4
    elif any(w in text_lower for w in ["extensive domain", "expert knowledge", "highly specialized"]) or ("domain knowledge" in text_lower and "essential" in text_lower):
        features["domain_knowledge_required"] = 5

    # Infer problem_clarity from descriptions
    if any(w in text_lower for w in ["crystal clear", "extremely clear", "very clear", "maximum clarity"]):
        features["problem_clarity"] = 5
    elif any(w in text_lower for w in ["clear", "well-documented", "clear problem"]):
        features["problem_clarity"] = 4
    elif any(w in text_lower for w in ["moderately clear", "somewhat clear"]):
        features["problem_clarity"] = 3
    elif any(w in text_lower for w in ["unclear", "vague", "ambiguous"]):
        features["problem_clarity"] = 2
    elif any(w in text_lower for w in ["very unclear", "extremely vague"]):
        features["problem_clarity"] = 1

    # Infer fix_in_description from descriptions
    if any(w in text_lower for w in ["explicit solution", "explicitly states", "exact solution", "hints at the exact", "provides the fix"]):
        features["fix_in_description"] = 3
    elif any(w in text_lower for w in ["hints", "suggested", "provides hints", "hint in description"]):
        features["fix_in_description"] = 2
    elif any(w in text_lower for w in ["partial hint", "some guidance"]):
        features["fix_in_description"] = 1
    elif any(w in text_lower for w in ["no hint", "no guidance", "without hints"]):
        features["fix_in_description"] = 0

    # Infer fix_locality from descriptions
    if any(w in text_lower for w in ["single line", "1 line", "one line", "one location", "single location", "minimal locality", "low locality"]):
        features["fix_locality"] = 1
    elif any(w in text_lower for w in ["few lines", "single file", "localized"]):
        features["fix_locality"] = 2
    elif any(w in text_lower for w in ["multiple files", "multiple locations", "spread across"]):
        features["fix_locality"] = 3

    # Infer logical_reasoning from descriptions
    if any(w in text_lower for w in ["no reasoning", "mechanical", "zero problem-solving", "no logical reasoning"]):
        features["logical_reasoning_required"] = 1
    elif any(w in text_lower for w in ["minimal reasoning", "simple reasoning", "basic reasoning"]):
        features["logical_reasoning_required"] = 2
    elif any(w in text_lower for w in ["moderate reasoning", "moderate logical"]):
        features["logical_reasoning_required"] = 3
    elif any(w in text_lower for w in ["significant reasoning", "complex reasoning", "multi-step reasoning"]):
        features["logical_reasoning_required"] = 4
    elif any(w in text_lower for w in ["extensive reasoning", "deep reasoning"]):
        features["logical_reasoning_required"] = 5

    # Infer atypicality from descriptions
    if any(w in text_lower for w in ["common pattern", "common bug", "typical bug", "very common"]):
        features["atypicality"] = 1
    elif any(w in text_lower for w in ["relatively common", "standard pattern"]):
        features["atypicality"] = 2
    elif any(w in text_lower for w in ["moderately unusual", "somewhat unusual"]):
        features["atypicality"] = 3
    elif any(w in text_lower for w in ["unusual", "uncommon", "atypical"]):
        features["atypicality"] = 4
    elif any(w in text_lower for w in ["very unusual", "rare", "novel"]):
        features["atypicality"] = 5

    # Infer reproduction_steps
    if any(w in text_lower for w in ["reproduction steps", "steps to reproduce", "repro steps", "complete reproduction"]):
        features["reproduction_steps"] = 1
    elif any(w in text_lower for w in ["no reproduction", "without steps"]):
        features["reproduction_steps"] = 0

    # Infer error_message_provided
    if any(w in text_lower for w in ["error message", "traceback", "exception", "error output", "clear error"]):
        features["error_message_provided"] = 1
    elif any(w in text_lower for w in ["no error message", "without error"]):
        features["error_message_provided"] = 0

    # Infer reproduction_steps from mentions
    if any(w in text_lower for w in ["reproduction case", "having a reproduction", "reproduction steps"]):
        features["reproduction_steps"] = 1

    # Extract patch_line_count from "~X lines" pattern
    patch_match = re.search(r"~(\d+)\s*lines", text_lower)
    if patch_match:
        patch_lines = int(patch_match.group(1))
        if 1 <= patch_lines <= 1000:
            features["patch_line_count"] = patch_lines

    # Extract patch_file_count from "X files" pattern
    files_match = re.search(r"(\d+)\s*files?", text_lower)
    if files_match:
        num_files = int(files_match.group(1))
        if 1 <= num_files <= 50:
            features["patch_file_count"] = num_files

    # Use score to infer overall difficulty if we don't have fix_complexity
    if score is not None and "fix_complexity" not in features:
        # Score is 0-1, map to 1-5
        features["fix_complexity"] = max(1, min(5, int(round(1 + score * 4))))

    return features


def parse_raw_response(raw_data: dict, task_id: str) -> Optional[Dict]:
    """Parse a raw Lunette response to extract features.

    Args:
        raw_data: The raw response dict from Lunette
        task_id: The task instance ID

    Returns:
        Feature dict or None if extraction failed
    """
    features = {}

    # Get the response
    if "raw_response" not in raw_data:
        return None

    response = raw_data["raw_response"]

    if not isinstance(response, dict):
        return None

    score = response.get("score", None)
    explanation = response.get("explanation", "")

    # FIRST: Check if features are nested under a "features" key
    if "features" in response and isinstance(response["features"], dict):
        count = extract_features_from_dict(response["features"], features)
        if count >= 5:
            if score is not None and isinstance(score, (int, float)):
                features["lunette_score"] = score
            features["_instance_id"] = task_id
            features["_run_id"] = raw_data.get("run_id", "unknown")
            features["_extraction_method"] = "postprocess_nested"
            features["_features_extracted"] = count
            return features

    # SECOND: Check if features are directly in the response dict
    count = extract_features_from_dict(response, features)
    if count >= 5:
        if score is not None and isinstance(score, (int, float)):
            features["lunette_score"] = score
        features["_instance_id"] = task_id
        features["_run_id"] = raw_data.get("run_id", "unknown")
        features["_extraction_method"] = "postprocess_direct"
        features["_features_extracted"] = count
        return features

    # THIRD: Try to extract numeric values from explanation text
    features = {}  # Reset
    for feature_name, spec in FEATURE_SPECS.items():
        value = extract_feature_value(explanation, feature_name, spec)
        if value is not None:
            features[feature_name] = value

    # FOURTH: Infer features from qualitative descriptions in text
    inferred = infer_features_from_text(explanation, score)
    for k, v in inferred.items():
        if k not in features:  # Don't overwrite explicit values
            features[k] = v

    # Add score if present
    if score is not None and isinstance(score, (int, float)):
        features["lunette_score"] = score

    # Return if we have at least some useful features
    semantic_features = ["fix_in_description", "domain_knowledge_required",
                        "fix_complexity", "problem_clarity", "atypicality",
                        "logical_reasoning_required", "fix_locality"]
    has_semantic = sum(1 for f in semantic_features if f in features) >= 1

    # Also accept if we have score + at least 1 semantic feature or patch info
    has_useful = has_semantic or ("lunette_score" in features and
                                   ("patch_line_count" in features or "patch_file_count" in features))

    if has_useful:
        features["_instance_id"] = task_id
        features["_run_id"] = raw_data.get("run_id", "unknown")
        features["_extraction_method"] = "postprocess_inferred"
        features["_features_extracted"] = len([k for k in features if not k.startswith("_")])
        return features

    return None


def postprocess_all(features_dir: Path, dry_run: bool = False) -> Tuple[int, int, List[Dict]]:
    """Post-process all raw response files.

    Args:
        features_dir: Directory containing raw JSON files
        dry_run: If True, don't save files

    Returns:
        Tuple of (processed_count, success_count, all_features)
    """
    raw_files = list(features_dir.glob("*_raw.json"))
    print(f"Found {len(raw_files)} raw files to process")

    processed = 0
    success = 0
    all_features = []

    for raw_file in sorted(raw_files):
        task_id = raw_file.stem.replace("_raw", "")

        # Skip if we already have a parsed feature file
        feature_file = features_dir / f"{task_id}.json"
        if feature_file.exists():
            print(f"  {task_id}: Already has feature file, skipping")
            continue

        processed += 1

        with open(raw_file) as f:
            raw_data = json.load(f)

        features = parse_raw_response(raw_data, task_id)

        if features:
            success += 1
            all_features.append(features)

            n_features = features.get("_features_extracted", 0)
            print(f"  {task_id}: Extracted {n_features} features")

            if not dry_run:
                # Save to feature file
                with open(feature_file, "w") as f:
                    json.dump(features, f, indent=2)
        else:
            print(f"  {task_id}: Could not extract features")

    return processed, success, all_features


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Lunette raw responses to extract features"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be extracted without saving",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="chris_output/experiment_a/lunette_features",
        help="Directory containing raw response files",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    if not features_dir.exists():
        print(f"Error: Directory not found: {features_dir}")
        return

    print(f"Post-processing raw responses in: {features_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    processed, success, all_features = postprocess_all(features_dir, args.dry_run)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Raw files processed: {processed}")
    print(f"Successfully extracted: {success}")
    print(f"Success rate: {success/processed*100:.1f}%" if processed > 0 else "N/A")

    if all_features:
        # Show feature coverage
        print()
        print("Feature coverage across extracted tasks:")
        feature_counts = {}
        for f in all_features:
            for key in f:
                if not key.startswith("_"):
                    feature_counts[key] = feature_counts.get(key, 0) + 1

        for name, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
            pct = count / len(all_features) * 100
            print(f"  {name}: {count}/{len(all_features)} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
