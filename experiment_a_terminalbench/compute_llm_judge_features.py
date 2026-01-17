"""Compute LLM judge semantic features for Experiment A - TerminalBench.

This script uses direct LLM API calls (Anthropic/OpenAI) to extract semantic
features from TerminalBench tasks. It analyzes task instructions and reference
solutions to predict task difficulty.

Usage:
    # Dry run to see execution plan and cost estimate
    python -m experiment_a_terminalbench.compute_llm_judge_features --dry_run

    # Run on small subset for validation
    python -m experiment_a_terminalbench.compute_llm_judge_features --limit 10

    # Full run with GPT-5.2 (all 89 tasks)
    python -m experiment_a_terminalbench.compute_llm_judge_features --provider openai --model gpt-5.2

    # Use Claude instead
    python -m experiment_a_terminalbench.compute_llm_judge_features --provider anthropic --model claude-sonnet-4-20250514
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a_terminalbench.data_loader import load_task_list_from_items
from experiment_a_terminalbench.llm_judge_prompt import (
    format_llm_judge_prompt,
    LLM_JUDGE_SEMANTIC_FEATURES,
)


# Try to import API clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Output directory
OUTPUT_DIR = ROOT / "chris_output" / "experiment_a_terminalbench" / "llm_judge_features"


def parse_llm_response(text: str) -> Optional[dict]:
    """Parse the LLM response to extract semantic features."""
    if not text:
        return None

    # Try to extract JSON from response
    # First try: look for ```json block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Second try: look for any JSON object with our features
        json_match = re.search(r'\{[^{}]*"solution_in_instruction"[^{}]*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

    try:
        data = json.loads(text)

        # Validate required fields (at least some of the semantic features)
        has_features = any(f in data for f in LLM_JUDGE_SEMANTIC_FEATURES)
        if not has_features:
            return None

        return data
    except json.JSONDecodeError:
        return None


def call_anthropic(prompt: str, model: str = "claude-opus-4-5-20251101") -> str:
    """Call Anthropic API and return response text."""
    if not HAS_ANTHROPIC:
        raise ImportError("anthropic package not installed")

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


def call_openai(prompt: str, model: str = "gpt-5.2") -> str:
    """Call OpenAI API and return response text."""
    if not HAS_OPENAI:
        raise ImportError("openai package not installed")

    client = openai.OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text.strip()


def extract_features_for_task(
    task: dict,
    provider: str = "openai",
    model: Optional[str] = None,
) -> Optional[dict]:
    """Call LLM to extract semantic features from a task.

    Args:
        task: TerminalBench task dict with task_id, instruction, solution, etc.
        provider: "anthropic" or "openai"
        model: Specific model to use (default: gpt-5.2 or claude-opus-4-5-20251101)

    Returns:
        Parsed feature dict or None if failed
    """
    prompt = format_llm_judge_prompt(
        task_id=task["task_id"],
        instruction=task["instruction"],
        solution=task["solution"],
        category=task.get("category", ""),
        tags=task.get("tags", []),
        claimed_difficulty=task.get("claimed_difficulty", ""),
    )

    try:
        if provider == "anthropic":
            model = model or "claude-opus-4-5-20251101"
            response_text = call_anthropic(prompt, model)
        elif provider == "openai":
            model = model or "gpt-5.2"
            response_text = call_openai(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        features = parse_llm_response(response_text)
        if features:
            features["_task_id"] = task["task_id"]
            features["_model"] = model
            features["_provider"] = provider
            features["_extracted_at"] = datetime.now().isoformat()
        return features

    except Exception as e:
        print(f"    Error calling LLM: {e}")
        return None


def aggregate_to_csv(output_dir: Path) -> Path:
    """Aggregate individual JSON files into a single CSV."""
    rows = []

    for json_file in output_dir.glob("*.json"):
        if json_file.name.startswith("compute_stats"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            rows.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    if not rows:
        print("No feature files found to aggregate")
        return None

    df = pd.DataFrame(rows)

    # Rename _task_id to task_id for compatibility with train_evaluate.py
    if "_task_id" in df.columns:
        df = df.rename(columns={"_task_id": "task_id"})

    csv_path = output_dir / "llm_judge_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Aggregated {len(rows)} tasks to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Compute LLM judge semantic features for Experiment A - TerminalBench")
    parser.add_argument("--dry_run", action="store_true", help="Show plan without running")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to process")
    parser.add_argument("--task_ids", nargs="+", help="Specific task IDs to process")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                       help="Skip tasks with existing features")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["anthropic", "openai"],
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model to use (default: gpt-5.2)")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls in seconds (default: 0.5)")
    parser.add_argument("--aggregate_only", action="store_true",
                       help="Only aggregate existing JSON files to CSV")
    parser.add_argument("--items_path", type=str,
                       default="chris_output/terminal_bench_2.0_binomial_1pl/1d/items.csv",
                       help="Path to items.csv")
    parser.add_argument("--repo_path", type=str,
                       default="terminal-bench",
                       help="Path to terminal-bench repo")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate only mode
    if args.aggregate_only:
        aggregate_to_csv(OUTPUT_DIR)
        return

    # Resolve paths
    items_path = ROOT / args.items_path
    repo_path = ROOT / args.repo_path

    # Load tasks using shared function from data_loader
    print(f"Loading tasks from: {items_path}")
    print(f"Terminal-bench repo: {repo_path}")
    tasks = load_task_list_from_items(items_path, repo_path)
    print(f"Loaded {len(tasks)} tasks")

    # Filter to specific tasks if requested
    if args.task_ids:
        task_ids_set = set(args.task_ids)
        tasks = [t for t in tasks if t["task_id"] in task_ids_set]
        print(f"Filtered to {len(tasks)} specified tasks")

    # Apply limit
    if args.limit and len(tasks) > args.limit:
        tasks = tasks[:args.limit]
        print(f"Limited to {args.limit} tasks")

    # Filter out existing
    if args.skip_existing:
        original_count = len(tasks)
        tasks = [
            t for t in tasks
            if not (OUTPUT_DIR / f"{t['task_id']}.json").exists()
        ]
        skipped = original_count - len(tasks)
        if skipped > 0:
            print(f"Skipping {skipped} existing, {len(tasks)} remaining")

    print(f"\nTotal tasks to process: {len(tasks)}")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"\nProvider: {args.provider}")
        model = args.model or ("gpt-5.2" if args.provider == "openai" else "claude-opus-4-5-20251101")
        print(f"Model: {model}")

        # Show sample of tasks
        print(f"\nSample tasks (first 10):")
        for task in tasks[:10]:
            print(f"  {task['task_id']} ({task.get('category', 'N/A')})")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")

        # Estimate cost
        if args.provider == "openai":
            cost_per_call = 0.01  # GPT-5.2 rough estimate
        else:
            if "opus" in model.lower():
                cost_per_call = 0.015  # Claude Opus rough estimate
            else:
                cost_per_call = 0.003  # Claude Sonnet rough estimate

        print(f"\nEstimated cost: ~${len(tasks) * cost_per_call:.2f} ({len(tasks)} tasks × ${cost_per_call}/task)")
        return

    # Process tasks
    stats = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
    }

    for i, task in enumerate(tasks):
        output_file = OUTPUT_DIR / f"{task['task_id']}.json"

        print(f"[{i+1}/{len(tasks)}] {task['task_id']}...")

        # Extract features
        features = extract_features_for_task(
            task=task,
            provider=args.provider,
            model=args.model,
        )

        if features:
            # Save features
            with open(output_file, "w") as f:
                json.dump(features, f, indent=2)
            stats["success"] += 1
            # Show key features
            sol_desc = features.get("solution_in_instruction", "?")
            task_comp = features.get("task_complexity", "?")
            print(f"    -> solution_in_instruction: {sol_desc}, task_complexity: {task_comp}")
        else:
            stats["failed"] += 1
            print(f"    Failed to extract features")

        # Rate limiting
        if args.delay and i < len(tasks) - 1:
            time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total processed: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")

    # Save stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = OUTPUT_DIR / f"compute_stats_{timestamp}.json"
    with open(stats_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "provider": args.provider,
            "model": args.model or ("gpt-5.2" if args.provider == "openai" else "claude-opus-4-5-20251101"),
            "stats": stats,
        }, f, indent=2)
    print(f"\nStats saved to: {stats_file}")

    # Aggregate to CSV
    aggregate_to_csv(OUTPUT_DIR)


if __name__ == "__main__":
    main()
