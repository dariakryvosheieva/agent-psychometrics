"""
LLM-as-judge for extracting discrete features from SWE-bench tasks.

This script uses an LLM to analyze task descriptions and patches to extract
features that may predict IRT difficulty. It includes correlation analysis
with ground-truth IRT difficulty.

Usage:
    python llm_judge/llm_judge.py --num_tasks 10 --output_path chris_output/llm_judge/features.csv
    python llm_judge/llm_judge.py --task_id django__django-13658  # Single task
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from experiment_ab_shared.llm_judge import LLMApiClient, parse_llm_response


# Feature names for this prompt (used for correlation analysis)
FEATURE_COLS = [
    'fix_in_description', 'patch_matches_suggestion', 'problem_clarity',
    'error_message_provided', 'reproduction_steps', 'fix_locality',
    'domain_knowledge_required', 'fix_complexity'
]

JUDGE_PROMPT = """You are an expert software engineer analyzing a GitHub issue and its solution patch.

Your task is to evaluate specific features of this issue that might predict how difficult it would be for an AI coding agent to solve.

## Issue Information

**Task ID:** {task_id}
**Repository:** {repo}

**Problem Statement:**
{problem_statement}

**Gold Patch (the correct solution):**
```diff
{patch}
```

## Features to Evaluate

Please evaluate each feature and respond with a JSON object. Be precise and consistent.

1. **fix_in_description** (0-3): Does the problem statement contain or suggest the fix?
   - 0: No hint at the solution
   - 1: Vague hint or direction
   - 2: Clear description of what needs to change
   - 3: Exact code fix provided in the description

2. **patch_matches_suggestion** (0-2): If a fix is suggested, does the gold patch match it?
   - 0: No suggestion in description, or suggestion is wrong/different from patch
   - 1: Suggestion is partially correct or in the right direction
   - 2: Gold patch is essentially the suggested fix

3. **problem_clarity** (1-5): How clear and well-specified is the problem?
   - 1: Very vague, unclear what's wrong
   - 2: Somewhat unclear, missing important details
   - 3: Reasonably clear but some ambiguity
   - 4: Clear problem statement with good details
   - 5: Crystal clear with reproduction steps and expected behavior

4. **error_message_provided** (0/1): Does the problem include an error message or traceback?
   - 0: No
   - 1: Yes

5. **reproduction_steps** (0/1): Are concrete reproduction steps provided?
   - 0: No
   - 1: Yes

6. **fix_locality** (1-3): How localized is the fix based on the patch?
   - 1: Single location, few lines changed
   - 2: Multiple locations in same file, or moderate changes
   - 3: Multiple files or significant changes

7. **domain_knowledge_required** (1-5): How much specialized knowledge is needed?
   - 1: Basic Python, obvious fix
   - 2: Standard library knowledge
   - 3: Framework-specific knowledge (Django, pytest, etc.)
   - 4: Deep framework internals or complex algorithms
   - 5: Obscure APIs, protocols, or very specialized knowledge

8. **fix_complexity** (1-5): How complex is the actual fix?
   - 1: Trivial (add parameter, change value, simple one-liner)
   - 2: Simple (straightforward logic change)
   - 3: Moderate (requires understanding context)
   - 4: Complex (multiple interacting changes)
   - 5: Very complex (architectural changes, subtle edge cases)

Respond with ONLY a JSON object in this exact format:
{{
    "fix_in_description": <0-3>,
    "patch_matches_suggestion": <0-2>,
    "problem_clarity": <1-5>,
    "error_message_provided": <0 or 1>,
    "reproduction_steps": <0 or 1>,
    "fix_locality": <1-3>,
    "domain_knowledge_required": <1-5>,
    "fix_complexity": <1-5>,
    "reasoning": "<brief explanation of your ratings, 2-3 sentences>"
}}
"""


def judge_task(
    task_id: str,
    repo: str,
    problem_statement: str,
    patch: str,
    client: LLMApiClient,
) -> dict:
    """Judge a single task and return features."""

    prompt = JUDGE_PROMPT.format(
        task_id=task_id,
        repo=repo,
        problem_statement=problem_statement[:8000],  # Truncate if too long
        patch=patch[:4000],
    )

    response_text = client.call(prompt)
    result = parse_llm_response(response_text, expected_features=FEATURE_COLS)

    if result is None:
        raise ValueError(f"Failed to parse response for task {task_id}")

    result["task_id"] = task_id
    result["model"] = client.model
    return result


def load_tasks() -> pd.DataFrame:
    """Load tasks with IRT difficulty."""
    from datasets import load_dataset

    # Load IRT items
    items = pd.read_csv("clean_data/swebench_verified_20250930_full/1d/items.csv", index_col=0)

    # Load SWE-bench from Hugging Face
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    swebench_df = pd.DataFrame({
        'instance_id': ds['instance_id'],
        'repo': ds['repo'],
        'problem_statement': ds['problem_statement'],
        'patch': ds['patch'],
    }).set_index('instance_id')

    merged = swebench_df.join(items[['b', 'a']], how='inner')
    return merged


def main():
    parser = argparse.ArgumentParser(description='LLM-as-judge for task features')
    parser.add_argument('--provider', type=str, default='anthropic',
                        choices=['anthropic', 'openai'],
                        help='LLM provider to use')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to use (default: claude-sonnet-4-20250514 or gpt-4o)')
    parser.add_argument('--num_tasks', type=int, default=None,
                        help='Number of tasks to judge (samples across difficulty range)')
    parser.add_argument('--task_id', type=str, default=None,
                        help='Judge a specific task by ID')
    parser.add_argument('--output_path', type=str,
                        default='chris_output/llm_judge/features.csv',
                        help='Output path for results')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between API calls (seconds)')
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create API client
    client = LLMApiClient(provider=args.provider, model=args.model)
    print(f"Using {client.get_info()}")

    # Load tasks
    print("Loading tasks...")
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks")

    # Select tasks to judge
    if args.task_id:
        if args.task_id not in tasks.index:
            raise ValueError(f"Task {args.task_id} not found")
        task_ids = [args.task_id]
    elif args.num_tasks:
        # Sample across difficulty range
        tasks_sorted = tasks.sort_values('b')
        step = len(tasks_sorted) // args.num_tasks
        indices = list(range(0, len(tasks_sorted), step))[:args.num_tasks]
        task_ids = tasks_sorted.iloc[indices].index.tolist()
    else:
        task_ids = tasks.index.tolist()

    print(f"Judging {len(task_ids)} tasks with {args.provider}...")

    # Load existing results if any
    results = []
    if output_path.exists():
        existing = pd.read_csv(output_path)
        existing_ids = set(existing['task_id'].tolist())
        task_ids = [tid for tid in task_ids if tid not in existing_ids]
        results = existing.to_dict('records')
        print(f"Found {len(existing_ids)} existing results, {len(task_ids)} remaining")

    # Judge tasks
    for i, task_id in enumerate(task_ids):
        row = tasks.loc[task_id]
        print(f"[{i+1}/{len(task_ids)}] Judging {task_id} (b={row['b']:.2f})...")

        try:
            result = judge_task(
                task_id=task_id,
                repo=row['repo'],
                problem_statement=row['problem_statement'],
                patch=row['patch'],
                client=client,
            )
            result['irt_difficulty'] = row['b']
            results.append(result)

            # Save incrementally
            pd.DataFrame(results).to_csv(output_path, index=False)

        except Exception as e:
            print(f"  Error: {e}")
            continue

        if args.delay and i < len(task_ids) - 1:
            time.sleep(args.delay)

    # Final save and summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} results to {output_path}")

    # Print correlation with IRT difficulty
    if len(results_df) > 5:
        print("\n" + "=" * 60)
        print("CORRELATION WITH IRT DIFFICULTY")
        print("=" * 60)

        for col in FEATURE_COLS:
            if col in results_df.columns:
                corr = results_df['irt_difficulty'].corr(results_df[col])
                print(f"  {col:30s}: r = {corr:+.3f}")


if __name__ == "__main__":
    main()
