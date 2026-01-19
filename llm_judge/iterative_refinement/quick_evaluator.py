"""Quick evaluation on stratified task samples for iterative refinement.

Evaluates prompt variants cheaply on n=30 tasks stratified by difficulty terciles
(10 easy, 10 medium, 10 hard). Uses the OpenAI Responses API with automatic
prefix caching - task content is placed first (cached) and instructions second.
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from scipy.stats import pearsonr

from llm_judge.iterative_refinement.config import IterativeRefinementConfig
from llm_judge.iterative_refinement.feature_metrics import (
    compute_entropy,
    compute_pairwise_correlations,
    find_redundant_features,
)
from llm_judge.iterative_refinement.prompt_format import format_cacheable_prompt
from llm_judge.iterative_refinement.prompt_store import FeatureDefinition


@dataclass
class QuickEvalResult:
    """Results from quick evaluation on stratified sample."""

    # Core metrics
    pearson_r: Optional[float]  # Correlation with ground truth difficulty
    p_value: Optional[float]

    # Feature quality metrics
    feature_entropies: Dict[str, float]  # Entropy per feature
    low_entropy_features: List[str]  # Features with entropy < threshold
    redundant_pairs: List[Tuple[str, str, float]]  # Highly correlated pairs

    # Evaluation details
    n_tasks: int
    n_successful: int
    features_df: Optional[pd.DataFrame]  # Extracted features
    predictions: Optional[Dict[str, float]]  # Task -> predicted difficulty

    # Cost tracking
    input_tokens: int
    output_tokens: int
    estimated_cost: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "pearson_r": self.pearson_r,
            "p_value": self.p_value,
            "feature_entropies": self.feature_entropies,
            "low_entropy_features": self.low_entropy_features,
            "redundant_pairs": [
                {"f1": p[0], "f2": p[1], "r": p[2]} for p in self.redundant_pairs
            ],
            "n_tasks": self.n_tasks,
            "n_successful": self.n_successful,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": self.estimated_cost,
        }


def stratified_sample_tasks(
    items_df: pd.DataFrame,
    n_per_tercile: int = 10,
    seed: int = 42,
) -> List[str]:
    """Select stratified sample of tasks by difficulty terciles.

    Args:
        items_df: DataFrame with index=task_id, columns include 'b' (difficulty)
        n_per_tercile: Number of tasks per tercile (easy, medium, hard)
        seed: Random seed for reproducibility

    Returns:
        List of task_ids (10 easy + 10 medium + 10 hard = 30 by default)
    """
    rng = np.random.RandomState(seed)

    # Sort by difficulty
    sorted_items = items_df.sort_values("b")
    n_tasks = len(sorted_items)
    tercile_size = n_tasks // 3

    # Split into terciles
    easy_tasks = list(sorted_items.index[:tercile_size])
    medium_tasks = list(sorted_items.index[tercile_size : 2 * tercile_size])
    hard_tasks = list(sorted_items.index[2 * tercile_size :])

    # Sample from each
    sampled = []
    for tasks, name in [
        (easy_tasks, "easy"),
        (medium_tasks, "medium"),
        (hard_tasks, "hard"),
    ]:
        n_sample = min(n_per_tercile, len(tasks))
        indices = rng.choice(len(tasks), n_sample, replace=False)
        sampled.extend([tasks[i] for i in indices])

    return sampled


def load_swebench_tasks(task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load SWE-bench task data for the given task IDs.

    Args:
        task_ids: List of task IDs to load

    Returns:
        Dict mapping task_id -> task data dict
    """
    from datasets import load_dataset

    # Load SWE-bench Verified
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # Index by instance_id
    tasks = {}
    for row in ds:
        if row["instance_id"] in task_ids:
            tasks[row["instance_id"]] = {
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "version": row.get("version", ""),
                "problem_statement": row["problem_statement"],
                "patch": row["patch"],
                "FAIL_TO_PASS": row.get("FAIL_TO_PASS", ""),
                "PASS_TO_PASS": row.get("PASS_TO_PASS", ""),
                "hints_text": row.get("hints_text", ""),
            }

    return tasks


async def extract_features_batch(
    client: AsyncOpenAI,
    tasks: Dict[str, Dict[str, Any]],
    feature_instructions: str,
    feature_names: List[str],
    model: str = "gpt-5.2",
    max_concurrent: int = 10,
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """Extract features for multiple tasks using async API calls.

    Uses the OpenAI Responses API with automatic prefix caching.
    Task content is placed first (cached), instructions second (varies).

    Args:
        client: AsyncOpenAI client
        tasks: Dict mapping task_id -> task data
        feature_instructions: The feature extraction instructions
        feature_names: List of feature names to extract
        model: Model ID
        max_concurrent: Maximum concurrent requests

    Returns:
        Tuple of (results_dict, total_input_tokens, total_output_tokens)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    total_input = 0
    total_output = 0

    async def process_task(task_id: str, task_data: Dict[str, Any]):
        nonlocal total_input, total_output

        async with semaphore:
            prompt = format_cacheable_prompt(task_data, feature_instructions)

            try:
                response = await client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=500,
                )

                total_input += response.usage.input_tokens
                total_output += response.usage.output_tokens

                # Parse JSON response
                text = response.output_text.strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                features = json.loads(text)

                # Validate features
                valid_features = {}
                for name in feature_names:
                    if name in features:
                        try:
                            valid_features[name] = float(features[name])
                        except (ValueError, TypeError):
                            pass

                results[task_id] = {
                    "features": valid_features,
                    "reasoning": features.get("reasoning", ""),
                    "success": True,
                }

            except Exception as e:
                results[task_id] = {
                    "features": {},
                    "error": str(e),
                    "success": False,
                }

    # Run all tasks concurrently
    await asyncio.gather(*[process_task(tid, tdata) for tid, tdata in tasks.items()])

    return results, total_input, total_output


def compute_difficulty_predictions(
    features_df: pd.DataFrame,
    ground_truth: pd.Series,
    feature_names: List[str],
) -> Dict[str, float]:
    """Fit simple Ridge regression and return predictions.

    Args:
        features_df: DataFrame with features for each task
        ground_truth: Series with ground truth difficulties
        feature_names: Features to use

    Returns:
        Dict mapping task_id -> predicted difficulty
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Align data
    common_tasks = features_df.index.intersection(ground_truth.index)
    X = features_df.loc[common_tasks, feature_names].values
    y = ground_truth.loc[common_tasks].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Scale and fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Predict
    preds = model.predict(X_scaled)
    return dict(zip(common_tasks, preds.tolist()))


async def run_quick_evaluation(
    feature_definitions: List[FeatureDefinition],
    config: IterativeRefinementConfig,
    task_ids: Optional[List[str]] = None,
    ground_truth_items: Optional[pd.DataFrame] = None,
) -> QuickEvalResult:
    """Run quick evaluation on stratified task sample.

    Args:
        feature_definitions: List of feature definitions to evaluate
        config: Configuration
        task_ids: Optional pre-selected task IDs (otherwise stratified sample)
        ground_truth_items: Optional pre-loaded items DataFrame

    Returns:
        QuickEvalResult with metrics and diagnostics
    """
    # Load ground truth if not provided
    if ground_truth_items is None:
        ground_truth_items = pd.read_csv(config.items_path, index_col=0)

    # Select stratified sample if task_ids not provided
    if task_ids is None:
        task_ids = stratified_sample_tasks(
            ground_truth_items,
            n_per_tercile=config.tasks_per_tercile,
            seed=42,
        )

    # Load task data
    tasks = load_swebench_tasks(task_ids)
    print(f"Loaded {len(tasks)} / {len(task_ids)} tasks")

    # Build feature instructions
    from llm_judge.iterative_refinement.prompt_store import generate_prompt_from_schema

    feature_instructions = generate_prompt_from_schema(feature_definitions)
    feature_names = [f.name for f in feature_definitions]

    # Extract features
    client = AsyncOpenAI()
    results, input_tokens, output_tokens = await extract_features_batch(
        client=client,
        tasks=tasks,
        feature_instructions=feature_instructions,
        feature_names=feature_names,
        model=config.model,
    )

    # Build features DataFrame
    rows = []
    for task_id, result in results.items():
        if result["success"] and result["features"]:
            row = {"task_id": task_id}
            row.update(result["features"])
            rows.append(row)

    n_successful = len(rows)

    if n_successful < 5:
        # Not enough data
        return QuickEvalResult(
            pearson_r=None,
            p_value=None,
            feature_entropies={},
            low_entropy_features=[],
            redundant_pairs=[],
            n_tasks=len(task_ids),
            n_successful=n_successful,
            features_df=None,
            predictions=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=_estimate_cost(input_tokens, output_tokens, config.model),
        )

    features_df = pd.DataFrame(rows).set_index("task_id")

    # Compute feature entropies
    feature_entropies = {}
    for name in feature_names:
        if name in features_df.columns:
            values = features_df[name].dropna().values
            if len(values) > 0:
                feature_entropies[name] = compute_entropy(values)

    # Find low entropy features
    low_entropy_features = [
        name
        for name, entropy in feature_entropies.items()
        if entropy < config.entropy_threshold
    ]

    # Find redundant feature pairs
    corr_matrix = compute_pairwise_correlations(features_df[feature_names])
    redundant_pairs = find_redundant_features(corr_matrix, config.redundancy_threshold)

    # Compute correlation with ground truth
    common_tasks = features_df.index.intersection(ground_truth_items.index)
    if len(common_tasks) >= 5:
        # Simple average of features as baseline predictor
        predictions = compute_difficulty_predictions(
            features_df, ground_truth_items["b"], feature_names
        )

        # Compute Pearson r
        pred_values = [predictions[t] for t in common_tasks if t in predictions]
        true_values = ground_truth_items.loc[
            [t for t in common_tasks if t in predictions], "b"
        ].values

        if len(pred_values) >= 5:
            r, p = pearsonr(pred_values, true_values)
            pearson_r = float(r)
            p_value = float(p)
        else:
            pearson_r = None
            p_value = None
            predictions = None
    else:
        pearson_r = None
        p_value = None
        predictions = None

    return QuickEvalResult(
        pearson_r=pearson_r,
        p_value=p_value,
        feature_entropies=feature_entropies,
        low_entropy_features=low_entropy_features,
        redundant_pairs=redundant_pairs,
        n_tasks=len(task_ids),
        n_successful=n_successful,
        features_df=features_df,
        predictions=predictions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=_estimate_cost(input_tokens, output_tokens, config.model),
    )


def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate API cost in USD.

    GPT-5.2 pricing (approximate):
    - Input: $0.50 / 1M tokens
    - Output: $2.00 / 1M tokens
    - Cached input: $0.05 / 1M tokens (90% discount)
    """
    # Assume ~50% cache hit rate for iterative refinement
    input_cost = input_tokens * 0.275 / 1_000_000  # Blended rate
    output_cost = output_tokens * 2.00 / 1_000_000
    return round(input_cost + output_cost, 4)


def run_quick_evaluation_sync(
    feature_definitions: List[FeatureDefinition],
    config: IterativeRefinementConfig,
    task_ids: Optional[List[str]] = None,
    ground_truth_items: Optional[pd.DataFrame] = None,
) -> QuickEvalResult:
    """Synchronous wrapper for run_quick_evaluation."""
    return asyncio.run(
        run_quick_evaluation(
            feature_definitions, config, task_ids, ground_truth_items
        )
    )