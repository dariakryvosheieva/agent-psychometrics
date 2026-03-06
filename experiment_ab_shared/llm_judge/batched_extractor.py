"""Batched feature extraction with per-level info isolation and prefix caching.

Groups features by InfoLevel, builds a shared cacheable prefix per level,
then splits into batches of <=7 features per API call. Merges all batch
results into a single dict per task.

Usage:
    from experiment_ab_shared.llm_judge.batched_extractor import BatchedFeatureExtractor
    from experiment_ab_shared.llm_judge.task_context import get_task_context

    ctx = get_task_context("swebench_verified")
    extractor = BatchedFeatureExtractor(
        feature_names=["solution_hint", "problem_clarity", "solution_complexity"],
        task_context=ctx,
        provider="openai",
    )
    results = extractor.run(tasks, output_dir=Path("output/"))
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

import pandas as pd

from experiment_ab_shared.llm_judge.api_client import LLMApiClient
from experiment_ab_shared.llm_judge.feature_registry import get_features
from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition, InfoLevel
from experiment_ab_shared.llm_judge.response_parser import parse_llm_response
from experiment_ab_shared.llm_judge.task_context import TaskContext

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 7


def _build_suffix(
    features: List[FeatureDefinition],
    scale_variant: str,
) -> str:
    """Build the suffix (non-cached part) for a batch of features.

    Contains the feature scale descriptions and JSON output format instruction.
    """
    parts = ["## FEATURES TO EVALUATE\n"]
    parts.append(
        "Analyze the task information above to evaluate these features. "
        "Be precise and consistent with your ratings.\n"
    )

    for feat in features:
        parts.append(feat.get_scale_text(scale_variant))
        parts.append("")  # blank line between features

    # Build output format
    json_example = ", ".join(
        f'"{feat.name}": <{feat.min_value}-{feat.max_value}>'
        for feat in features
    )

    parts.append("## OUTPUT FORMAT\n")
    parts.append(
        "Respond with ONLY a JSON object containing your ratings. "
        "Do not include any explanation or commentary outside the JSON.\n"
    )
    parts.append(f"```json\n{{{json_example}}}\n```")

    return "\n".join(parts)


def _group_by_level(
    features: List[FeatureDefinition],
) -> Dict[InfoLevel, List[FeatureDefinition]]:
    """Group features by their info level, preserving order."""
    groups: Dict[InfoLevel, List[FeatureDefinition]] = defaultdict(list)
    for feat in features:
        groups[feat.info_level].append(feat)
    return dict(groups)


def _batch(features: List[FeatureDefinition], size: int) -> List[List[FeatureDefinition]]:
    """Split features into batches of at most `size`."""
    return [features[i : i + size] for i in range(0, len(features), size)]


class BatchedFeatureExtractor:
    """Extract features in batches grouped by info level.

    Features are grouped by InfoLevel so each group's prompt contains only
    the appropriate task data. Within each level, features are split into
    batches of <=7 for manageable LLM output. Batches within the same level
    share a prefix (cacheable).

    Environment-level features are skipped (handled by auditor agent pipeline).
    """

    def __init__(
        self,
        feature_names: List[str],
        task_context: TaskContext,
        provider: str = "openai",
        model: Optional[str] = None,
        batch_size: int = MAX_BATCH_SIZE,
    ):
        self.features = get_features(feature_names)
        self.task_context = task_context
        self.client = LLMApiClient(provider, model)
        self.batch_size = batch_size

        # Group and validate
        self.level_groups = _group_by_level(self.features)

        if InfoLevel.ENVIRONMENT in self.level_groups:
            env_names = [f.name for f in self.level_groups[InfoLevel.ENVIRONMENT]]
            raise ValueError(
                f"Environment-level features cannot be extracted via BatchedFeatureExtractor "
                f"(they require the auditor agent pipeline): {env_names}"
            )

        for level in self.level_groups:
            if level not in self.task_context.system_intros:
                raise ValueError(
                    f"TaskContext '{self.task_context.name}' does not support "
                    f"InfoLevel.{level.name}, needed for features: "
                    f"{[f.name for f in self.level_groups[level]]}"
                )

    # =========================================================================
    # Core extraction logic (shared by sync and async paths)
    # =========================================================================

    def _build_batch_calls(
        self, task: Dict[str, Any]
    ) -> List[tuple]:
        """Plan all API calls for a task: list of (prefix, suffix, batch_features).

        Returns calls ordered by info level, batches sequential within level.
        """
        calls = []
        for level in (InfoLevel.PROBLEM, InfoLevel.TEST, InfoLevel.SOLUTION):
            level_features = self.level_groups.get(level)
            if not level_features:
                continue
            prefix = self.task_context.build_prefix(task, level)
            for batch_features in _batch(level_features, self.batch_size):
                suffix = _build_suffix(batch_features, self.task_context.scale_variant)
                calls.append((prefix, suffix, batch_features))
        return calls

    def _merge_batch_result(
        self,
        merged: Dict[str, Any],
        parsed: Optional[Dict[str, Any]],
        batch_features: List[FeatureDefinition],
        task_id: str,
    ) -> None:
        """Merge a single batch's parsed response into the merged dict."""
        if parsed is None:
            logger.warning(
                f"Failed to parse batch response for task {task_id}, "
                f"features: {[f.name for f in batch_features]}"
            )
            return

        for feat in batch_features:
            if feat.name in parsed:
                value = parsed[feat.name]
                if feat.validate(value):
                    merged[feat.name] = value
                else:
                    logger.warning(
                        f"Invalid value {value} for '{feat.name}' "
                        f"(expected {feat.min_value}-{feat.max_value}), task={task_id}"
                    )
            else:
                logger.warning(f"Missing '{feat.name}' in response, task={task_id}")

    def _check_completeness(self, merged: Dict[str, Any], task_id: str) -> None:
        """Log warning if any features are missing from merged results."""
        missing = [f.name for f in self.features if f.name not in merged]
        if missing:
            logger.warning(
                f"Task {task_id}: missing {len(missing)}/{len(self.features)} "
                f"features: {missing}"
            )

    # =========================================================================
    # Sync extraction
    # =========================================================================

    def _extract_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all features for a single task (sync)."""
        task_id = self.task_context.get_task_id(task)
        merged: Dict[str, Any] = {}

        for prefix, suffix, batch_features in self._build_batch_calls(task):
            batch_names = [f.name for f in batch_features]
            response_text = self.client.call_with_prefix(prefix, suffix)
            parsed = parse_llm_response(response_text, expected_features=batch_names)
            self._merge_batch_result(merged, parsed, batch_features, task_id)

        self._check_completeness(merged, task_id)
        return merged

    # =========================================================================
    # Async extraction
    # =========================================================================

    async def _extract_task_async(
        self, task: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Extract all features for a single task (async).

        Batches within a level run sequentially for prefix cache hits.
        The semaphore limits concurrent tasks.
        """
        async with semaphore:
            task_id = self.task_context.get_task_id(task)
            merged: Dict[str, Any] = {}

            for prefix, suffix, batch_features in self._build_batch_calls(task):
                batch_names = [f.name for f in batch_features]
                response_text = await self.client.call_with_prefix_async(prefix, suffix)
                parsed = parse_llm_response(response_text, expected_features=batch_names)
                self._merge_batch_result(merged, parsed, batch_features, task_id)

            self._check_completeness(merged, task_id)
            return merged

    # =========================================================================
    # Output helpers
    # =========================================================================

    def _get_output_path(self, output_dir: Path, task_id: str) -> Path:
        safe_id = task_id.replace("/", "__")
        return output_dir / f"{safe_id}.json"

    def _add_metadata(self, features: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        features["_task_id"] = task_id
        features["_model"] = self.client.model
        features["_provider"] = self.client.provider
        features["_extracted_at"] = datetime.now().isoformat()
        return features

    def _save_task_result(
        self,
        task: Dict[str, Any],
        features: Dict[str, Any],
        output_dir: Path,
        stats: Dict[str, Any],
        index: int,
        total: int,
    ) -> None:
        """Save extraction result and update stats."""
        task_id = self.task_context.get_task_id(task)
        if features:
            self._add_metadata(features, task_id)
            output_path = self._get_output_path(output_dir, task_id)
            with open(output_path, "w") as f:
                json.dump(features, f, indent=2)
            stats["success"] += 1
            print(f"[{index}/{total}] {task_id} OK")
        else:
            stats["failed"] += 1
            stats["failed_task_ids"].append(task_id)
            print(f"[{index}/{total}] {task_id} FAILED")

    # =========================================================================
    # Public run methods
    # =========================================================================

    def run(
        self,
        tasks: List[Dict[str, Any]],
        output_dir: Path,
        skip_existing: bool = True,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        delay: float = 0.5,
    ) -> Optional[Path]:
        """Run extraction synchronously on all tasks.

        Returns:
            Path to aggregated CSV, or None if no tasks processed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = self._filter_tasks(tasks, output_dir, skip_existing, limit, task_ids)

        if not tasks:
            print("No tasks to process")
            return self._aggregate_to_csv(output_dir)

        stats = {"total": len(tasks), "success": 0, "failed": 0, "failed_task_ids": []}

        for i, task in enumerate(tasks):
            try:
                features = self._extract_task(task)
            except Exception as e:
                task_id = self.task_context.get_task_id(task)
                logger.error(f"Error extracting features for {task_id}: {e}")
                features = {}

            self._save_task_result(task, features, output_dir, stats, i + 1, len(tasks))

            if delay and i < len(tasks) - 1:
                time.sleep(delay)

        self._print_summary(stats)
        self._save_stats(stats, output_dir)
        return self._aggregate_to_csv(output_dir)

    def run_parallel(
        self,
        tasks: List[Dict[str, Any]],
        output_dir: Path,
        skip_existing: bool = True,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        concurrency: int = 10,
    ) -> Optional[Path]:
        """Run extraction in parallel with configurable concurrency.

        Returns:
            Path to aggregated CSV, or None if no tasks processed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = self._filter_tasks(tasks, output_dir, skip_existing, limit, task_ids)

        if not tasks:
            print("No tasks to process")
            return self._aggregate_to_csv(output_dir)

        print(f"\nProcessing {len(tasks)} tasks with concurrency={concurrency}...\n")
        stats = {"total": len(tasks), "success": 0, "failed": 0, "failed_task_ids": []}

        async def _run():
            semaphore = asyncio.Semaphore(concurrency)

            async def process_one(i: int, task: Dict[str, Any]):
                try:
                    features = await self._extract_task_async(task, semaphore)
                except Exception as e:
                    task_id = self.task_context.get_task_id(task)
                    logger.error(f"Error for {task_id}: {e}")
                    features = {}
                self._save_task_result(
                    task, features, output_dir, stats, i + 1, len(tasks)
                )

            await asyncio.gather(
                *(process_one(i, task) for i, task in enumerate(tasks))
            )

        asyncio.run(_run())

        self._print_summary(stats)
        self._save_stats(stats, output_dir)
        return self._aggregate_to_csv(output_dir)

    def dry_run(
        self,
        tasks: List[Dict[str, Any]],
        output_dir: Path,
        skip_existing: bool = True,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> None:
        """Show execution plan and cost estimate without running."""
        output_dir = Path(output_dir)
        filtered = self._filter_tasks(tasks, output_dir, skip_existing, limit, task_ids)

        print("\n" + "=" * 60)
        print("DRY RUN - EXECUTION PLAN")
        print("=" * 60)

        print(f"\nDataset: {self.task_context.name}")
        print(f"Provider: {self.client.provider}")
        print(f"Model: {self.client.model}")
        print(f"Output directory: {output_dir}")
        print(f"\nTasks: {len(tasks)} total, {len(filtered)} to process")

        # Show batch plan
        total_batches_per_task = 0
        print(f"\nFeatures ({len(self.features)} total):")
        for level in (InfoLevel.PROBLEM, InfoLevel.TEST, InfoLevel.SOLUTION):
            level_features = self.level_groups.get(level)
            if not level_features:
                continue
            batches = _batch(level_features, self.batch_size)
            total_batches_per_task += len(batches)
            print(f"  {level.value}: {len(level_features)} features in {len(batches)} batch(es)")
            for j, b in enumerate(batches):
                print(f"    batch {j+1}: {[f.name for f in b]}")

        total_api_calls = total_batches_per_task * len(filtered)
        print(f"\nTotal API calls: {total_api_calls} ({len(filtered)} tasks x {total_batches_per_task} batches)")

        if filtered:
            cost_info = self.client.estimate_cost(total_api_calls)
            print(f"\nEstimated cost:")
            print(f"  Input tokens: ~{cost_info['input_tokens']:,} (${cost_info['input_cost']:.2f})")
            print(f"  Output tokens: ~{cost_info['output_tokens']:,} (${cost_info['output_cost']:.2f})")
            print(f"  Total: ~${cost_info['total_cost']:.2f}")
            print(f"  ({cost_info['pricing_note']})")

        # Show sample prefix + suffix for first task
        if filtered:
            task = filtered[0]
            task_id = self.task_context.get_task_id(task)
            print(f"\n--- Sample prompts for task: {task_id} ---")
            for prefix, suffix, batch_features in self._build_batch_calls(task):
                level = batch_features[0].info_level
                names = [f.name for f in batch_features]
                print(f"\n[{level.value}] features={names}")
                print(f"  Prefix ({len(prefix)} chars):")
                print("  " + prefix[:300].replace("\n", "\n  ") + ("..." if len(prefix) > 300 else ""))
                print(f"  Suffix ({len(suffix)} chars):")
                print("  " + suffix[:300].replace("\n", "\n  ") + ("..." if len(suffix) > 300 else ""))

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _filter_tasks(
        self,
        tasks: List[Dict[str, Any]],
        output_dir: Path,
        skip_existing: bool,
        limit: Optional[int],
        task_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        if task_ids:
            task_ids_set = set(task_ids)
            tasks = [t for t in tasks if self.task_context.get_task_id(t) in task_ids_set]

        if limit and len(tasks) > limit:
            tasks = tasks[:limit]

        if skip_existing:
            original_count = len(tasks)
            tasks = [
                t for t in tasks
                if not self._get_output_path(output_dir, self.task_context.get_task_id(t)).exists()
            ]
            skipped = original_count - len(tasks)
            if skipped > 0:
                logger.info(f"Skipping {skipped} existing, {len(tasks)} remaining")

        return tasks

    def _aggregate_to_csv(self, output_dir: Path) -> Optional[Path]:
        rows = []
        for json_file in output_dir.glob("*.json"):
            if json_file.name.startswith("compute_stats"):
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                rows.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {json_file}: {e}")

        if not rows:
            logger.warning("No feature files found to aggregate")
            return None

        df = pd.DataFrame(rows)

        feature_cols = [f.name for f in self.features if f.name in df.columns]
        meta_cols = sorted(c for c in df.columns if c.startswith("_"))
        other_cols = [c for c in df.columns if c not in feature_cols and c not in meta_cols]

        if "_task_id" in meta_cols:
            meta_cols.remove("_task_id")
            meta_cols = ["_task_id"] + meta_cols

        ordered = [c for c in feature_cols + other_cols + meta_cols if c in df.columns]
        df = df[ordered]

        csv_path = output_dir / "llm_judge_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"Aggregated {len(rows)} tasks to {csv_path}")
        return csv_path

    def _print_summary(self, stats: Dict[str, Any]) -> None:
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total processed: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        if stats.get("failed_task_ids"):
            print(f"Failed task IDs: {stats['failed_task_ids']}")

    def _save_stats(self, stats: Dict[str, Any], output_dir: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = output_dir / f"compute_stats_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "dataset": self.task_context.name,
                "provider": self.client.provider,
                "model": self.client.model,
                "features": [feat.name for feat in self.features],
                "stats": stats,
            }, f, indent=2)
