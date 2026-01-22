"""LLM Feature Extractor for semantic feature extraction from tasks.

This module provides the main extraction class that handles:
- Per-task feature extraction with caching
- Resume capability (skip existing JSONs)
- Aggregation to CSV
- Dry-run cost estimation
- Parallel extraction with configurable concurrency
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from experiment_ab_shared.llm_judge.api_client import LLMApiClient
from experiment_ab_shared.llm_judge.prompt_config import PromptConfig
from experiment_ab_shared.llm_judge.response_parser import parse_llm_response, validate_features

logger = logging.getLogger(__name__)


class LLMFeatureExtractor:
    """Extract LLM judge features with caching and resumption.

    This class handles the extraction of semantic features from tasks using
    an LLM. It supports:
    - Per-task JSON caching for resumption
    - Aggregation to CSV
    - Dry-run mode with cost estimation
    - Multiple LLM providers (Anthropic, OpenAI)

    Example:
        >>> from experiment_ab_shared.llm_judge.prompts import get_prompt_config
        >>> config = get_prompt_config("swebench")
        >>> extractor = LLMFeatureExtractor(config, Path("output"))
        >>> csv_path = extractor.run(tasks)
    """

    def __init__(
        self,
        prompt_config: PromptConfig,
        output_dir: Path,
        provider: str = "anthropic",
        model: Optional[str] = None,
    ):
        """Initialize the extractor.

        Args:
            prompt_config: Configuration for the prompt template and features
            output_dir: Directory to save per-task JSON files and aggregated CSV
            provider: LLM provider ("anthropic" or "openai")
            model: Specific model to use (defaults to provider's default)
        """
        self.config = prompt_config
        self.output_dir = Path(output_dir)
        self.client = LLMApiClient(provider, model)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_task_id(self, task: Dict[str, Any]) -> str:
        """Extract task ID using the configured field name."""
        task_id_field = self.config.task_id_field
        if task_id_field not in task:
            raise KeyError(
                f"Task missing required field '{task_id_field}'. "
                f"Available fields: {list(task.keys())}"
            )
        return task[task_id_field]

    def _get_output_path(self, task_id: str) -> Path:
        """Get the JSON output path for a task."""
        # Sanitize task_id for filesystem (replace / with __)
        safe_id = task_id.replace("/", "__")
        return self.output_dir / f"{safe_id}.json"

    def extract_features(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract features for a single task.

        Args:
            task: Task dictionary with required fields for the prompt

        Returns:
            Dictionary with extracted features and metadata, or None if failed
        """
        task_id = self._get_task_id(task)

        # Format prompt
        prompt = self.config.format_prompt(task)

        try:
            # Call LLM
            response_text = self.client.call(prompt)

            # Parse response
            feature_names = self.config.get_feature_names()
            features = parse_llm_response(response_text, expected_features=feature_names)

            if features is None:
                logger.warning(f"Failed to parse response for task {task_id}")
                return None

            # Validate features
            if not validate_features(features, feature_names, require_all=False):
                logger.warning(f"Response missing expected features for task {task_id}")
                return None

            # Add metadata
            features["_task_id"] = task_id
            features["_model"] = self.client.model
            features["_provider"] = self.client.provider
            features["_extracted_at"] = datetime.now().isoformat()

            return features

        except Exception as e:
            logger.error(f"Error extracting features for task {task_id}: {e}")
            return None

    def run(
        self,
        tasks: List[Dict[str, Any]],
        skip_existing: bool = True,
        delay: float = 0.5,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Optional[Path]:
        """Run extraction on all tasks.

        Args:
            tasks: List of task dictionaries
            skip_existing: Skip tasks with existing JSON files
            delay: Delay between API calls in seconds
            limit: Maximum number of tasks to process
            task_ids: If provided, only process these specific task IDs
            progress_callback: Optional callback(current, total, task_id) for progress

        Returns:
            Path to aggregated CSV file, or None if no tasks processed
        """
        # Filter to specific task IDs if requested
        if task_ids:
            task_ids_set = set(task_ids)
            tasks = [t for t in tasks if self._get_task_id(t) in task_ids_set]
            logger.info(f"Filtered to {len(tasks)} specified tasks")

        # Apply limit
        if limit and len(tasks) > limit:
            tasks = tasks[:limit]
            logger.info(f"Limited to {limit} tasks")

        # Filter out existing
        if skip_existing:
            original_count = len(tasks)
            tasks = [
                t for t in tasks
                if not self._get_output_path(self._get_task_id(t)).exists()
            ]
            skipped = original_count - len(tasks)
            if skipped > 0:
                logger.info(f"Skipping {skipped} existing, {len(tasks)} remaining")

        if not tasks:
            logger.info("No tasks to process")
            return self.aggregate_to_csv()

        # Process tasks
        stats = {"total": len(tasks), "success": 0, "failed": 0}

        for i, task in enumerate(tasks):
            task_id = self._get_task_id(task)
            output_path = self._get_output_path(task_id)

            if progress_callback:
                progress_callback(i + 1, len(tasks), task_id)
            else:
                print(f"[{i+1}/{len(tasks)}] {task_id}...")

            # Extract features
            features = self.extract_features(task)

            if features:
                # Save features
                with open(output_path, "w") as f:
                    json.dump(features, f, indent=2)
                stats["success"] += 1

                # Log key features (first 2)
                feature_names = self.config.get_feature_names()[:2]
                feature_preview = ", ".join(
                    f"{name}: {features.get(name, '?')}" for name in feature_names
                )
                logger.debug(f"    -> {feature_preview}")
            else:
                stats["failed"] += 1
                logger.warning(f"    Failed to extract features")

            # Rate limiting
            if delay and i < len(tasks) - 1:
                time.sleep(delay)

        # Summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total processed: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")

        # Save stats
        self._save_stats(stats)

        # Aggregate to CSV
        return self.aggregate_to_csv()

    async def _extract_single_async(
        self,
        task: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Optional[Dict[str, Any]]:
        """Extract features for a single task asynchronously."""
        async with semaphore:
            task_id = self._get_task_id(task)
            prompt = self.config.format_prompt(task)

            try:
                response_text = await self.client.call_async(prompt)

                feature_names = self.config.get_feature_names()
                features = parse_llm_response(response_text, expected_features=feature_names)

                if features is None:
                    logger.warning(f"Failed to parse response for task {task_id}")
                    return None

                if not validate_features(features, feature_names, require_all=False):
                    logger.warning(f"Response missing expected features for task {task_id}")
                    return None

                features["_task_id"] = task_id
                features["_model"] = self.client.model
                features["_provider"] = self.client.provider
                features["_extracted_at"] = datetime.now().isoformat()

                return features

            except Exception as e:
                logger.error(f"Error extracting features for task {task_id}: {e}")
                return None

    async def _run_parallel_async(
        self,
        tasks: List[Dict[str, Any]],
        concurrency: int = 10,
    ) -> Dict[str, int]:
        """Run extraction on tasks in parallel with limited concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        stats = {"total": len(tasks), "success": 0, "failed": 0}

        async def process_task(i: int, task: Dict[str, Any]):
            task_id = self._get_task_id(task)
            output_path = self._get_output_path(task_id)

            features = await self._extract_single_async(task, semaphore)

            if features:
                with open(output_path, "w") as f:
                    json.dump(features, f, indent=2)
                stats["success"] += 1
                print(f"[{i+1}/{len(tasks)}] {task_id}... ✓")
            else:
                stats["failed"] += 1
                print(f"[{i+1}/{len(tasks)}] {task_id}... ✗")

        # Create all tasks
        coroutines = [process_task(i, task) for i, task in enumerate(tasks)]

        # Run with progress
        await asyncio.gather(*coroutines)

        return stats

    def run_parallel(
        self,
        tasks: List[Dict[str, Any]],
        skip_existing: bool = True,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        concurrency: int = 10,
    ) -> Optional[Path]:
        """Run extraction on all tasks in parallel.

        Args:
            tasks: List of task dictionaries
            skip_existing: Skip tasks with existing JSON files
            limit: Maximum number of tasks to process
            task_ids: If provided, only process these specific task IDs
            concurrency: Maximum number of concurrent API calls (default: 10)

        Returns:
            Path to aggregated CSV file, or None if no tasks processed
        """
        # Filter to specific task IDs if requested
        if task_ids:
            task_ids_set = set(task_ids)
            tasks = [t for t in tasks if self._get_task_id(t) in task_ids_set]
            logger.info(f"Filtered to {len(tasks)} specified tasks")

        # Apply limit
        if limit and len(tasks) > limit:
            tasks = tasks[:limit]
            logger.info(f"Limited to {limit} tasks")

        # Filter out existing
        if skip_existing:
            original_count = len(tasks)
            tasks = [
                t for t in tasks
                if not self._get_output_path(self._get_task_id(t)).exists()
            ]
            skipped = original_count - len(tasks)
            if skipped > 0:
                logger.info(f"Skipping {skipped} existing, {len(tasks)} remaining")

        if not tasks:
            logger.info("No tasks to process")
            return self.aggregate_to_csv()

        print(f"\nProcessing {len(tasks)} tasks with concurrency={concurrency}...\n")

        # Run async extraction
        stats = asyncio.run(self._run_parallel_async(tasks, concurrency))

        # Summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total processed: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")

        # Save stats
        self._save_stats(stats)

        # Aggregate to CSV
        return self.aggregate_to_csv()

    def aggregate_to_csv(self) -> Optional[Path]:
        """Combine per-task JSONs into a single CSV.

        Returns:
            Path to the CSV file, or None if no JSON files found
        """
        rows = []

        for json_file in self.output_dir.glob("*.json"):
            # Skip stats files
            if json_file.name.startswith("compute_stats"):
                continue
            try:
                with open(json_file) as f:
                    data = json.load(f)
                rows.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {json_file}: {e}")
                continue

        if not rows:
            logger.warning("No feature files found to aggregate")
            return None

        df = pd.DataFrame(rows)

        # Reorder columns: task_id first, then features, then metadata
        feature_cols = self.config.get_feature_names()
        meta_cols = [c for c in df.columns if c.startswith("_")]
        other_cols = [c for c in df.columns if c not in feature_cols and c not in meta_cols]

        # Put _task_id first in meta cols
        if "_task_id" in meta_cols:
            meta_cols.remove("_task_id")
            meta_cols = ["_task_id"] + meta_cols

        # Final column order
        ordered_cols = []
        for c in feature_cols:
            if c in df.columns:
                ordered_cols.append(c)
        for c in other_cols:
            if c in df.columns:
                ordered_cols.append(c)
        for c in meta_cols:
            if c in df.columns:
                ordered_cols.append(c)

        df = df[ordered_cols]

        csv_path = self.output_dir / "llm_judge_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"Aggregated {len(rows)} tasks to {csv_path}")
        return csv_path

    def dry_run(
        self,
        tasks: List[Dict[str, Any]],
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> None:
        """Show execution plan and cost estimate without running.

        Args:
            tasks: List of task dictionaries
            limit: Maximum number of tasks to process
            task_ids: If provided, only process these specific task IDs
            skip_existing: Whether to skip existing tasks in count
        """
        # Filter to specific task IDs if requested
        if task_ids:
            task_ids_set = set(task_ids)
            tasks = [t for t in tasks if self._get_task_id(t) in task_ids_set]

        # Apply limit
        if limit and len(tasks) > limit:
            tasks = tasks[:limit]

        # Filter out existing
        original_count = len(tasks)
        if skip_existing:
            tasks = [
                t for t in tasks
                if not self._get_output_path(self._get_task_id(t)).exists()
            ]
            skipped = original_count - len(tasks)
        else:
            skipped = 0

        print("\n" + "=" * 60)
        print("DRY RUN - EXECUTION PLAN")
        print("=" * 60)

        print(f"\nDataset: {self.config.name}")
        print(f"Provider: {self.client.provider}")
        print(f"Model: {self.client.model}")
        print(f"Output directory: {self.output_dir}")

        print(f"\nTasks: {original_count} total")
        if skipped > 0:
            print(f"  - Skipping {skipped} with existing features")
        print(f"  - {len(tasks)} to process")

        # Show sample tasks
        if tasks:
            print(f"\nSample tasks (first 10):")
            for task in tasks[:10]:
                task_id = self._get_task_id(task)
                print(f"  {task_id}")
            if len(tasks) > 10:
                print(f"  ... and {len(tasks) - 10} more")

        # Cost estimate
        if tasks:
            cost_info = self.client.estimate_cost(len(tasks))
            print(f"\nEstimated cost:")
            print(f"  Input tokens: ~{cost_info['input_tokens']:,} (${cost_info['input_cost']:.2f})")
            print(f"  Output tokens: ~{cost_info['output_tokens']:,} (${cost_info['output_cost']:.2f})")
            print(f"  Total: ~${cost_info['total_cost']:.2f}")
            print(f"  ({cost_info['pricing_note']})")

        print(f"\nFeatures to extract ({len(self.config.features)}):")
        for feat in self.config.features:
            print(f"  - {feat.name} ({feat.min_value}-{feat.max_value})")

    def _save_stats(self, stats: Dict[str, int]) -> Path:
        """Save extraction statistics to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = self.output_dir / f"compute_stats_{timestamp}.json"

        with open(stats_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "dataset": self.config.name,
                "provider": self.client.provider,
                "model": self.client.model,
                "stats": stats,
            }, f, indent=2)

        logger.info(f"Stats saved to: {stats_file}")
        return stats_file
