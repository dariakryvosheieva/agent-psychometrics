"""Simple single-prompt feature extractor for trajectory features.

Experiment B extracts features from agent trajectories using a single prompt
per task (no info levels or batching). This module provides the lightweight
PromptConfig and extractor that experiment B's trajectory features need.

For the batched, multi-level extractor used by Experiment A, see
experiment_ab_shared.llm_judge.batched_extractor.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from experiment_ab_shared.llm_judge.api_client import LLMApiClient
from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition
from experiment_ab_shared.llm_judge.response_parser import parse_llm_response, validate_features

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPromptConfig:
    """Simple prompt config for single-prompt trajectory feature extraction.

    Unlike the batched extractor (which groups features by info level),
    this sends one prompt per task containing all features.
    """

    name: str
    features: List[FeatureDefinition]
    prompt_template: str
    task_id_field: str
    truncation_limits: Dict[str, int] = field(default_factory=dict)
    format_prompt_fn: Optional[Callable[[Dict[str, Any]], str]] = None

    def get_feature_names(self) -> List[str]:
        return [f.name for f in self.features]

    def format_prompt(self, task: Dict[str, Any]) -> str:
        if self.format_prompt_fn is not None:
            return self.format_prompt_fn(task)

        truncated_task = {}
        for key, value in task.items():
            if isinstance(value, str) and key in self.truncation_limits:
                limit = self.truncation_limits[key]
                truncated_task[key] = value[:limit] if len(value) > limit else value
            else:
                truncated_task[key] = value

        return self.prompt_template.format(**truncated_task)


# Backwards-compatible alias
PromptConfig = TrajectoryPromptConfig


class SimpleFeatureExtractor:
    """Single-prompt feature extractor with caching and resumption.

    Sends one prompt per task (no batching or info levels). Used for
    experiment B's trajectory feature extraction.
    """

    def __init__(
        self,
        prompt_config: TrajectoryPromptConfig,
        output_dir: Path,
        provider: str = "openai",
        model: Optional[str] = None,
    ):
        self.config = prompt_config
        self.output_dir = Path(output_dir)
        self.client = LLMApiClient(provider, model)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_task_id(self, task: Dict[str, Any]) -> str:
        field = self.config.task_id_field
        if field not in task:
            raise KeyError(f"Task missing '{field}'. Available: {list(task.keys())}")
        return task[field]

    def _get_output_path(self, task_id: str) -> Path:
        safe_id = task_id.replace("/", "__")
        return self.output_dir / f"{safe_id}.json"

    def _extract_one(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        task_id = self._get_task_id(task)
        prompt = self.config.format_prompt(task)

        try:
            response_text = self.client.call_with_prefix(prompt, "", max_tokens=1024)
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

    def _filter_tasks(
        self,
        tasks: List[Dict[str, Any]],
        skip_existing: bool,
        limit: Optional[int],
        task_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        if task_ids:
            ids_set = set(task_ids)
            tasks = [t for t in tasks if self._get_task_id(t) in ids_set]
        if limit and len(tasks) > limit:
            tasks = tasks[:limit]
        if skip_existing:
            tasks = [
                t for t in tasks
                if not self._get_output_path(self._get_task_id(t)).exists()
            ]
        return tasks

    def run(
        self,
        tasks: List[Dict[str, Any]],
        skip_existing: bool = True,
        delay: float = 0.5,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> Optional[Path]:
        tasks = self._filter_tasks(tasks, skip_existing, limit, task_ids)

        if not tasks:
            print("No tasks to process")
            return self.aggregate_to_csv()

        stats = {"total": len(tasks), "success": 0, "failed": 0, "failed_task_ids": []}

        for i, task in enumerate(tasks):
            task_id = self._get_task_id(task)
            print(f"[{i+1}/{len(tasks)}] {task_id}...")

            features = self._extract_one(task)
            if features:
                with open(self._get_output_path(task_id), "w") as f:
                    json.dump(features, f, indent=2)
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["failed_task_ids"].append(task_id)

            if delay and i < len(tasks) - 1:
                time.sleep(delay)

        self._print_summary(stats)
        return self.aggregate_to_csv()

    def run_parallel(
        self,
        tasks: List[Dict[str, Any]],
        skip_existing: bool = True,
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        concurrency: int = 10,
    ) -> Optional[Path]:
        tasks = self._filter_tasks(tasks, skip_existing, limit, task_ids)

        if not tasks:
            print("No tasks to process")
            return self.aggregate_to_csv()

        print(f"\nProcessing {len(tasks)} tasks with concurrency={concurrency}...\n")
        stats = {"total": len(tasks), "success": 0, "failed": 0, "failed_task_ids": []}

        async def _run():
            sem = asyncio.Semaphore(concurrency)

            async def process(i, task):
                async with sem:
                    task_id = self._get_task_id(task)
                    prompt = self.config.format_prompt(task)
                    try:
                        text = await self.client.call_with_prefix_async(prompt, "")
                        names = self.config.get_feature_names()
                        features = parse_llm_response(text, expected_features=names)
                        if features and validate_features(features, names):
                            features["_task_id"] = task_id
                            features["_model"] = self.client.model
                            features["_provider"] = self.client.provider
                            features["_extracted_at"] = datetime.now().isoformat()
                            with open(self._get_output_path(task_id), "w") as f:
                                json.dump(features, f, indent=2)
                            stats["success"] += 1
                            print(f"[{i+1}/{len(tasks)}] {task_id} OK")
                        else:
                            stats["failed"] += 1
                            stats["failed_task_ids"].append(task_id)
                            print(f"[{i+1}/{len(tasks)}] {task_id} FAILED")
                    except Exception as e:
                        logger.error(f"Error for {task_id}: {e}")
                        stats["failed"] += 1
                        stats["failed_task_ids"].append(task_id)

            await asyncio.gather(*(process(i, t) for i, t in enumerate(tasks)))

        asyncio.run(_run())
        self._print_summary(stats)
        return self.aggregate_to_csv()

    def dry_run(
        self,
        tasks: List[Dict[str, Any]],
        limit: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
        skip_existing: bool = True,
    ) -> None:
        filtered = self._filter_tasks(tasks, skip_existing, limit, task_ids)

        print("\n" + "=" * 60)
        print("DRY RUN - EXECUTION PLAN")
        print("=" * 60)
        print(f"\nDataset: {self.config.name}")
        print(f"Provider: {self.client.provider}, Model: {self.client.model}")
        print(f"Output: {self.output_dir}")
        print(f"Tasks: {len(filtered)} to process")
        print(f"Features ({len(self.config.features)}):")
        for f in self.config.features:
            print(f"  - {f.name} ({f.min_value}-{f.max_value})")

        if filtered:
            cost = self.client.estimate_cost(len(filtered))
            print(f"\nEstimated cost: ~${cost['total_cost']:.2f}")

    def aggregate_to_csv(self) -> Optional[Path]:
        rows = []
        for json_file in self.output_dir.glob("*.json"):
            if json_file.name.startswith("compute_stats"):
                continue
            try:
                with open(json_file) as f:
                    rows.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        feature_cols = [f for f in self.config.get_feature_names() if f in df.columns]
        meta_cols = sorted(c for c in df.columns if c.startswith("_"))
        other_cols = [c for c in df.columns if c not in feature_cols and c not in meta_cols]

        if "_task_id" in meta_cols:
            meta_cols.remove("_task_id")
            meta_cols = ["_task_id"] + meta_cols

        ordered = [c for c in feature_cols + other_cols + meta_cols if c in df.columns]
        df = df[ordered]

        csv_path = self.output_dir / "llm_judge_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"Aggregated {len(rows)} tasks to {csv_path}")
        return csv_path

    def _print_summary(self, stats):
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
        if stats.get("failed_task_ids"):
            print(f"Failed: {stats['failed_task_ids']}")
