#!/usr/bin/env python3
"""Overnight Lunette feature extraction script with robust retry logic.

This script runs Lunette feature extraction with:
- Exponential backoff retries (up to 3 attempts per task)
- Periodic post-processing of raw responses
- Progress tracking and resume capability
- Separate output directory (v2) for clean results
- Prioritizes test tasks for stable AUC measurement

Usage:
    # Dry run to see execution plan
    python -m experiment_a.overnight_lunette_extraction --dry_run

    # Run overnight (all 500 tasks)
    python -m experiment_a.overnight_lunette_extraction

    # Run with limited concurrency
    python -m experiment_a.overnight_lunette_extraction --concurrency 3

    # Resume from previous run
    python -m experiment_a.overnight_lunette_extraction --resume

Output:
    chris_output/experiment_a/lunette_features_v2/
        - *.json: Parsed feature files
        - *_raw.json: Raw responses (for post-processing)
        - *_error.json: Error logs
        - progress.json: Progress tracking
        - overnight_log.txt: Detailed log
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.lunette_grading_prompt import (
    LUNETTE_FEATURE_NAMES,
    format_grading_prompt,
)
from experiment_a.postprocess_lunette_features import (
    parse_raw_response,
    FEATURE_SPECS,
)
from experiment_a.data_loader import stable_split_tasks

# Try to import Lunette
try:
    from lunette import LunetteClient
    from lunette.analysis import GradingPlan
    from lunette.models.run import Run
    from lunette.models.trajectory import Trajectory, ScalarScore
    from lunette.models.messages import AssistantMessage
    HAS_LUNETTE = True
except ImportError:
    HAS_LUNETTE = False

# Configuration
VERSION = "v2"  # Version tag for this script's output
OUTPUT_DIR = ROOT / "chris_output" / "experiment_a" / f"lunette_features_{VERSION}"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
LOG_FILE = OUTPUT_DIR / "overnight_log.txt"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 5  # seconds
MAX_BACKOFF = 120  # seconds
BACKOFF_MULTIPLIER = 2

# Batch configuration
BATCH_SIZE = 20  # Process in batches, post-process after each
POSTPROCESS_INTERVAL = 10  # Post-process after every N extractions


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger("overnight_extraction")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_swebench_verified() -> List[dict]:
    """Load SWE-bench Verified dataset."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    tasks = []
    for item in ds:
        tasks.append({
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "version": item["version"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "hints_text": item["hints_text"],
            "base_commit": item["base_commit"],
            "FAIL_TO_PASS": item["FAIL_TO_PASS"],
            "PASS_TO_PASS": item["PASS_TO_PASS"],
        })

    return tasks


def create_dummy_trajectory(task_id: str) -> Trajectory:
    """Create a minimal trajectory for Lunette sandbox access."""
    return Trajectory(
        sample=task_id,
        messages=[
            AssistantMessage(
                position=0,
                content="Dummy agent: No attempt made. This trajectory exists only to enable Lunette sandbox access for feature extraction.",
            )
        ],
        scores={"resolved": ScalarScore(value=0.0)},
        solution=None,
        metadata={"dummy": True},
    )


def parse_feature_response(text: str) -> Optional[Dict]:
    """Parse JSON response from Lunette grading."""
    if not text:
        return None

    # Try to extract JSON from response
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        json_match = re.search(r"\{[^{}]*\"repo_file_count\"[^{}]*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

    try:
        data = json.loads(text)
        if "repo_file_count" not in data and "fix_in_description" not in data:
            return None
        return data
    except json.JSONDecodeError:
        return None


class ProgressTracker:
    """Track extraction progress for resume capability."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        """Load progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return {
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "completed_tasks": [],
            "failed_tasks": [],
            "retry_counts": {},
            "stats": {
                "total_attempted": 0,
                "success": 0,
                "failed": 0,
                "post_processed": 0,
            }
        }

    def save(self):
        """Save progress to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def mark_completed(self, task_id: str):
        """Mark a task as successfully completed."""
        if task_id not in self.data["completed_tasks"]:
            self.data["completed_tasks"].append(task_id)
        self.data["stats"]["success"] += 1
        self.data["stats"]["total_attempted"] += 1
        self.save()

    def mark_failed(self, task_id: str, error: str):
        """Mark a task as failed."""
        if task_id not in self.data["failed_tasks"]:
            self.data["failed_tasks"].append(task_id)
        self.data["stats"]["failed"] += 1
        self.data["stats"]["total_attempted"] += 1
        self.save()

    def get_retry_count(self, task_id: str) -> int:
        """Get retry count for a task."""
        return self.data["retry_counts"].get(task_id, 0)

    def increment_retry(self, task_id: str):
        """Increment retry count for a task."""
        self.data["retry_counts"][task_id] = self.get_retry_count(task_id) + 1
        self.save()

    def is_completed(self, task_id: str) -> bool:
        """Check if task is already completed."""
        return task_id in self.data["completed_tasks"]

    def should_skip(self, task_id: str) -> bool:
        """Check if task should be skipped (completed or max retries)."""
        if self.is_completed(task_id):
            return True
        if self.get_retry_count(task_id) >= MAX_RETRIES:
            return True
        return False


async def extract_with_retry(
    client: LunetteClient,
    task: dict,
    output_dir: Path,
    tracker: ProgressTracker,
    logger: logging.Logger,
) -> Optional[Dict]:
    """Extract features for a task with retry logic."""
    task_id = task["instance_id"]

    # Check for existing feature file
    output_file = output_dir / f"{task_id}.json"
    if output_file.exists():
        logger.debug(f"  {task_id}: Already has features, loading from cache")
        with open(output_file) as f:
            return json.load(f)

    retry_count = tracker.get_retry_count(task_id)
    backoff = INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** retry_count)
    backoff = min(backoff, MAX_BACKOFF)

    try:
        # 1. Create and upload dummy trajectory
        trajectory = create_dummy_trajectory(task_id)
        run = Run(
            task="swebench-verified",
            model=f"dummy_solver_{VERSION}",
            trajectories=[trajectory],
            metadata={
                "repo": task["repo"],
                "patch": task["patch"],
                "test_patch": task.get("test_patch", ""),
                "FAIL_TO_PASS": task.get("FAIL_TO_PASS", "[]"),
                "PASS_TO_PASS": task.get("PASS_TO_PASS", "[]"),
                "version": task.get("version", ""),
                "base_commit": task.get("base_commit", ""),
                "extraction_version": VERSION,
            },
        )

        logger.debug(f"  {task_id}: Uploading trajectory...")
        run_meta = await client.save_run(run)
        run_id = run_meta["run_id"]

        # 2. Format and run grading
        grading_prompt = format_grading_prompt(
            instance_id=task_id,
            repo=task["repo"],
            version=task.get("version", "unknown"),
            problem_statement=task["problem_statement"],
            patch=task["patch"],
            fail_to_pass=task.get("FAIL_TO_PASS", "[]"),
            pass_to_pass=task.get("PASS_TO_PASS", "[]"),
            hints_text=task.get("hints_text", ""),
        )

        logger.debug(f"  {task_id}: Running Lunette grading...")
        results = await client.investigate(
            run_id=run_id,
            plan=GradingPlan(name="task-difficulty-features-v2", prompt=grading_prompt),
            limit=1,
        )

        if not results.results:
            raise ValueError("No results returned from Lunette")

        # 3. Parse response
        result_data = results.results[0].data

        if isinstance(result_data, dict):
            if "explanation" in result_data:
                features = parse_feature_response(result_data["explanation"])
            else:
                features = result_data
        else:
            features = parse_feature_response(str(result_data))

        # 4. Save raw response for post-processing if parsing failed
        if features is None:
            logger.info(f"  {task_id}: Saving raw response for post-processing")
            raw_file = output_dir / f"{task_id}_raw.json"
            with open(raw_file, "w") as f:
                json.dump({
                    "run_id": run_id,
                    "raw_response": result_data,
                    "extraction_version": VERSION,
                    "extracted_at": datetime.now().isoformat(),
                }, f, indent=2)

            # Try to post-process immediately
            with open(raw_file) as f:
                raw_data = json.load(f)
            features = parse_raw_response(raw_data, task_id)

            if features:
                features["_extraction_version"] = VERSION
                with open(output_file, "w") as f:
                    json.dump(features, f, indent=2)
                tracker.mark_completed(task_id)
                logger.info(f"  {task_id}: SUCCESS (post-processed)")
                return features
            else:
                # Still failed - increment retry
                tracker.increment_retry(task_id)
                return None

        # 5. Add metadata and save
        features["_instance_id"] = task_id
        features["_run_id"] = run_id
        features["_extracted_at"] = datetime.now().isoformat()
        features["_extraction_version"] = VERSION

        with open(output_file, "w") as f:
            json.dump(features, f, indent=2)

        tracker.mark_completed(task_id)
        logger.info(f"  {task_id}: SUCCESS")
        return features

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"  {task_id}: ERROR - {error_msg}")

        # Save error for debugging
        error_file = output_dir / f"{task_id}_error.json"
        with open(error_file, "w") as f:
            json.dump({
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "task_id": task_id,
                "retry_count": retry_count,
                "timestamp": datetime.now().isoformat(),
                "extraction_version": VERSION,
            }, f, indent=2)

        tracker.increment_retry(task_id)

        # Backoff before returning
        if retry_count < MAX_RETRIES - 1:
            logger.info(f"  {task_id}: Will retry after {backoff}s backoff")
            await asyncio.sleep(backoff)

        return None


def postprocess_raw_files(output_dir: Path, logger: logging.Logger) -> int:
    """Post-process all raw files that don't have feature files yet."""
    raw_files = list(output_dir.glob("*_raw.json"))
    processed = 0

    for raw_file in raw_files:
        task_id = raw_file.stem.replace("_raw", "")
        feature_file = output_dir / f"{task_id}.json"

        if feature_file.exists():
            continue

        try:
            with open(raw_file) as f:
                raw_data = json.load(f)

            features = parse_raw_response(raw_data, task_id)

            if features:
                features["_extraction_version"] = VERSION
                with open(feature_file, "w") as f:
                    json.dump(features, f, indent=2)
                processed += 1
                logger.debug(f"  Post-processed: {task_id}")
        except Exception as e:
            logger.warning(f"  Failed to post-process {task_id}: {e}")

    return processed


def aggregate_to_csv(output_dir: Path, logger: logging.Logger) -> Path:
    """Aggregate all feature files to a single CSV."""
    feature_files = list(output_dir.glob("*.json"))
    feature_files = [f for f in feature_files
                     if not f.name.endswith("_raw.json")
                     and not f.name.endswith("_error.json")
                     and f.name != "progress.json"]

    rows = []
    for f in feature_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            task_id = f.stem
            if "_instance_id" in data:
                task_id = data["_instance_id"]
            row = {"task_id": task_id}
            row.update(data)
            rows.append(row)
        except Exception as e:
            logger.warning(f"  Failed to load {f.name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / "lunette_features.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Aggregated {len(rows)} features to {csv_path}")
        return csv_path

    return None


def get_task_priority_order(tasks: List[dict]) -> List[dict]:
    """Order tasks with test tasks first, then train tasks."""
    # Load items to get task IDs
    items_path = ROOT / "clean_data" / "swebench_verified_20251115_full" / "1d_1pl" / "items.csv"
    items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(items.index)

    # Split into train/test
    train_tasks_ids, test_tasks_ids = stable_split_tasks(
        all_task_ids, test_fraction=0.2, seed=42
    )
    test_tasks_set = set(test_tasks_ids)

    # Separate tasks
    test_tasks = [t for t in tasks if t["instance_id"] in test_tasks_set]
    train_tasks = [t for t in tasks if t["instance_id"] not in test_tasks_set]

    # Return test tasks first
    return test_tasks + train_tasks


async def main():
    parser = argparse.ArgumentParser(
        description="Overnight Lunette feature extraction with retry logic"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show execution plan without running"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of concurrent tasks (default: 5)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run"
    )
    parser.add_argument(
        "--test_only", action="store_true",
        help="Only process test tasks"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tasks to process"
    )
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_FILE)

    logger.info("=" * 70)
    logger.info(f"OVERNIGHT LUNETTE EXTRACTION - VERSION {VERSION}")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # Load tasks
    logger.info("Loading SWE-bench Verified dataset...")
    tasks = load_swebench_verified()
    logger.info(f"Loaded {len(tasks)} tasks")

    # Prioritize test tasks
    tasks = get_task_priority_order(tasks)

    # Filter to test only if requested
    if args.test_only:
        items_path = ROOT / "clean_data" / "swebench_verified_20251115_full" / "1d_1pl" / "items.csv"
        items = pd.read_csv(items_path, index_col=0)
        all_task_ids = list(items.index)
        _, test_tasks_ids = stable_split_tasks(all_task_ids, test_fraction=0.2, seed=42)
        test_set = set(test_tasks_ids)
        tasks = [t for t in tasks if t["instance_id"] in test_set]
        logger.info(f"Filtered to {len(tasks)} test tasks")

    # Apply limit
    if args.limit:
        tasks = tasks[:args.limit]
        logger.info(f"Limited to {args.limit} tasks")

    # Initialize progress tracker
    tracker = ProgressTracker(PROGRESS_FILE)

    # Filter out already completed tasks if resuming
    if args.resume:
        original_count = len(tasks)
        tasks = [t for t in tasks if not tracker.should_skip(t["instance_id"])]
        skipped = original_count - len(tasks)
        logger.info(f"Resuming: skipping {skipped} completed/max-retry tasks")

    # Also skip tasks with existing feature files
    original_count = len(tasks)
    tasks = [t for t in tasks if not (OUTPUT_DIR / f"{t['instance_id']}.json").exists()]
    skipped = original_count - len(tasks)
    if skipped > 0:
        logger.info(f"Skipping {skipped} tasks with existing feature files")

    logger.info(f"Tasks to process: {len(tasks)}")

    if args.dry_run:
        logger.info("\n=== DRY RUN ===")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info(f"Concurrency: {args.concurrency}")
        logger.info(f"Max retries: {MAX_RETRIES}")
        logger.info(f"\nFirst 10 tasks:")
        for task in tasks[:10]:
            logger.info(f"  - {task['instance_id']} ({task['repo']})")
        if len(tasks) > 10:
            logger.info(f"  ... and {len(tasks) - 10} more")

        # Estimate cost
        cost_per_task = 0.15
        logger.info(f"\nEstimated cost: ~${len(tasks) * cost_per_task:.2f}")
        return

    if not HAS_LUNETTE:
        logger.error("lunette-sdk not installed. Run: pip install lunette-sdk")
        return

    # Process tasks
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_with_semaphore(client, task, idx, total):
        async with semaphore:
            task_id = task["instance_id"]
            logger.info(f"\n[{idx + 1}/{total}] {task_id}")
            return await extract_with_retry(client, task, OUTPUT_DIR, tracker, logger)

    all_features = []
    extraction_count = 0

    async with LunetteClient() as client:
        # Process in batches
        for batch_start in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, len(tasks))

            logger.info(f"\n{'='*50}")
            logger.info(f"BATCH {batch_start//BATCH_SIZE + 1}: Tasks {batch_start + 1}-{batch_end}")
            logger.info(f"{'='*50}")

            # Process batch concurrently
            coroutines = [
                process_with_semaphore(client, task, batch_start + i, len(tasks))
                for i, task in enumerate(batch)
            ]

            results = await asyncio.gather(*coroutines, return_exceptions=True)

            for features in results:
                if isinstance(features, Exception):
                    logger.warning(f"Exception in batch: {features}")
                elif features:
                    all_features.append(features)
                extraction_count += 1

            # Post-process raw files after each batch
            logger.info(f"\nPost-processing raw files...")
            processed = postprocess_raw_files(OUTPUT_DIR, logger)
            if processed > 0:
                logger.info(f"  Post-processed {processed} additional files")
                tracker.data["stats"]["post_processed"] += processed
                tracker.save()

            # Aggregate to CSV
            aggregate_to_csv(OUTPUT_DIR, logger)

            # Log progress
            logger.info(f"\nProgress: {tracker.data['stats']}")

    # Final post-processing
    logger.info("\n" + "=" * 70)
    logger.info("FINAL POST-PROCESSING")
    logger.info("=" * 70)

    processed = postprocess_raw_files(OUTPUT_DIR, logger)
    logger.info(f"Final post-processing: {processed} files")

    # Final aggregation
    csv_path = aggregate_to_csv(OUTPUT_DIR, logger)

    # Check train/test coverage
    if csv_path:
        df = pd.read_csv(csv_path)
        items_path = ROOT / "clean_data" / "swebench_verified_20251115_full" / "1d_1pl" / "items.csv"
        items = pd.read_csv(items_path, index_col=0)
        all_task_ids = list(items.index)
        train_tasks_ids, test_tasks_ids = stable_split_tasks(
            all_task_ids, test_fraction=0.2, seed=42
        )

        task_ids = set(df["task_id"].tolist())
        in_train = len([t for t in task_ids if t in train_tasks_ids])
        in_test = len([t for t in task_ids if t in test_tasks_ids])

        logger.info(f"\nCoverage:")
        logger.info(f"  Train: {in_train} / {len(train_tasks_ids)}")
        logger.info(f"  Test:  {in_test} / {len(test_tasks_ids)}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total attempted: {tracker.data['stats']['total_attempted']}")
    logger.info(f"Successful: {tracker.data['stats']['success']}")
    logger.info(f"Failed: {tracker.data['stats']['failed']}")
    logger.info(f"Post-processed: {tracker.data['stats']['post_processed']}")
    logger.info(f"\nCompleted at: {datetime.now().isoformat()}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Save final progress
    tracker.save()


if __name__ == "__main__":
    asyncio.run(main())
