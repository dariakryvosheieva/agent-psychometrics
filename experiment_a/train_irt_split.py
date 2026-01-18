"""Train IRT model on train tasks only to avoid data leakage.

The standard experiment_a pipeline uses IRT difficulties computed on ALL tasks,
which creates data leakage: the ground truth `b` values for training the
difficulty predictor are influenced by test task responses.

This script trains a separate IRT model on just the train tasks, producing
uncontaminated difficulty estimates for use as training targets.

Supports both Bernoulli (SWE-bench) and Binomial (TerminalBench) data formats.

Usage:
    # Dry run to see what would be done
    python -m experiment_a.train_irt_split --dry_run

    # Train IRT on train split (80% of tasks)
    python -m experiment_a.train_irt_split

    # Use custom split parameters
    python -m experiment_a.train_irt_split --test_fraction 0.2 --split_seed 42

    # Force retrain even if cached model exists
    python -m experiment_a.train_irt_split --force

    # For binomial data (TerminalBench)
    python -m experiment_a.train_irt_split --binomial --responses_path data/terminal_bench/...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.data_loader import stable_split_tasks


def get_split_cache_dir(
    output_base: Path,
    test_fraction: float,
    split_seed: int,
    model_type: str = "1pl",
    is_binomial: bool = False,
    fold_idx: Optional[int] = None,
    k_folds: Optional[int] = None,
) -> Path:
    """Get the cache directory for a specific split configuration.

    Args:
        output_base: Base output directory
        test_fraction: Test fraction (e.g., 0.2)
        split_seed: Split random seed
        model_type: IRT model type
        is_binomial: Whether this is binomial (TerminalBench) or Bernoulli (SWE-bench)
        fold_idx: For k-fold CV, the fold index (0 to k-1)
        k_folds: For k-fold CV, the total number of folds

    Returns:
        Path to cache directory for this configuration
    """
    suffix = "_binomial" if is_binomial else ""
    if fold_idx is not None and k_folds is not None:
        # k-fold cross-validation naming
        split_name = f"seed{split_seed}_fold{fold_idx}of{k_folds}_{model_type}{suffix}"
    else:
        # Single holdout naming (legacy)
        split_name = f"seed{split_seed}_test{int(test_fraction*100)}pct_{model_type}{suffix}"
    return output_base / split_name / "1d"


def check_cached_irt(cache_dir: Path) -> bool:
    """Check if a valid cached IRT model exists.

    Args:
        cache_dir: Directory to check

    Returns:
        True if valid cached model exists
    """
    required_files = ["abilities.csv", "items.csv", "split_info.json"]
    return all((cache_dir / f).exists() for f in required_files)


def load_cached_split_info(cache_dir: Path) -> dict:
    """Load cached split info.

    Args:
        cache_dir: Cache directory

    Returns:
        Split info dict or empty dict if not found
    """
    split_info_path = cache_dir / "split_info.json"
    if split_info_path.exists():
        with open(split_info_path) as f:
            return json.load(f)
    return {}


def load_response_matrix(responses_path: Path) -> dict:
    """Load the full agent x task response matrix.

    Works for both Bernoulli and Binomial formats.

    Args:
        responses_path: Path to JSONL with response matrix

    Returns:
        Dict of {agent_id: {task_id: response}}
    """
    responses = {}
    with open(responses_path) as f:
        for line in f:
            record = json.loads(line)
            agent_id = record["subject_id"]
            responses[agent_id] = record["responses"]
    return responses


def filter_responses_to_tasks(
    responses: dict,
    task_ids: list,
) -> dict:
    """Filter response matrix to only include specified tasks.

    Args:
        responses: Full response matrix
        task_ids: List of task IDs to keep

    Returns:
        Filtered response matrix
    """
    task_set = set(task_ids)
    filtered = {}
    for agent_id, agent_responses in responses.items():
        filtered_agent = {
            task_id: response
            for task_id, response in agent_responses.items()
            if task_id in task_set
        }
        if filtered_agent:  # Only include agents with at least one response
            filtered[agent_id] = filtered_agent
    return filtered


def save_filtered_responses(
    responses: dict,
    output_path: Path,
    task_ids: list,
    is_binomial: bool = False,
):
    """Save filtered responses as JSONL in py_irt format.

    Args:
        responses: Filtered response matrix
        output_path: Path to write JSONL
        task_ids: Complete list of task IDs (for complete matrix)
        is_binomial: If True, use binomial format for missing values
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for agent_id, agent_responses in responses.items():
            # Create complete matrix
            if is_binomial:
                complete_responses = {
                    task_id: agent_responses.get(task_id, {"successes": 0, "trials": 0})
                    for task_id in task_ids
                }
            else:
                complete_responses = {
                    task_id: agent_responses.get(task_id, 0)
                    for task_id in task_ids
                }
            record = {
                "subject_id": agent_id,
                "responses": complete_responses,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(responses)} agents to {output_path}")


def get_or_train_split_irt(
    responses_path: Path,
    output_base: Path,
    test_fraction: float = 0.2,
    split_seed: int = 0,
    model_type: str = "1pl",
    epochs: int = 2000,
    force_retrain: bool = False,
    dry_run: bool = False,
    is_binomial: bool = False,
    train_tasks: Optional[List[str]] = None,
    fold_idx: Optional[int] = None,
    k_folds: Optional[int] = None,
) -> Path:
    """Get cached IRT model or train a new one for the specified split.

    Works for both Bernoulli (SWE-bench) and Binomial (TerminalBench) data.

    For k-fold cross-validation, provide train_tasks, fold_idx, and k_folds.
    For single holdout (legacy), omit these parameters.

    Args:
        responses_path: Path to full response matrix JSONL
        output_base: Base directory for cached IRT models
        test_fraction: Fraction of tasks to hold out (ignored if train_tasks provided)
        split_seed: Random seed for split
        model_type: IRT model type ("1pl" or "2pl")
        epochs: Training epochs
        force_retrain: If True, retrain even if cached
        dry_run: If True, just print what would be done
        is_binomial: If True, use binomial IRT (for TerminalBench)
        train_tasks: For k-fold CV, the explicit list of train task IDs
        fold_idx: For k-fold CV, the fold index (0 to k-1)
        k_folds: For k-fold CV, the total number of folds

    Returns:
        Path to IRT output directory (contains abilities.csv, items.csv, split_info.json)
    """
    cache_dir = get_split_cache_dir(
        output_base, test_fraction, split_seed, model_type, is_binomial,
        fold_idx=fold_idx, k_folds=k_folds
    )

    # Check for cached model
    if not force_retrain and check_cached_irt(cache_dir):
        cached_info = load_cached_split_info(cache_dir)
        data_type = "binomial" if is_binomial else "Bernoulli"
        print(f"Found cached {data_type} IRT model at {cache_dir}")
        print(f"  Split seed: {cached_info.get('split_seed')}")
        print(f"  Test fraction: {cached_info.get('test_fraction')}")
        print(f"  Train tasks: {cached_info.get('n_train_tasks')}")
        print(f"  Test tasks: {cached_info.get('n_test_tasks')}")
        return cache_dir

    data_type = "BINOMIAL" if is_binomial else "BERNOULLI"
    print("=" * 60)
    print(f"TRAIN {data_type} IRT ON TRAIN SPLIT (NO DATA LEAKAGE)")
    print("=" * 60)

    # Load full response matrix
    print(f"\n1. Loading responses from {responses_path}...")
    responses = load_response_matrix(responses_path)
    print(f"   Loaded {len(responses)} agents")

    # Get all task IDs
    all_tasks = set()
    for agent_responses in responses.values():
        all_tasks.update(agent_responses.keys())
    all_tasks = sorted(all_tasks)
    print(f"   Found {len(all_tasks)} tasks")

    # Determine train/test split
    if train_tasks is not None:
        # For k-fold CV: use explicitly provided train tasks
        print(f"\n2. Using provided train tasks (fold {fold_idx + 1}/{k_folds})...")
        test_tasks = [t for t in all_tasks if t not in set(train_tasks)]
        print(f"   Train tasks: {len(train_tasks)}")
        print(f"   Test tasks: {len(test_tasks)}")
    else:
        # Legacy: compute split from test_fraction
        print(f"\n2. Splitting tasks (test_fraction={test_fraction}, seed={split_seed})...")
        train_tasks, test_tasks = stable_split_tasks(all_tasks, test_fraction, split_seed)
        print(f"   Train tasks: {len(train_tasks)}")
        print(f"   Test tasks: {len(test_tasks)}")

    # Filter responses to train tasks only
    print("\n3. Filtering responses to train tasks...")
    train_responses = filter_responses_to_tasks(responses, train_tasks)
    print(f"   Agents with train responses: {len(train_responses)}")

    # Count responses
    n_responses = sum(len(r) for r in train_responses.values())
    print(f"   Total train responses: {n_responses}")

    # Output paths
    train_responses_path = cache_dir.parent / "train_responses.jsonl"

    print(f"\n4. Output paths:")
    print(f"   Train responses: {train_responses_path}")
    print(f"   IRT model output: {cache_dir}")

    if dry_run:
        print(f"\n[DRY RUN] Would train {data_type.lower()} IRT on train tasks only")
        print(f"   Training {model_type.upper()} model for {epochs} epochs")
        return cache_dir

    # Save filtered responses
    print("\n5. Saving filtered responses...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_filtered_responses(train_responses, train_responses_path, train_tasks, is_binomial)

    # Train IRT model
    print(f"\n6. Training {model_type.upper()} IRT model...")

    if is_binomial:
        # Import and run binomial IRT training
        from swebench_irt.train_binomial import (
            load_count_data,
            fit_1d_binomial_1pl,
            fit_1d_binomial,
        )

        subjects, items, counts, trials, subject_ids, item_ids = load_count_data(
            str(train_responses_path)
        )
        print(f"   Dataset: {len(subject_ids)} subjects, {len(item_ids)} items")
        print(f"   Observations: {len(counts)}")

        if model_type == "1pl":
            fit_1d_binomial_1pl(
                subjects, items, counts, trials, subject_ids, item_ids,
                epochs=epochs, output_dir=cache_dir.parent
            )
        else:
            fit_1d_binomial(
                subjects, items, counts, trials, subject_ids, item_ids,
                epochs=epochs, output_dir=cache_dir.parent
            )
    else:
        # Import and run standard IRT training
        from py_irt.config import IrtConfig
        from py_irt.training import IrtModelTrainer

        # Configure and train
        config = IrtConfig(
            model_type=model_type,
            epochs=epochs,
            priors="hierarchical",
            dims=1,
        )

        trainer = IrtModelTrainer(
            data_path=train_responses_path,
            config=config,
        )
        n_subjects = len(trainer._dataset.subject_ids)
        n_items = len(trainer._dataset.item_ids)
        print(f"   Dataset: {n_subjects} subjects, {n_items} items")

        trainer.train(device="cpu")

        # Save results
        print("\n7. Saving IRT parameters...")

        # Get parameters from best_params
        abilities = trainer.best_params["ability"]
        difficulties = trainer.best_params["diff"]
        item_ids = trainer.best_params["item_ids"]
        subject_ids = trainer.best_params["subject_ids"]

        # Map to original IDs - item_ids and subject_ids are dicts {idx: id}
        abilities_df = pd.DataFrame({
            "theta": [abilities[i] for i in range(len(subject_ids))],
        }, index=[subject_ids[i] for i in range(len(subject_ids))])
        abilities_df.index.name = "subject_id"
        abilities_df.to_csv(cache_dir / "abilities.csv")

        items_df = pd.DataFrame({
            "b": [difficulties[i] for i in range(len(item_ids))],
        }, index=[item_ids[i] for i in range(len(item_ids))])
        items_df.index.name = "item_id"
        items_df.to_csv(cache_dir / "items.csv")

    # Save split info for cache validation
    split_info = {
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "model_type": model_type,
        "epochs": epochs,
        "responses_path": str(responses_path),
        "is_binomial": is_binomial,
    }
    with open(cache_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n   Saved abilities: {cache_dir / 'abilities.csv'}")
    print(f"   Saved items: {cache_dir / 'items.csv'}")
    print(f"   Saved split info: {cache_dir / 'split_info.json'}")

    print("\n" + "=" * 60)
    print("DONE - Use these IRT parameters for uncontaminated training")
    print("=" * 60)

    return cache_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train IRT model on train tasks only to avoid data leakage"
    )
    parser.add_argument(
        "--responses_path",
        type=Path,
        default=Path("clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"),
        help="Path to full response matrix JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/experiment_a/irt_splits"),
        help="Output directory for split IRT models",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of tasks for test set",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="1pl",
        choices=["1pl", "2pl"],
        help="IRT model type",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Training epochs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if cached model exists",
    )
    parser.add_argument(
        "--binomial",
        action="store_true",
        help="Use binomial IRT (for TerminalBench data)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without training",
    )
    args = parser.parse_args()

    # Resolve paths relative to ROOT
    responses_path = ROOT / args.responses_path
    output_dir = ROOT / args.output_dir

    get_or_train_split_irt(
        responses_path=responses_path,
        output_base=output_dir,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        model_type=args.model_type,
        epochs=args.epochs,
        force_retrain=args.force,
        dry_run=args.dry_run,
        is_binomial=args.binomial,
    )


if __name__ == "__main__":
    main()
