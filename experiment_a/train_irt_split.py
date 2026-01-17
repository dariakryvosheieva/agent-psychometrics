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
) -> Path:
    """Get the cache directory for a specific split configuration.

    Args:
        output_base: Base output directory
        test_fraction: Test fraction (e.g., 0.2)
        split_seed: Split random seed
        model_type: IRT model type
        is_binomial: Whether this is binomial (TerminalBench) or Bernoulli (SWE-bench)

    Returns:
        Path to cache directory for this configuration
    """
    suffix = "_binomial" if is_binomial else ""
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
) -> Path:
    """Get cached IRT model or train a new one for the specified split.

    Works for both Bernoulli (SWE-bench) and Binomial (TerminalBench) data.

    Args:
        responses_path: Path to full response matrix JSONL
        output_base: Base directory for cached IRT models
        test_fraction: Fraction of tasks to hold out
        split_seed: Random seed for split
        model_type: IRT model type ("1pl" or "2pl")
        epochs: Training epochs
        force_retrain: If True, retrain even if cached
        dry_run: If True, just print what would be done
        is_binomial: If True, use binomial IRT (for TerminalBench)

    Returns:
        Path to IRT output directory (contains abilities.csv, items.csv, split_info.json)
    """
    cache_dir = get_split_cache_dir(output_base, test_fraction, split_seed, model_type, is_binomial)

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

    # Split tasks
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
        from swebench_irt.train_binomial import train_binomial_irt
        train_binomial_irt(
            data_path=train_responses_path,
            output_dir=cache_dir.parent,
            model_type=model_type,
            epochs=epochs,
            dims=1,
        )
    else:
        # Import and run standard IRT training
        from py_irt.dataset import Dataset
        from py_irt.config import IrtConfig
        from py_irt.training import IrtModelTrainer

        if model_type == "1pl":
            from py_irt.models import OneParamLog as IrtModel
        else:
            from py_irt.models import TwoParamLog as IrtModel

        # Load dataset
        dataset = Dataset.from_jsonlines(str(train_responses_path))
        print(f"   Dataset: {dataset.num_subjects} subjects, {dataset.num_items} items")

        # Configure and train
        config = IrtConfig(
            model_type=model_type,
            epochs=epochs,
            priors="hierarchical",
            dims=1,
        )

        trainer = IrtModelTrainer(
            config=config,
            data=dataset,
            model=IrtModel(
                priors=config.priors,
                device=config.device,
                num_items=dataset.num_items,
                num_subjects=dataset.num_subjects,
                dims=1,
            ),
        )

        trainer.train(device="cpu")

        # Save results
        print("\n7. Saving IRT parameters...")

        # Get parameters
        abilities = trainer.model.export("ability")
        difficulties = trainer.model.export("diff")

        # Map to original IDs
        abilities_df = pd.DataFrame({
            "subject_id": dataset.subject_ids,
            "theta": abilities.flatten(),
        }).set_index("subject_id")
        abilities_df.to_csv(cache_dir / "abilities.csv")

        items_df = pd.DataFrame({
            "item_id": dataset.item_ids,
            "b": difficulties.flatten(),
        }).set_index("item_id")
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
