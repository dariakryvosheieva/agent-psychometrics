"""Train IRT model on train tasks only to avoid data leakage.

The standard experiment_a pipeline uses IRT difficulties computed on ALL tasks,
which creates data leakage: the ground truth `b` values for training the
difficulty predictor are influenced by test task responses.

This module trains a separate IRT model on just the train tasks for each
k-fold CV split, producing uncontaminated difficulty estimates for use as
training targets.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd


def set_torch_determinism(enabled: bool) -> None:
    """Toggle PyTorch deterministic algorithm behavior (best-effort).

    IRT training can be numerically unstable with deterministic algorithms
    enabled. This function temporarily disables determinism during IRT training.

    Copied from predict_question_difficulty.py - see that file for full rationale.
    """
    import torch
    on = bool(enabled)
    try:
        torch.use_deterministic_algorithms(on, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(on)
        except Exception:
            pass
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = on
        torch.backends.cudnn.benchmark = (not on)
    except Exception:
        pass


def get_split_cache_dir(
    output_base: Path,
    split_seed: int,
    fold_idx: int,
    k_folds: int,
    model_type: str = "1pl",
    exclude_unsolved: bool = False,
) -> Path:
    """Get the cache directory for a specific k-fold split configuration.

    Args:
        output_base: Base output directory
        split_seed: Split random seed
        fold_idx: Fold index (0 to k-1)
        k_folds: Total number of folds
        model_type: IRT model type
        exclude_unsolved: Whether unsolved tasks were filtered out

    Returns:
        Path to cache directory for this configuration
    """
    suffix = "_filtered" if exclude_unsolved else ""
    split_name = f"seed{split_seed}_fold{fold_idx}of{k_folds}_{model_type}{suffix}"
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
):
    """Save filtered responses as JSONL in py_irt format.

    Args:
        responses: Filtered response matrix
        output_path: Path to write JSONL
        task_ids: Complete list of task IDs (for complete matrix)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for agent_id, agent_responses in responses.items():
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
    train_tasks: List[str],
    fold_idx: int,
    k_folds: int,
    split_seed: int = 0,
    model_type: str = "1pl",
    epochs: int = 2000,
    force_retrain: bool = False,
    dry_run: bool = False,
    exclude_unsolved: bool = False,
) -> Path:
    """Get cached IRT model or train a new one for the specified k-fold split.

    Args:
        responses_path: Path to full response matrix JSONL
        output_base: Base directory for cached IRT models
        train_tasks: Explicit list of train task IDs for this fold
        fold_idx: Fold index (0 to k-1)
        k_folds: Total number of folds
        split_seed: Random seed for split (used for cache naming)
        model_type: IRT model type ("1pl" or "2pl")
        epochs: Training epochs
        force_retrain: If True, retrain even if cached
        dry_run: If True, just print what would be done
        exclude_unsolved: If True, unsolved tasks were filtered out (affects cache key)

    Returns:
        Path to IRT output directory (contains abilities.csv, items.csv, split_info.json)
    """
    cache_dir = get_split_cache_dir(
        output_base, split_seed, fold_idx, k_folds,
        model_type=model_type, exclude_unsolved=exclude_unsolved
    )

    # Check for cached model
    if not force_retrain and check_cached_irt(cache_dir):
        cached_info = load_cached_split_info(cache_dir)

        # Verify cache was trained on the same response matrix
        cached_responses_path = cached_info.get("responses_path", "")
        if cached_responses_path and str(responses_path) != cached_responses_path:
            print(f"Cache invalidated: response matrix changed")
            print(f"  Cached: {cached_responses_path}")
            print(f"  Current: {responses_path}")
            # Fall through to retrain
        else:
            print(f"Found cached IRT model at {cache_dir}")
            print(f"  Split seed: {cached_info.get('split_seed')}")
            print(f"  Train tasks: {cached_info.get('n_train_tasks')}")
            print(f"  Test tasks: {cached_info.get('n_test_tasks')}")
            return cache_dir

    print("=" * 60)
    print("TRAIN IRT ON TRAIN SPLIT (NO DATA LEAKAGE)")
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
    if train_tasks is None:
        raise ValueError("train_tasks must be provided (use load_dataset_for_fold for k-fold CV)")
    print(f"\n2. Using provided train tasks (fold {fold_idx + 1}/{k_folds})...")
    test_tasks = [t for t in all_tasks if t not in set(train_tasks)]
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
        print(f"\n[DRY RUN] Would train IRT on train tasks only")
        print(f"   Training {model_type.upper()} model for {epochs} epochs")
        return cache_dir

    # Save filtered responses
    print("\n5. Saving filtered responses...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_filtered_responses(train_responses, train_responses_path, train_tasks)

    # Train IRT model
    print(f"\n6. Training {model_type.upper()} IRT model...")

    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    # Configure and train (seed=0 matches predict_question_difficulty.py for reproducibility)
    config = IrtConfig(
        model_type=model_type,
        epochs=epochs,
        priors="hierarchical",
        dims=1,
        seed=0,
    )

    trainer = IrtModelTrainer(
        data_path=train_responses_path,
        config=config,
    )
    n_subjects = len(trainer._dataset.subject_ids)
    n_items = len(trainer._dataset.item_ids)
    print(f"   Dataset: {n_subjects} subjects, {n_items} items")

    # Disable torch determinism during IRT for numerical stability
    # (see predict_question_difficulty.py for rationale)
    set_torch_determinism(False)
    trainer.train(device="cpu")
    set_torch_determinism(True)

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
        "split_seed": split_seed,
        "fold_idx": fold_idx,
        "k_folds": k_folds,
        "n_train_tasks": len(train_tasks),
        "n_test_tasks": len(test_tasks),
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "model_type": model_type,
        "epochs": epochs,
        "responses_path": str(responses_path),
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
