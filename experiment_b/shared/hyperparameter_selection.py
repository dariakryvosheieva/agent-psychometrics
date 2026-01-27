"""Hyperparameter selection via held-out response AUC.

This module provides a unified utility for selecting hyperparameters for any
IRT-based method (Feature-IRT, Ridge predictors, etc.) using held-out
response pairs and AUC as the selection criterion.

The key insight is that for IRT models, we can:
1. Hold out a fraction of (agent, task) response pairs
2. Train on the remaining pairs (ensuring all agents and tasks are seen)
3. Evaluate AUC on held-out pairs using predicted probabilities
4. Select hyperparameters that maximize held-out AUC
"""

import itertools
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def stratified_pair_split(
    responses: Dict[str, Dict[str, int]],
    agent_ids: List[str],
    task_ids: List[str],
    val_frac: float = 0.2,
    random_state: int = 42,
    max_retries: int = 100,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Split response matrix into train/val, ensuring all agents and tasks appear in train.

    Randomly holds out val_frac of (agent, task) pairs. If this results in any
    agent or task being completely absent from the training set, retries with
    a different random split.

    Args:
        responses: Response matrix {agent_id: {task_id: response}}
        agent_ids: List of all agent IDs to consider
        task_ids: List of all task IDs to consider
        val_frac: Fraction of pairs to hold out for validation (default 0.2)
        random_state: Random seed for reproducibility
        max_retries: Maximum number of retries before raising an error

    Returns:
        train_responses: {agent_id: {task_id: response}} for training
        val_responses: {agent_id: {task_id: response}} for validation

    Raises:
        ValueError: If unable to find a valid split after max_retries
    """
    rng = np.random.RandomState(random_state)
    agent_set = set(agent_ids)
    task_set = set(task_ids)

    # Collect all (agent, task, response) tuples
    all_pairs = []
    for agent in agent_ids:
        if agent not in responses:
            continue
        for task in task_ids:
            if task in responses[agent]:
                resp = responses[agent][task]
                # Handle binomial responses
                if isinstance(resp, dict) and "successes" in resp:
                    resp = 1 if resp["successes"] > 0 else 0
                all_pairs.append((agent, task, resp))

    if not all_pairs:
        raise ValueError("No response pairs found for the given agents and tasks")

    n_val = int(len(all_pairs) * val_frac)
    if n_val < 1:
        raise ValueError(f"val_frac={val_frac} results in 0 validation pairs")

    for attempt in range(max_retries):
        # Shuffle and split
        rng.shuffle(all_pairs)
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]

        # Check coverage
        train_agents = set(p[0] for p in train_pairs)
        train_tasks = set(p[1] for p in train_pairs)

        missing_agents = agent_set - train_agents
        missing_tasks = task_set - train_tasks

        if not missing_agents and not missing_tasks:
            # Valid split found
            break
    else:
        raise ValueError(
            f"Failed to find valid train/val split after {max_retries} attempts. "
            f"Missing agents: {len(missing_agents)}, missing tasks: {len(missing_tasks)}. "
            f"Try reducing val_frac or ensuring sufficient response coverage."
        )

    # Convert back to response dict format
    train_responses: Dict[str, Dict[str, int]] = {}
    for agent, task, resp in train_pairs:
        if agent not in train_responses:
            train_responses[agent] = {}
        train_responses[agent][task] = resp

    val_responses: Dict[str, Dict[str, int]] = {}
    for agent, task, resp in val_pairs:
        if agent not in val_responses:
            val_responses[agent] = {}
        val_responses[agent][task] = resp

    return train_responses, val_responses


def fit_with_cv_hyperparams(
    train_fn: Callable[[Dict[str, Any], Dict[str, Dict[str, int]]], Tuple[Dict[str, float], Dict[str, float]]],
    hyperparam_grid: Dict[str, List[Any]],
    responses: Dict[str, Dict[str, int]],
    agent_ids: List[str],
    task_ids: List[str],
    val_frac: float = 0.2,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Tuple[Dict[str, float], Dict[str, float]]]:
    """Select hyperparameters via held-out AUC, then train on full dataset.

    This is the unified utility for any IRT-based method with hyperparameters.
    It performs grid search over hyperparameter combinations, evaluating each
    by training on a subset of response pairs and computing AUC on held-out pairs.

    Args:
        train_fn: Function (hyperparams, responses) -> (abilities, difficulties)
            - hyperparams: Dict of hyperparameter values
            - responses: Response matrix {agent: {task: response}}
            - Returns: (abilities dict {agent: theta}, difficulties dict {task: beta})
        hyperparam_grid: Grid of hyperparameters to search, e.g.,
            {"l2_weight": [0.001, 0.01, 0.1], "l2_residual": [1.0, 10.0]}
        responses: Full response matrix {agent: {task: 0|1}}
        agent_ids: List of agent IDs to use
        task_ids: List of task IDs to use
        val_frac: Fraction of response pairs to hold out for validation
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        random_state: Random seed for reproducibility
        verbose: Print progress and best hyperparameters

    Returns:
        best_params: Selected hyperparameters dict
        (abilities, difficulties): From final training on full dataset
    """
    # 1. Split responses into train/val (all agents/tasks appear in train)
    train_responses, val_responses = stratified_pair_split(
        responses, agent_ids, task_ids, val_frac, random_state
    )

    if verbose:
        n_train = sum(len(tasks) for tasks in train_responses.values())
        n_val = sum(len(tasks) for tasks in val_responses.values())
        print(f"Hyperparameter CV: {n_train} train pairs, {n_val} val pairs")

    # 2. Define evaluation function for a single hyperparameter combination
    def evaluate_params(params: Dict[str, Any]) -> float:
        """Train and evaluate one hyperparameter combination."""
        abilities, difficulties = train_fn(params, train_responses)

        # Compute AUC on val_responses
        y_true, y_pred = [], []
        for agent in val_responses:
            if agent not in abilities:
                raise ValueError(f"Agent {agent} not in learned abilities")
            for task, response in val_responses[agent].items():
                if task not in difficulties:
                    raise ValueError(f"Task {task} not in learned difficulties")
                prob = sigmoid(abilities[agent] - difficulties[task])
                y_true.append(response)
                y_pred.append(prob)

        if len(y_true) < 2:
            raise ValueError(f"Only {len(y_true)} validation pairs - need at least 2")
        if len(set(y_true)) < 2:
            raise ValueError(
                f"Validation set has only one class (all {y_true[0]}s) - cannot compute AUC"
            )

        return roc_auc_score(y_true, y_pred)

    # 3. Generate all parameter combinations
    param_names = list(hyperparam_grid.keys())
    param_combos = [
        dict(zip(param_names, combo))
        for combo in itertools.product(*hyperparam_grid.values())
    ]

    if verbose:
        print(f"Grid search over {len(param_combos)} hyperparameter combinations...")

    # 4. Parallel evaluation using joblib with progress bar
    from joblib import Parallel, delayed

    grid_start_time = time.time()

    # Use tqdm for progress tracking with joblib
    # joblib doesn't directly support tqdm, so we use batch_size to get periodic updates
    with tqdm(total=len(param_combos), desc="Grid search", disable=not verbose) as pbar:
        # Define wrapper to update progress bar
        def evaluate_with_progress(params: Dict[str, Any]) -> float:
            result = evaluate_params(params)
            return result

        # Run parallel jobs - tqdm updates after all complete
        # For better progress, we can use smaller batch sizes
        aucs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(evaluate_with_progress)(params) for params in param_combos
        )
        pbar.update(len(param_combos))

    grid_elapsed = time.time() - grid_start_time

    # 5. Find best hyperparameters
    best_idx = int(np.argmax(aucs))
    best_params = param_combos[best_idx]
    best_auc = aucs[best_idx]

    if verbose:
        # Print timing and AUC statistics
        auc_array = np.array(aucs)
        print(f"Grid search completed in {grid_elapsed:.1f}s "
              f"({grid_elapsed/len(param_combos):.2f}s per combo)")
        print(f"  AUC range: {auc_array.min():.4f} - {auc_array.max():.4f} "
              f"(mean: {auc_array.mean():.4f}, std: {auc_array.std():.4f})")
        print(f"Best hyperparams (val AUC={best_auc:.4f}): {best_params}")

    # 6. Final training on full dataset with best hyperparams
    if verbose:
        print("Training final model on full dataset...")
    final_start_time = time.time()
    abilities, difficulties = train_fn(best_params, responses)
    final_elapsed = time.time() - final_start_time

    if verbose:
        print(f"Final training completed in {final_elapsed:.1f}s")

    return best_params, (abilities, difficulties)


# Default hyperparameter grids
# Full grid (216 combos for single, 1296 for grouped with 2 sources)
L2_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Coarse grid for faster initial search (27 combos for single, 81 for grouped)
# Based on empirical findings: l2_weight ~0.01, l2_residual ~1.0, l2_ability ~0.01
COARSE_L2_GRID = [0.01, 1.0, 100.0]

SINGLE_SOURCE_GRID = {
    "l2_weight": COARSE_L2_GRID,
    "l2_residual": COARSE_L2_GRID,
    "l2_ability": COARSE_L2_GRID,
}

SINGLE_SOURCE_GRID_FINE = {
    "l2_weight": L2_GRID,
    "l2_residual": L2_GRID,
    "l2_ability": L2_GRID,
}


def make_grouped_source_grid(source_names: List[str], fine: bool = False) -> Dict[str, List[Any]]:
    """Create hyperparameter grid for grouped feature sources.

    For grouped sources, we have one alpha per source (replacing l2_weight),
    plus l2_residual and l2_ability.

    Args:
        source_names: List of source names, e.g., ["Embedding", "Trajectory"]
        fine: If True, use fine grid (6 values). If False, use coarse grid (3 values).

    Returns:
        Hyperparameter grid dict
    """
    l2_grid = L2_GRID if fine else COARSE_L2_GRID
    grid = {}
    for name in source_names:
        grid[f"alpha_{name}"] = l2_grid
    grid["l2_residual"] = l2_grid
    grid["l2_ability"] = l2_grid
    return grid
