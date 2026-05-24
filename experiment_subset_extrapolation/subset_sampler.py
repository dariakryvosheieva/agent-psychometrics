"""Random subset sampling for the subset extrapolation experiment."""

from typing import List, Tuple

import numpy as np


def sample_subset_by_count(
    all_task_ids: List[str], count: int, seed: int
) -> Tuple[List[str], List[str]]:
    """Sample `count` tasks at random (without replacement).

    Args:
        all_task_ids: The full task universe (ordered).
        count: Number of tasks to put in the observed subset (>= 1).
        seed: Random seed (each (count, seed) yields a deterministic split).

    Returns:
        Tuple (observed_task_ids, heldout_task_ids). Order within each list
        follows the input order.
    """
    n_total = len(all_task_ids)
    if n_total == 0:
        raise ValueError("all_task_ids is empty")
    n_obs = int(count)
    if n_obs < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if n_obs > n_total:
        raise ValueError(f"count={count} exceeds n_total={n_total}")

    rng = np.random.RandomState(seed)
    obs_idx = set(int(i) for i in rng.choice(n_total, size=n_obs, replace=False))
    observed = [t for i, t in enumerate(all_task_ids) if i in obs_idx]
    heldout = [t for i, t in enumerate(all_task_ids) if i not in obs_idx]
    return observed, heldout
