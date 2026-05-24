"""Random subset sampling for the subset extrapolation experiment."""

from typing import List, Tuple

import numpy as np


def sample_subset(
    all_task_ids: List[str], subset_size: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Sample a random subset of tasks.

    Args:
        all_task_ids: The full task universe (ordered).
        subset_size: Fraction in (0, 1] of tasks to put in the observed subset.
        seed: Random seed (each (subset_size, seed) yields a deterministic split).

    Returns:
        Tuple (observed_task_ids, heldout_task_ids). Order within each list
        follows the input order.
    """
    if not 0.0 < subset_size <= 1.0:
        raise ValueError(f"subset_size must be in (0, 1], got {subset_size}")
    n_total = len(all_task_ids)
    if n_total == 0:
        raise ValueError("all_task_ids is empty")

    n_obs = max(1, int(round(subset_size * n_total)))
    n_obs = min(n_obs, n_total)

    rng = np.random.RandomState(seed)
    obs_idx = set(int(i) for i in rng.choice(n_total, size=n_obs, replace=False))
    observed = [t for i, t in enumerate(all_task_ids) if i in obs_idx]
    heldout = [t for i, t in enumerate(all_task_ids) if i not in obs_idx]
    return observed, heldout
