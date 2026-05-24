"""Seed-level bootstrap for method-vs-baseline mean differences.

Resamples per-seed mean differences (one per shuffled fold split) with
replacement to produce a percentile CI and a two-sided p-value for the null
that the mean difference is zero. The fold seed is the resampling unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_is_upper_bound: bool
    n_bootstrap: int
    ci: float


def bootstrap_seed_mean_differences(
    seed_mean_differences: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap the mean of per-seed differences vs. a baseline.

    Resamples `n_bootstrap` times with replacement from the observed
    `seed_mean_differences` (length N), computes the mean per replicate, and
    returns a percentile CI plus a two-sided p-value for H0: mean = 0.

    The p-value uses an add-one correction:
        p = min(1, 2 * (min(count(boot <= 0), count(boot >= 0)) + 1) / (B + 1))
    so a finite Monte Carlo run never reports an exact p=0. If no bootstrap
    sample crosses zero, `p_value_is_upper_bound=True` and callers should
    render the printed p-value with a leading "<".
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    arr = np.asarray(list(seed_mean_differences), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("seed_mean_differences must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("seed_mean_differences must all be finite")

    n_seeds = arr.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
    boot_means = arr[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    n_le = int(np.sum(boot_means <= 0.0))
    n_ge = int(np.sum(boot_means >= 0.0))
    tail = min(n_le, n_ge)
    p_value = min(1.0, 2.0 * (tail + 1) / (n_bootstrap + 1))
    p_value_is_upper_bound = tail == 0

    return BootstrapResult(
        mean_difference=float(arr.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=float(p_value),
        p_value_is_upper_bound=bool(p_value_is_upper_bound),
        n_bootstrap=int(n_bootstrap),
        ci=float(ci),
    )
