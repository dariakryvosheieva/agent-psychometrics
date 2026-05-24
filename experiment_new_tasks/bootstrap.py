"""Seed-level bootstrap for method-vs-baseline mean differences.

Resamples per-seed mean differences (one per shuffled fold split) with
replacement to produce a percentile CI and a two-sided p-value for the null
that the mean difference is zero. The fold seed is the resampling unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class BootstrapAUCResult:
    """Result of a paired clustered bootstrap comparison."""

    auc: float
    baseline_auc: float
    delta_auc: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_is_upper_bound: bool
    n_bootstrap: int
    n_observations: int
    n_clusters: int
    cluster_key: str
    skipped_bootstrap_samples: int


@dataclass
class SeedMeanDifferenceBootstrapResult:
    """Bootstrap result for seed-level mean AUC differences."""

    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_is_upper_bound: bool
    n_bootstrap: int
    n_seeds: int
    ci: float


def _auc_or_none(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Return ROC-AUC, or None when a sample has only one class."""
    if len(y_true) < 2 or len(set(int(y) for y in y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _align_prediction_records(
    records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Mapping[str, Any]]]:
    """Align two prediction record sets by observation key.

    Each record is expected to contain ``agent_id``, ``task_id``, ``y_true``, and
    ``y_score``. The pair (agent_id, task_id) is the observation identity.
    """
    by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for rec in records:
        key = (str(rec["agent_id"]), str(rec["task_id"]))
        if key in by_key:
            raise ValueError(f"Duplicate prediction record for {key}")
        by_key[key] = rec

    baseline_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for rec in baseline_records:
        key = (str(rec["agent_id"]), str(rec["task_id"]))
        if key in baseline_by_key:
            raise ValueError(f"Duplicate baseline prediction record for {key}")
        baseline_by_key[key] = rec

    missing_baseline = sorted(set(by_key) - set(baseline_by_key))
    missing_method = sorted(set(baseline_by_key) - set(by_key))
    if missing_baseline or missing_method:
        raise ValueError(
            "Prediction records must be paired exactly; "
            f"missing_baseline={len(missing_baseline)}, missing_method={len(missing_method)}"
        )

    aligned_records: List[Mapping[str, Any]] = []
    y_true: List[int] = []
    y_score: List[float] = []
    y_score_baseline: List[float] = []
    for key in sorted(by_key):
        rec = by_key[key]
        base_rec = baseline_by_key[key]
        y = int(rec["y_true"])
        y_base = int(base_rec["y_true"])
        if y != y_base:
            raise ValueError(f"Mismatched labels for {key}: {y} != {y_base}")
        aligned_records.append(rec)
        y_true.append(y)
        y_score.append(float(rec["y_score"]))
        y_score_baseline.append(float(base_rec["y_score"]))

    return (
        np.asarray(y_true, dtype=np.int8),
        np.asarray(y_score, dtype=np.float64),
        np.asarray(y_score_baseline, dtype=np.float64),
        aligned_records,
    )


def paired_clustered_auc_bootstrap(
    records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    *,
    cluster_key: str = "task_id",
    n_bootstrap: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> BootstrapAUCResult:
    """Compare paired out-of-fold AUCs with a clustered bootstrap.

    The bootstrap resamples clusters with replacement, keeps all observations in
    a sampled cluster together, and recomputes ``AUC(method) - AUC(baseline)``.
    This is appropriate for New Tasks CV where observations from the same held-
    out task share the same train/test split and task-level difficulty estimate.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be between 0 and 1, got {ci}")

    y_true, y_score, y_score_baseline, aligned_records = _align_prediction_records(
        records, baseline_records
    )
    auc = _auc_or_none(y_true.tolist(), y_score.tolist())
    baseline_auc = _auc_or_none(y_true.tolist(), y_score_baseline.tolist())
    if auc is None or baseline_auc is None:
        raise ValueError("Cannot compute AUC comparison: predictions contain fewer than two classes")
    delta_auc = float(auc - baseline_auc)

    clusters: Dict[str, List[int]] = {}
    for idx, rec in enumerate(aligned_records):
        if cluster_key not in rec:
            raise ValueError(f"Prediction record is missing cluster key {cluster_key!r}")
        clusters.setdefault(str(rec[cluster_key]), []).append(idx)
    if not clusters:
        raise ValueError("No prediction clusters available for bootstrap")

    cluster_ids = np.asarray(sorted(clusters.keys()), dtype=object)
    cluster_indices = [np.asarray(clusters[str(cid)], dtype=np.int64) for cid in cluster_ids]

    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    skipped = 0
    for _ in range(int(n_bootstrap)):
        sampled_cluster_positions = rng.integers(
            low=0,
            high=len(cluster_indices),
            size=len(cluster_indices),
        )
        sampled_indices = np.concatenate(
            [cluster_indices[int(pos)] for pos in sampled_cluster_positions]
        )
        sample_y = y_true[sampled_indices]
        sample_auc = _auc_or_none(sample_y.tolist(), y_score[sampled_indices].tolist())
        sample_baseline_auc = _auc_or_none(
            sample_y.tolist(), y_score_baseline[sampled_indices].tolist()
        )
        if sample_auc is None or sample_baseline_auc is None:
            skipped += 1
            continue
        deltas.append(float(sample_auc - sample_baseline_auc))

    if not deltas:
        raise ValueError("All bootstrap samples were degenerate; cannot compute CI or p-value")

    delta_arr = np.asarray(deltas, dtype=np.float64)
    alpha = 1.0 - float(ci)
    ci_low, ci_high = np.quantile(delta_arr, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Two-sided percentile bootstrap p-value for H0: delta == 0. The add-one
    # correction avoids reporting p=0 when no resample crosses zero; in that
    # case, the result is only bounded by the Monte Carlo resolution.
    count_lower = int(np.sum(delta_arr <= 0.0))
    count_upper = int(np.sum(delta_arr >= 0.0))
    tail_count = min(count_lower, count_upper)
    p_value = min(1.0, 2.0 * float(tail_count + 1) / float(len(delta_arr) + 1))
    p_value_is_upper_bound = tail_count == 0

    return BootstrapAUCResult(
        auc=float(auc),
        baseline_auc=float(baseline_auc),
        delta_auc=float(delta_auc),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        p_value_is_upper_bound=bool(p_value_is_upper_bound),
        n_bootstrap=int(len(deltas)),
        n_observations=int(len(y_true)),
        n_clusters=int(len(cluster_ids)),
        cluster_key=str(cluster_key),
        skipped_bootstrap_samples=int(skipped),
    )


def bootstrap_seed_mean_differences(
    seed_mean_differences: Sequence[float],
    *,
    n_bootstrap: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> SeedMeanDifferenceBootstrapResult:
    """Bootstrap the mean of per-seed mean AUC differences.

    Each input value is the mean paired fold difference
    ``AUC(method) - AUC(baseline)`` from one fold seed. Bootstrap samples draw
    ``N`` seeds with replacement from the ``N`` observed fold seeds.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be between 0 and 1, got {ci}")

    differences = np.asarray(seed_mean_differences, dtype=np.float64)
    if differences.size == 0:
        raise ValueError("At least one seed mean difference is required")
    if not np.all(np.isfinite(differences)):
        raise ValueError("Seed mean differences must all be finite")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        low=0,
        high=differences.size,
        size=(int(n_bootstrap), differences.size),
    )
    bootstrap_means = differences[sample_indices].mean(axis=1)

    alpha = 1.0 - float(ci)
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Two-sided bootstrap p-value for H0: mean difference == 0. The add-one
    # correction avoids reporting p=0 at finite Monte Carlo resolution.
    count_lower = int(np.sum(bootstrap_means <= 0.0))
    count_upper = int(np.sum(bootstrap_means >= 0.0))
    tail_count = min(count_lower, count_upper)
    p_value = min(
        1.0,
        2.0 * float(tail_count + 1) / float(len(bootstrap_means) + 1),
    )
    p_value_is_upper_bound = tail_count == 0

    return SeedMeanDifferenceBootstrapResult(
        mean_difference=float(np.mean(differences)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        p_value_is_upper_bound=bool(p_value_is_upper_bound),
        n_bootstrap=int(len(bootstrap_means)),
        n_seeds=int(differences.size),
        ci=float(ci),
    )
