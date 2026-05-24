#!/usr/bin/env python3
"""CLI entry point for the subset extrapolation experiment.

Sweeps (dataset, subset_size, seed) cells in parallel via ProcessPoolExecutor.
Each cell evaluates all requested methods (empirical baseline + IRT-feature
methods). Results are aggregated into a single summary.json and rendered to a
PNG figure of MAE vs subset size per dataset.

Usage:
    # Full sweep with defaults (all 4 datasets, all 7 sizes, 20 seeds)
    python -m experiment_subset_extrapolation.run_all_datasets

    # Fast smoke test on a single dataset
    python -m experiment_subset_extrapolation.run_all_datasets \
        --datasets terminalbench \
        --subset_sizes 0.25 0.50 \
        --n_seeds 3

    # Reduce parallelism to leave cores for other work
    python -m experiment_subset_extrapolation.run_all_datasets --max_workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from experiment_subset_extrapolation.config import (
    DEFAULT_DATASETS,
    DEFAULT_METHODS,
    DEFAULT_SUBSET_SIZES,
    SubsetExtrapolationConfig,
)
from experiment_subset_extrapolation.pipeline import (
    aggregate_sweep_results,
    run_one_cell,
    save_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _worker_init() -> None:
    """Initializer for ProcessPoolExecutor workers.

    Cap BLAS/MKL/OMP thread counts to 1 per worker so they don't oversubscribe
    the CPU when many workers run in parallel (Pyro / numpy / scipy linear
    algebra all default to using every available core).
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _enumerate_cells(
    sweep_cfg: SubsetExtrapolationConfig,
    n_seeds_to_attempt: int,
) -> List[Tuple[str, float, int]]:
    cells: List[Tuple[str, float, int]] = []
    for dataset in sweep_cfg.datasets:
        for size in sweep_cfg.subset_sizes:
            if sweep_cfg.is_excluded(dataset, size):
                continue
            for seed_offset in range(n_seeds_to_attempt):
                cells.append((dataset, float(size), sweep_cfg.seed_start + seed_offset))
    return cells


def _run_sweep(
    sweep_cfg: SubsetExtrapolationConfig,
    n_seeds_to_attempt: int,
    max_workers: int,
    sequential: bool,
) -> List[dict]:
    cells = _enumerate_cells(sweep_cfg, n_seeds_to_attempt)
    print(f"Total cells to run: {len(cells)} "
          f"({len(sweep_cfg.datasets)} datasets × "
          f"{len(sweep_cfg.subset_sizes)} sizes × "
          f"{n_seeds_to_attempt} seeds, excluding {len(sweep_cfg.excluded_cells)} cell(s))")

    raw_results: List[dict] = []
    t0 = time.time()

    if sequential or max_workers <= 1:
        for i, (dataset, size, seed) in enumerate(cells, 1):
            print(f"[{i}/{len(cells)}] {dataset} size={size:.2f} seed={seed}", flush=True)
            res = run_one_cell(dataset, sweep_cfg, size, seed, ROOT)
            raw_results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(run_one_cell, dataset, sweep_cfg, size, seed, ROOT):
                    (dataset, size, seed)
                for (dataset, size, seed) in cells
            }
            done = 0
            for fut in as_completed(futures):
                dataset, size, seed = futures[fut]
                try:
                    res = fut.result()
                    raw_results.append(res)
                    status = "ok"
                    if res.get("irt_status") == "failed":
                        status = "IRT FAILED"
                except Exception as e:
                    res = {
                        "dataset": dataset, "size": float(size), "seed": int(seed),
                        "fatal_error": f"{type(e).__name__}: {e}",
                        "methods": {},
                    }
                    raw_results.append(res)
                    status = "WORKER CRASHED"
                done += 1
                if done % 10 == 0 or done == len(cells):
                    elapsed = time.time() - t0
                    print(
                        f"  [{done}/{len(cells)}] {dataset} size={size:.2f} seed={seed}: "
                        f"{status} ({elapsed:.0f}s elapsed)",
                        flush=True,
                    )

    return raw_results


def _print_summary_table(summary: dict) -> None:
    """Compact terminal-friendly table: per (dataset, size), mean MAE per method."""
    print()
    print("=" * 90)
    print("SUMMARY: mean MAE across seeds (lower is better)")
    print("=" * 90)
    cfg = summary["config"]
    methods = cfg["methods"]
    header = f"{'dataset':<22} {'size':>6} {'n':>3}"
    for m in methods:
        header += f" {m:>14}"
    print(header)
    print("-" * len(header))
    for dataset, by_size in summary["results"].items():
        for size_str, cell in sorted(by_size.items(), key=lambda kv: float(kv[0])):
            row = f"{dataset:<22} {float(size_str):>6.2f} {cell['n_irt_trained']:>3}"
            for m in methods:
                ms = cell["methods"].get(m, {})
                mean = ms.get("mean_mae")
                std = ms.get("std_mae")
                if mean is None:
                    row += f" {'-':>14}"
                else:
                    row += f" {mean:>7.4f}±{std:.4f}"
            print(row)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_root", type=Path,
                        default=Path("output/experiment_subset_extrapolation"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        choices=list(DEFAULT_DATASETS))
    parser.add_argument("--subset_sizes", nargs="+", type=float,
                        default=list(DEFAULT_SUBSET_SIZES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(DEFAULT_METHODS))
    parser.add_argument("--n_seeds", type=int, default=20,
                        help="Target number of successful seeds per (dataset, size).")
    parser.add_argument("--n_seeds_to_attempt", type=int, default=None,
                        help="Seeds submitted per cell (>= n_seeds; extras cover IRT failures). "
                             "Default: ceil(n_seeds * 1.5).")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Worker process count for ProcessPoolExecutor.")
    parser.add_argument("--sequential", action="store_true",
                        help="Run cells sequentially (useful for debugging IRT failures).")
    parser.add_argument("--plot", action="store_true",
                        help="After the sweep, render plot.py output to PNG.")
    parser.add_argument("--no_save_summary", action="store_true",
                        help="Skip writing summary.json (useful for tests).")
    args = parser.parse_args()

    n_seeds_to_attempt = args.n_seeds_to_attempt
    if n_seeds_to_attempt is None:
        n_seeds_to_attempt = max(args.n_seeds, int((args.n_seeds * 3 + 1) // 2))
    if n_seeds_to_attempt < args.n_seeds:
        parser.error("--n_seeds_to_attempt must be >= --n_seeds")

    sweep_cfg = SubsetExtrapolationConfig(
        output_root=args.output_root,
        subset_sizes=tuple(args.subset_sizes),
        methods=tuple(args.methods),
        datasets=tuple(args.datasets),
        target_n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        max_seed_attempts_per_cell=n_seeds_to_attempt,
    )

    print(f"Subset extrapolation sweep")
    print(f"  datasets:      {list(sweep_cfg.datasets)}")
    print(f"  subset_sizes:  {list(sweep_cfg.subset_sizes)}")
    print(f"  methods:       {list(sweep_cfg.methods)}")
    print(f"  target seeds:  {sweep_cfg.target_n_seeds}  "
          f"(attempting {n_seeds_to_attempt} per cell)")
    print(f"  output_root:   {sweep_cfg.output_root}")
    print(f"  workers:       {'sequential' if args.sequential else args.max_workers}")
    print()

    raw_results = _run_sweep(
        sweep_cfg,
        n_seeds_to_attempt=n_seeds_to_attempt,
        max_workers=args.max_workers,
        sequential=args.sequential,
    )

    summary = aggregate_sweep_results(raw_results, sweep_cfg)
    summary["raw_cells"] = raw_results

    if not args.no_save_summary:
        summary_path = sweep_cfg.output_root / "summary.json"
        save_summary(summary, summary_path)
        print(f"\nSummary written to {summary_path}")

    _print_summary_table(summary)

    if args.plot:
        from experiment_subset_extrapolation.plot import plot_mae_vs_subset_size
        plot_path = sweep_cfg.output_root / "mae_vs_subset_size.png"
        plot_mae_vs_subset_size(summary, plot_path)
        print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
