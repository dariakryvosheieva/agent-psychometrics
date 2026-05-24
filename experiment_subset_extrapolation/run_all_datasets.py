#!/usr/bin/env python3
"""CLI entry point for the subset extrapolation experiment.

Sweeps (dataset, count, seed) cells in parallel via ProcessPoolExecutor.
`count` is the absolute number of observed target-benchmark tasks; defaults
sweep 2..(~20% of benchmark) stepping by 2.

Usage:
    # Full sweep with defaults (all 4 datasets, per-dataset counts, 20 seeds)
    python -m experiment_subset_extrapolation.run_all_datasets

    # Sanity test on a single dataset, low counts, few seeds
    python -m experiment_subset_extrapolation.run_all_datasets \
        --datasets swebench_verified \
        --subset_counts 2 6 20 \
        --n_seeds 3

    # On engaging (96 CPUs available, leave a few for OS)
    python -m experiment_subset_extrapolation.run_all_datasets --max_workers 64 --plot
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from experiment_subset_extrapolation.config import (
    DEFAULT_DATASETS,
    DEFAULT_METHODS,
    DEFAULT_SUBSET_COUNTS_BY_DATASET,
    SUPPORTED_METHODS,
    SubsetExtrapolationConfig,
)
from experiment_subset_extrapolation.pipeline import (
    aggregate_sweep_results,
    run_one_cell,
    save_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _worker_init() -> None:
    """Cap BLAS/MKL/OMP threads per worker so they don't oversubscribe."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _enumerate_cells(
    sweep_cfg: SubsetExtrapolationConfig,
    n_seeds_to_attempt: int,
) -> List[Tuple[str, int, int]]:
    cells: List[Tuple[str, int, int]] = []
    for dataset in sweep_cfg.datasets:
        for count in sweep_cfg.counts_for(dataset):
            for seed_offset in range(n_seeds_to_attempt):
                cells.append((dataset, int(count), sweep_cfg.seed_start + seed_offset))
    return cells


def _run_sweep(
    sweep_cfg: SubsetExtrapolationConfig,
    n_seeds_to_attempt: int,
    max_workers: int,
    sequential: bool,
) -> List[dict]:
    cells = _enumerate_cells(sweep_cfg, n_seeds_to_attempt)
    total_counts = sum(len(sweep_cfg.counts_for(d)) for d in sweep_cfg.datasets)
    print(
        f"Total cells to run: {len(cells)} "
        f"({len(sweep_cfg.datasets)} datasets × {total_counts} total counts × "
        f"{n_seeds_to_attempt} seeds)"
    )

    raw_results: List[dict] = []
    t0 = time.time()

    if sequential or max_workers <= 1:
        for i, (dataset, count, seed) in enumerate(cells, 1):
            print(f"[{i}/{len(cells)}] {dataset} count={count} seed={seed}", flush=True)
            res = run_one_cell(dataset, sweep_cfg, count, seed, ROOT)
            raw_results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as pool:
            futures = {
                pool.submit(run_one_cell, dataset, sweep_cfg, count, seed, ROOT):
                    (dataset, count, seed)
                for (dataset, count, seed) in cells
            }
            done = 0
            for fut in as_completed(futures):
                dataset, count, seed = futures[fut]
                try:
                    res = fut.result()
                    raw_results.append(res)
                    status = "ok"
                    if res.get("irt_status") == "failed":
                        status = "IRT FAILED"
                except Exception as e:
                    res = {
                        "dataset": dataset, "count": int(count), "seed": int(seed),
                        "fatal_error": f"{type(e).__name__}: {e}",
                        "methods": {},
                    }
                    raw_results.append(res)
                    status = "WORKER CRASHED"
                done += 1
                if done % 10 == 0 or done == len(cells):
                    elapsed = time.time() - t0
                    print(
                        f"  [{done}/{len(cells)}] {dataset} count={count} seed={seed}: "
                        f"{status} ({elapsed:.0f}s elapsed)",
                        flush=True,
                    )

    return raw_results


def _print_summary_table(summary: dict) -> None:
    """Compact terminal-friendly table: per (dataset, count), mean MAE per method."""
    print()
    print("=" * 90)
    print("SUMMARY: mean MAE across seeds (lower is better)")
    print("=" * 90)
    cfg = summary["config"]
    methods = cfg["methods"]
    header = f"{'dataset':<22} {'count':>5} {'n':>3}"
    for m in methods:
        header += f" {m:>20}"
    print(header)
    print("-" * len(header))
    for dataset, by_count in summary["results"].items():
        for count_str, cell in sorted(by_count.items(), key=lambda kv: int(kv[1]["count"])):
            row = f"{dataset:<22} {int(cell['count']):>5} {cell['n_irt_trained']:>3}"
            for m in methods:
                ms = cell["methods"].get(m, {})
                mean = ms.get("mean_mae")
                std = ms.get("std_mae")
                if mean is None:
                    row += f" {'-':>20}"
                else:
                    row += f" {mean:>10.4f}±{std:.4f}"
            print(row)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_root", type=Path,
                        default=Path("output/experiment_subset_extrapolation"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        choices=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--subset_counts", nargs="+", type=int, default=None,
        help=(
            "Optional override: a single list of absolute task counts applied to every "
            "selected dataset. If unset, each dataset uses its per-dataset default sweep "
            "(2..~20%% of benchmark, step 2)."
        ),
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(SUPPORTED_METHODS))
    parser.add_argument("--n_seeds", type=int, default=20,
                        help="Target number of successful seeds per (dataset, count).")
    parser.add_argument("--n_seeds_to_attempt", type=int, default=None,
                        help="Seeds submitted per cell (>= n_seeds; extras cover IRT failures). "
                             "Default: ceil(n_seeds * 1.5).")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4,
                        help="ProcessPoolExecutor worker count. Engaging allows up to 96.")
    parser.add_argument("--sequential", action="store_true",
                        help="Run cells one at a time (debugging only).")
    parser.add_argument("--irt_epochs", type=int, default=5000,
                        help="Pyro SVI epochs per multi-benchmark IRT fit.")
    parser.add_argument("--irt_lr", type=float, default=0.01)
    parser.add_argument("--irt_device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for IRT training. CPU is fine — Pyro 1PL on CPU "
                             "scales well, and parallelism comes from many workers.")
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

    if args.subset_counts is None:
        counts_by_dataset = {
            d: DEFAULT_SUBSET_COUNTS_BY_DATASET[d] for d in args.datasets
        }
    else:
        counts = tuple(int(c) for c in args.subset_counts)
        counts_by_dataset = {d: counts for d in args.datasets}

    sweep_cfg = SubsetExtrapolationConfig(
        output_root=args.output_root,
        subset_counts_by_dataset=counts_by_dataset,
        methods=tuple(args.methods),
        datasets=tuple(args.datasets),
        target_n_seeds=args.n_seeds,
        seed_start=args.seed_start,
        max_seed_attempts_per_cell=n_seeds_to_attempt,
        irt_epochs=args.irt_epochs,
        irt_lr=args.irt_lr,
        irt_device=args.irt_device,
    )

    print(f"Subset extrapolation sweep")
    print(f"  datasets:      {list(sweep_cfg.datasets)}")
    for d in sweep_cfg.datasets:
        print(f"  counts[{d}]: {list(sweep_cfg.counts_for(d))}")
    print(f"  methods:       {list(sweep_cfg.methods)}")
    print(f"  target seeds:  {sweep_cfg.target_n_seeds} (attempting {n_seeds_to_attempt} per cell)")
    print(f"  irt:           epochs={sweep_cfg.irt_epochs} lr={sweep_cfg.irt_lr} device={sweep_cfg.irt_device}")
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
        plot_path = sweep_cfg.output_root / "mae_vs_subset_count.png"
        plot_mae_vs_subset_size(summary, plot_path)
        print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
