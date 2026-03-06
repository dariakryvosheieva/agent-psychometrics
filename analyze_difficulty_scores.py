from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Scores = np.ndarray
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


@dataclass(frozen=True)
class BenchmarkScores:
    name: str
    path: Path
    scores: Scores


def load_b_scores(csv_path: Path) -> Scores:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    scores: List[float] = []
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        fns = list(r.fieldnames or [])
        if "b" not in set(fns):
            raise ValueError(f"Missing required column 'b' in {csv_path}; got columns={fns}")
        for row in r:
            s = str(row.get("b", "") or "").strip()
            if not s:
                continue
            try:
                scores.append(float(s))
            except ValueError:
                continue
    if not scores:
        raise ValueError(f"No valid difficulty scores found in {csv_path} (column 'b').")
    return np.asarray(scores, dtype=float)


def mean_and_variance(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.mean(x)), float(np.var(x, ddof=0)))


def compute_bin_edges(all_scores: np.ndarray, bins: str) -> np.ndarray:
    edges = np.histogram_bin_edges(all_scores, bins=bins)
    if edges.size < 2:
        edges = np.histogram_bin_edges(all_scores, bins=30)
    return edges


def plot_overlapping_histograms(
    *,
    benchmarks: Sequence[BenchmarkScores],
    out_path: Path,
    bins: str,
    title: str,
    alpha: float,
) -> None:
    all_scores = np.concatenate([b.scores for b in benchmarks], axis=0)
    edges = compute_bin_edges(all_scores, bins=bins)

    plt.figure(figsize=(10, 6))
    for b in benchmarks:
        weights = None
        plt.hist(
            b.scores,
            bins=edges,
            alpha=alpha,
            label=f"{b.name} (n={b.scores.size})",
            edgecolor="none",
            weights=weights,
        )
        mu, _ = mean_and_variance(b.scores)
        plt.axvline(mu, linewidth=1.5, linestyle="--")

    plt.title(title)
    plt.xlabel("IRT difficulty score (b)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--items_verified_csv",
        type=str,
        default="data/all_benchmarks/1d_1pl/items_verified.csv",
        help="Path to SWE-bench Verified items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--items_pro_csv",
        type=str,
        default="data/all_benchmarks/1d_1pl/items_pro.csv",
        help="Path to SWE-bench Pro items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--items_terminal_bench_csv",
        type=str,
        default="data/all_benchmarks/1d_1pl/items_terminal_bench.csv",
        help="Path to Terminal-Bench items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--items_gso_csv",
        type=str,
        default="data/all_benchmarks/1d_1pl/items_gso.csv",
        help="Path to GSO items CSV (must contain column 'b').",
    )
    p.add_argument(
        "--out_plot",
        type=str,
        default="data/difficulty_score_histograms.png",
        help="Where to write the histogram PNG.",
    )
    p.add_argument("--alpha", type=float, default=0.45, help="Histogram transparency.")
    p.add_argument(
        "--title",
        type=str,
        default="IRT difficulty score distributions",
        help="Plot title.",
    )
    args = p.parse_args(argv)

    verified_path = resolve_path(args.items_verified_csv)
    pro_path = resolve_path(args.items_pro_csv)
    tb_path = resolve_path(args.items_terminal_bench_csv)
    gso_path = resolve_path(args.items_gso_csv)
    out_plot_path = resolve_path(args.out_plot)

    benchmarks = [
        BenchmarkScores("SWE-bench Verified", verified_path, load_b_scores(verified_path)),
        BenchmarkScores("SWE-bench Pro", pro_path, load_b_scores(pro_path)),
        BenchmarkScores("Terminal-Bench", tb_path, load_b_scores(tb_path)),
        BenchmarkScores("GSO", gso_path, load_b_scores(gso_path)),
    ]

    print("Benchmark difficulty statistics (b):")
    for b in benchmarks:
        mu, var = mean_and_variance(b.scores)
        print(f"- {b.name}: n={b.scores.size}, mean={mu:.6f}, var={var:.6f}")

    plot_overlapping_histograms(
        benchmarks=benchmarks,
        out_path=out_plot_path,
        bins="fd",
        title=str(args.title),
        alpha=float(args.alpha),
    )
    print(f"Wrote plot: {out_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

