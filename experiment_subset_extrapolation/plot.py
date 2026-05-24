"""Plot MAE vs subset task count for the subset extrapolation experiment.

Renders a 2x2 grid (one panel per dataset). Each panel shows mean MAE across
seeds for each method, with shaded standard-deviation bands. The x-axis is the
absolute number of observed tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_DISPLAY = {
    "empirical": "Empirical-subset (baseline)",
    "combined_calibrated": "Multi-bench IRT + calibration",
    "oracle": "Oracle (full IRT)",
}

METHOD_COLORS = {
    "empirical": "#9e9e9e",
    "combined_calibrated": "#b05a1d",
    "oracle": "#59a14f",
}

DATASET_DISPLAY = {
    "swebench_verified": "SWE-bench Verified",
    "swebench_pro": "SWE-bench Pro",
    "gso": "GSO",
    "terminalbench": "Terminal-Bench 2.0",
}


def _series_for_method(
    dataset_results: Dict[str, Any], method: str
) -> Tuple[List[int], List[float], List[float]]:
    """Extract (counts, means, stds) for one method, dropping cells with no
    successful seeds."""
    counts: List[int] = []
    means: List[float] = []
    stds: List[float] = []
    for key in sorted(dataset_results.keys(), key=lambda k: int(dataset_results[k]["count"])):
        cell = dataset_results[key]
        ms = cell.get("methods", {}).get(method, {})
        if ms.get("mean_mae") is None:
            continue
        counts.append(int(cell["count"]))
        means.append(float(ms["mean_mae"]))
        stds.append(float(ms.get("std_mae") or 0.0))
    return counts, means, stds


def _approx_total_tasks(raw_cells: Optional[Iterable[Dict[str, Any]]], dataset: str) -> Optional[int]:
    if raw_cells is None:
        return None
    for r in raw_cells:
        if r.get("dataset") == dataset and "n_total_tasks" in r:
            return int(r["n_total_tasks"])
    return None


def plot_mae_vs_subset_size(
    summary: Dict[str, Any],
    out_path: Path,
    *,
    methods: Optional[List[str]] = None,
) -> None:
    """Render the 2x2 MAE-vs-size figure to `out_path` (PNG)."""
    cfg = summary["config"]
    if methods is None:
        methods = cfg["methods"]
    datasets = cfg["datasets"]
    raw_cells = summary.get("raw_cells")

    n_datasets = len(datasets)
    ncols = 2 if n_datasets > 1 else 1
    nrows = (n_datasets + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False
    )

    for i, dataset in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        ds_results = summary["results"].get(dataset, {})
        n_total = _approx_total_tasks(raw_cells, dataset)

        for method in methods:
            counts, means, stds = _series_for_method(ds_results, method)
            if not counts:
                continue
            color = METHOD_COLORS.get(method, "#444444")
            ax.plot(counts, means, marker="o", color=color, lw=1.6,
                    label=METHOD_DISPLAY.get(method, method))
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(counts, lower, upper, color=color, alpha=0.15)

        title = DATASET_DISPLAY.get(dataset, dataset)
        if n_total is not None:
            title = f"{title}  (N={n_total} tasks)"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of observed tasks")
        ax.set_ylabel("MAE of predicted overall %")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)

    # Hide unused subplots
    for j in range(n_datasets, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Single shared legend at the bottom — collect handles from every
    # subplot and dedupe by label, so the figure renders even when the
    # first subplot has no data.
    seen_labels: set = set()
    handles: list = []
    labels: list = []
    for row in axes:
        for ax in row:
            ah, al = ax.get_legend_handles_labels()
            for h, lbl in zip(ah, al):
                if lbl in seen_labels:
                    continue
                seen_labels.add(lbl)
                handles.append(h)
                labels.append(lbl)
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=len(handles), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Render MAE vs subset size plot.")
    parser.add_argument("--summary", type=Path,
                        default=Path("output/experiment_subset_extrapolation/summary.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("output/experiment_subset_extrapolation/mae_vs_subset_count.png"))
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    plot_mae_vs_subset_size(summary, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
