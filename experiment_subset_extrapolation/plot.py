"""Plot MAE vs subset size for the subset extrapolation experiment.

Renders a 2x2 grid (one panel per dataset). Each panel shows mean MAE across
seeds for each method, with shaded standard-deviation bands. The x-axis is the
subset size (fraction of tasks observed); a top axis shows the absolute task
count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_DISPLAY = {
    "empirical": "Empirical-subset (baseline)",
    "llm_judge": "LLM-Judge (Ridge)",
    "combined": "Combined (Embedding + LLM-Judge)",
    "oracle": "Oracle (full IRT)",
}

METHOD_COLORS = {
    "empirical": "#9e9e9e",
    "llm_judge": "#4a90d9",
    "combined": "#e8833a",
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
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Extract (sizes, means, stds, task_counts) for one method, dropping
    cells with no successful seeds."""
    sizes: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []
    for size_str in sorted(dataset_results.keys(), key=float):
        cell = dataset_results[size_str]
        ms = cell.get("methods", {}).get(method, {})
        if ms.get("mean_mae") is None:
            continue
        sizes.append(float(size_str))
        means.append(float(ms["mean_mae"]))
        stds.append(float(ms.get("std_mae") or 0.0))
        # Reconstruct task count by inverting (mean over per-seed n_observed
        # is roughly size * total_tasks). We don't store this in the summary,
        # so estimate it via the size and the n_total_tasks field in raw_cells
        # if present in the future. For now just leave empty.
        counts.append(0)
    return sizes, means, stds, counts


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

        size_points_set: set = set()
        for method in methods:
            sizes, means, stds, _ = _series_for_method(ds_results, method)
            if not sizes:
                continue
            size_points_set.update(sizes)
            color = METHOD_COLORS.get(method, "#444444")
            ax.plot(sizes, means, marker="o", color=color, lw=1.6,
                    label=METHOD_DISPLAY.get(method, method))
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(sizes, lower, upper, color=color, alpha=0.15)

        title = DATASET_DISPLAY.get(dataset, dataset)
        if n_total is not None:
            title = f"{title}  (N={n_total} tasks)"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Observed fraction")
        ax.set_ylabel("MAE of predicted overall %")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)

        size_points = sorted(size_points_set)
        if size_points:
            ax.set_xticks(size_points)
            ax.set_xticklabels([f"{s:g}" for s in size_points])

        if n_total is not None and size_points:
            twin = ax.twiny()
            twin.set_xlim(ax.get_xlim())
            twin.set_xticks(size_points)
            twin.set_xticklabels([f"{int(round(s * n_total))}" for s in size_points])
            twin.set_xlabel("Observed task count", fontsize=9)
            twin.tick_params(labelsize=8)

    # Hide unused subplots
    for j in range(n_datasets, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    # Single shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
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
                        default=Path("output/experiment_subset_extrapolation/mae_vs_subset_size.png"))
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    plot_mae_vs_subset_size(summary, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
