"""Plot leave-one-out feature source ablation bar graph.

Each dataset gets 4 bars: the full LLM-judge feature set, then one bar per
agentic feature source removed (Repo State, Tests, Solution). Each
leave-one-out bar is colored by the source that was removed, matching the
colors in plot_information_ablation.py. The per-dataset constant baseline is
drawn as a dashed line across each group.

The problem statement is deliberately not left out: the other sources rely on
it for extraction context, so its information cannot be removed by dropping
PROBLEM-level features (see the paper's New Tasks ablation discussion).

Rows are read from the ablation results CSV (see run_information_ablation.py
--output).

Usage:
    python -m experiment_new_tasks.plot_loo_ablation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_CSV = Path("output/information_ablation/results_with_std.csv")

# Display order matches plot_information_ablation.py.
DATASETS = [
    ("SWE-bench Verified", "SWE-bench\nVerified"),
    ("SWE-bench Pro", "SWE-bench\nPro"),
    ("GSO", "GSO"),
    ("Terminal-Bench 2.0", "Terminal-Bench\n2.0"),
]

# (results-CSV row name, legend label, bar color)
COLOR_FULL = "#4a4a4a"
COLOR_REPO = "#e8833a"
COLOR_TESTS = "#59a14f"
COLOR_SOLUTION = "#b07aa1"

BARS = [
    ("+ Solution (Full)", "Full (all sources)", COLOR_FULL),
    ("Full Minus Auditor", "− Repo State", COLOR_REPO),
    ("Problem + Auditor + Solution (No Test)", "− Tests", COLOR_TESTS),
    ("+ Test", "− Solution", COLOR_SOLUTION),
]
BASELINE_ROW = "Baseline"

BAR_WIDTH = 0.15
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 14
TITLE_SIZE = 16

OUT = Path("output/loo_ablation_barplot.png")


def load_results() -> Dict[str, Dict[str, Tuple[float, Optional[float]]]]:
    """Return {row_name: {dataset_display: (auc, sd_or_None)}}."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")
    results: Dict[str, Dict[str, Tuple[float, Optional[float]]]] = {}
    df = pd.read_csv(RESULTS_CSV)
    for _, row in df.iterrows():
        name = str(row["Info Level"])
        per_dataset: Dict[str, Tuple[float, Optional[float]]] = {}
        for display, _ in DATASETS:
            auc = float(row[f"{display} AUC"])
            sd_raw = row.get(f"{display} SD")
            sd = None if pd.isna(sd_raw) else float(sd_raw)
            per_dataset[display] = (auc, sd)
        results[name] = per_dataset
    return results


def main() -> None:
    results = load_results()

    needed = [row_name for row_name, _, _ in BARS] + [BASELINE_ROW]
    missing = [name for name in needed if name not in results]
    if missing:
        raise ValueError(
            f"Missing rows {missing} in {RESULTS_CSV}. "
            "Run experiment_new_tasks.run_information_ablation with --output "
            "pointing at that CSV."
        )

    x = np.arange(len(DATASETS))
    offsets = (np.arange(len(BARS)) - (len(BARS) - 1) / 2) * BAR_WIDTH

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for k, (row_name, _, color) in enumerate(BARS):
        aucs = [results[row_name][display][0] for display, _ in DATASETS]
        sds = [results[row_name][display][1] for display, _ in DATASETS]
        yerr = [0.0 if sd is None else sd for sd in sds]
        ax.bar(
            x + offsets[k], aucs, BAR_WIDTH,
            yerr=yerr, capsize=2, error_kw={"linewidth": 1.0},
            color=color, edgecolor="white", linewidth=0.5,
        )

    # Per-dataset baseline as a dashed line across each group
    half_span = (len(BARS) / 2 + 0.3) * BAR_WIDTH
    for j, (display, _) in enumerate(DATASETS):
        base_auc = results[BASELINE_ROW][display][0]
        ax.plot(
            [x[j] - half_span, x[j] + half_span], [base_auc, base_auc],
            linestyle="--", color="#9e9e9e", linewidth=1.4, zorder=1,
        )

    ax.set_title("New Tasks (feature source ablation)", fontsize=TITLE_SIZE)
    ax.set_ylabel("AUC-ROC", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0.6, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS], fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="white", label=label)
        for _, label, color in BARS
    ]
    legend_handles.append(
        mlines.Line2D([], [], linestyle="--", color="#9e9e9e", label="Baseline")
    )
    ax.legend(
        handles=legend_handles,
        fontsize=LEGEND_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15, dpi=200)
    print(OUT)


if __name__ == "__main__":
    main()
