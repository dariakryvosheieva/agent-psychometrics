"""Shared result extraction and formatting utilities for CV experiments."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiment_new_tasks.bootstrap import holm_bonferroni_adjust


def apply_holm_correction(
    all_results: Dict[str, Dict[str, Any]],
    family_methods: List[str],
) -> None:
    """Add Holm-Bonferroni adjusted p-values for one experiment's test family.

    The family is every (benchmark, method) bootstrap comparison for the
    given methods across all non-errored benchmarks in ``all_results``.
    Mutates each metrics dict in place, adding ``{method}__p_value_holm`` and
    ``{method}__p_value_holm_is_upper_bound``. Callers decide the family by
    what they pass: restrict ``all_results`` / ``family_methods`` to the rows
    reported together (e.g., exclude Oracle, which is a skyline rather than a
    tested claim).
    """
    entries: List[Any] = []
    p_values: List[float] = []
    bounds: List[bool] = []
    for dataset_name, metrics in all_results.items():
        if "error" in metrics:
            continue
        for method in family_methods:
            p_value = metrics.get(f"{method}__p_value")
            if p_value is None:
                raise ValueError(
                    f"Missing bootstrap p-value for method {method!r} on "
                    f"{dataset_name!r}; cannot apply the Holm correction to an "
                    "incomplete family"
                )
            entries.append((metrics, method))
            p_values.append(float(p_value))
            bounds.append(bool(metrics.get(f"{method}__p_value_is_upper_bound")))
    if not entries:
        raise ValueError("No bootstrap p-values found; cannot apply the Holm correction")

    adjusted = holm_bonferroni_adjust(p_values, bounds)
    for (metrics, method), result in zip(entries, adjusted):
        metrics[f"{method}__p_value_holm"] = result.p_value
        metrics[f"{method}__p_value_holm_is_upper_bound"] = result.p_value_is_upper_bound


def format_p_value(p_value: Optional[float], is_upper_bound: Optional[bool]) -> str:
    """Format a p-value, prefixing '<' when it is only an upper bound."""

    if p_value is None:
        return ""
    if is_upper_bound:
        return f"<{p_value:.6g}"
    return f"{p_value:.6g}"


def print_holm_summary(
    all_results: Dict[str, Dict[str, Any]],
    family_methods: List[str],
) -> None:
    """Print raw and Holm-adjusted p-values for the experiment family."""

    print("\nHolm-Bonferroni adjusted p-values "
          "(family = all benchmark x method comparisons above):")
    for dataset_name, metrics in all_results.items():
        if "error" in metrics:
            continue
        for method in family_methods:
            if f"{method}__p_value_holm" not in metrics:
                continue
            raw = format_p_value(
                metrics.get(f"{method}__p_value"),
                metrics.get(f"{method}__p_value_is_upper_bound"),
            )
            adjusted = format_p_value(
                metrics.get(f"{method}__p_value_holm"),
                metrics.get(f"{method}__p_value_holm_is_upper_bound"),
            )
            print(f"  {dataset_name} / {method}: p={raw}, Holm-adjusted p={adjusted}")


def extract_metrics(
    results: Dict[str, Any],
    name_mappings: Dict[str, str],
) -> Dict[str, Any]:
    """Extract display metrics from a repeated-CV result dictionary."""

    if "error" in results:
        return {"error": results["error"]}

    metrics: Dict[str, Optional[float]] = {}
    cv_results = results.get("cv_results", {})
    for internal_name, display_name in name_mappings.items():
        if internal_name not in cv_results:
            continue
        result = cv_results[internal_name]
        if result.get("mean_auc") is not None:
            metrics[display_name] = result["mean_auc"]
            if result.get("std_auc") is not None:
                metrics[f"{display_name}__std"] = result["std_auc"]
            bootstrap = result.get("bootstrap_difference_vs_baseline")
            if bootstrap is not None:
                metrics[f"{display_name}__delta"] = result.get(
                    "mean_difference_vs_baseline"
                )
                metrics[f"{display_name}__ci_low"] = bootstrap.get("ci_low")
                metrics[f"{display_name}__ci_high"] = bootstrap.get("ci_high")
                metrics[f"{display_name}__p_value"] = bootstrap.get("p_value")
                metrics[f"{display_name}__p_value_is_upper_bound"] = bootstrap.get(
                    "p_value_is_upper_bound"
                )
    return metrics


def finite_auc_values(values: List[Optional[float]], *, context: str) -> List[float]:
    """Return fold AUCs, failing if any requested fold has no valid AUC."""

    if any(value is None for value in values):
        missing = [idx for idx, value in enumerate(values) if value is None]
        raise ValueError(f"Missing fold AUCs for {context}: fold indices {missing}")
    return [float(value) for value in values]


def sample_std(values: List[float]) -> float:
    """Sample standard deviation for repeated-seed summaries."""

    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def print_repeated_cv_summary(
    *,
    display_name: str,
    cv_results: Dict[str, Dict[str, Any]],
    n_fold_seeds: int,
    k_folds: int,
) -> None:
    """Print the repeated-CV summary with deltas and bootstrap intervals."""

    print("\n" + "=" * 95)
    print(
        f"SUMMARY: {display_name} "
        f"({n_fold_seeds} SEEDS x {k_folds}-FOLD CROSS-VALIDATION)"
    )
    print("=" * 95)
    print(
        f"\n{'Method':<24} {'AUC Mean':>10} {'AUC SD':>10} "
        f"{'Delta':>10} {'95% CI':>23} {'p-value':>10}"
    )
    print("-" * 95)

    for method_name, result in sorted(
        cv_results.items(),
        key=lambda item: item[1]["mean_auc"],
        reverse=True,
    ):
        bootstrap = result["bootstrap_difference_vs_baseline"]
        p_value = bootstrap["p_value"]
        if bootstrap["p_value_is_upper_bound"]:
            p_value_str = f"<{p_value:.4f}"
        else:
            p_value_str = f"{p_value:.4f}"
        ci_str = f"[{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}]"
        print(
            f"{method_name:<24} {result['mean_auc']:>10.4f} "
            f"{result['std_auc']:>10.4f} "
            f"{result['mean_difference_vs_baseline']:>10.4f} "
            f"{ci_str:>23} {p_value_str:>10}"
        )


def format_results_table(
    all_results: Dict[str, Dict[str, Any]],
    methods: List[str],
) -> str:
    """Format extracted metrics as a markdown table."""

    data_rows = []
    for dataset_name, metrics in all_results.items():
        values = []
        for method in methods:
            if "error" in metrics:
                values.append("ERROR")
            elif method in metrics and metrics[method] is not None:
                std = metrics.get(f"{method}__std")
                if std is not None:
                    values.append(f"{metrics[method]:.4f} +/- {std:.4f}")
                else:
                    values.append(f"{metrics[method]:.4f}")
            else:
                values.append("-")
        data_rows.append((dataset_name, values))

    col_widths = [max(len("Benchmark"), max(len(row[0]) for row in data_rows))]
    for i, method in enumerate(methods):
        method_width = len(method)
        value_width = max(len(row[1][i]) for row in data_rows)
        col_widths.append(max(method_width, value_width))

    def pad(text: str, width: int) -> str:
        return text.ljust(width)

    header = "| " + " | ".join(
        pad(col, col_widths[i]) for i, col in enumerate(["Benchmark"] + methods)
    ) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
    rows = [header, separator]
    for dataset_name, values in data_rows:
        row = "| " + pad(dataset_name, col_widths[0]) + " | " + " | ".join(
            pad(value, col_widths[i + 1]) for i, value in enumerate(values)
        ) + " |"
        rows.append(row)
    return "\n".join(rows)


def save_results_csv(
    all_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    methods: List[str],
) -> None:
    """Save extracted metrics to a CSV file."""

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Benchmark"]
        for method in methods:
            header.extend(
                [
                    method,
                    f"{method} SD",
                    f"{method} Delta vs Baseline",
                    f"{method} Delta 95% CI Low",
                    f"{method} Delta 95% CI High",
                    f"{method} Delta p-value",
                    f"{method} Delta Holm p-value",
                ]
            )
        writer.writerow(header)

        for dataset_name, metrics in all_results.items():
            row = [dataset_name]
            for method in methods:
                if "error" in metrics:
                    row.extend(["ERROR", "", "", "", "", "", ""])
                elif method in metrics and metrics[method] is not None:
                    p_value_str = format_p_value(
                        metrics.get(f"{method}__p_value"),
                        metrics.get(f"{method}__p_value_is_upper_bound"),
                    )
                    holm_p_value_str = format_p_value(
                        metrics.get(f"{method}__p_value_holm"),
                        metrics.get(f"{method}__p_value_holm_is_upper_bound"),
                    )
                    row.extend(
                        [
                            f"{metrics[method]:.4f}",
                            _format_optional_float(metrics.get(f"{method}__std")),
                            _format_optional_float(metrics.get(f"{method}__delta")),
                            _format_optional_float(metrics.get(f"{method}__ci_low")),
                            _format_optional_float(metrics.get(f"{method}__ci_high")),
                            p_value_str,
                            holm_p_value_str,
                        ]
                    )
                else:
                    row.extend(["", "", "", "", "", "", ""])
            writer.writerow(row)


def save_summary_json(all_results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """Write metrics as JSON with numpy scalars converted."""

    serializable_results = {}
    for name, metrics in all_results.items():
        serializable_results[name] = {
            key: float(value)
            if isinstance(value, (np.floating, float)) and value is not None
            else value
            for key, value in metrics.items()
        }
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)


def _format_optional_float(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else ""
