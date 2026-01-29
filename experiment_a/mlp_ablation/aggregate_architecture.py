"""Aggregate and display results from architecture sweep.

Usage:
    python -m experiment_a.mlp_ablation.aggregate_architecture
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT / "chris_output/experiment_a/mlp_embedding"


def load_and_merge() -> dict:
    """Load part1, part2, part3, and part4 JSON files and merge them."""
    results = {}

    for part in [1, 2, 3, 4]:
        path = OUTPUT_DIR / f"architecture_sweep_part{part}.json"
        if path.exists():
            with open(path) as f:
                part_results = json.load(f)
            results.update(part_results)
            print(f"Loaded {len(part_results)} results from {path.name}")
        else:
            print(f"Warning: {path.name} not found")

    # Also try loading the non-partitioned file
    full_path = OUTPUT_DIR / "architecture_sweep.json"
    if full_path.exists() and not results:
        with open(full_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {full_path.name}")

    return results


def print_results_table(results: dict):
    """Print formatted results table sorted by test AUC."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'=' * 85}")
    print("ARCHITECTURE SWEEP RESULTS")
    print(f"{'=' * 85}")
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    # Sort by test AUC descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("mean_auc", 0),
        reverse=True
    )

    for name, r in sorted_results:
        test_auc = r.get("mean_auc", 0)
        train_auc = r.get("train_auc")
        display_name = r.get("display_name", name)

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Key comparisons
    print(f"\n{'-' * 85}")
    print("KEY COMPARISONS:")

    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best MLP (exclude baselines)
        mlp_results = [(name, r) for name, r in results.items()
                       if name not in ["oracle", "constant", "ridge"]]
        if mlp_results:
            best_name, best_r = max(mlp_results, key=lambda x: x[1]["mean_auc"])
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best MLP: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    if "oracle" in results:
        print(f"  Oracle upper bound: {results['oracle']['mean_auc']:.4f}")

    # Group by architecture type
    print(f"\n{'-' * 85}")
    print("BY ARCHITECTURE:")

    for arch_type in ["simple", "deep", "swiglu"]:
        arch_results = [(n, r) for n, r in results.items() if n.startswith(arch_type)]
        if arch_results:
            best_name, best_r = max(arch_results, key=lambda x: x[1]["mean_auc"])
            print(f"  Best {arch_type}: {best_r['display_name']}: {best_r['mean_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate architecture sweep results")
    parser.add_argument("--save", action="store_true",
                        help="Save aggregated results to a combined JSON file")
    args = parser.parse_args()

    results = load_and_merge()
    if results:
        print_results_table(results)
        if args.save:
            out_path = OUTPUT_DIR / "architecture_sweep_combined.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved combined results to: {out_path}")


if __name__ == "__main__":
    main()
