#!/usr/bin/env python3
"""Run Experiment A evaluation with v2 Lunette features only.

This script runs the evaluation using ONLY features from the overnight
extraction script (v2), ensuring clean provenance for results.

Usage:
    # Run with v2 features
    python -m experiment_a.run_evaluation_v2

    # Include embeddings for comparison
    python -m experiment_a.run_evaluation_v2 --with_embeddings

    # Dry run to see configuration
    python -m experiment_a.run_evaluation_v2 --dry_run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths
V2_FEATURES_DIR = ROOT / "chris_output" / "experiment_a" / "lunette_features_v2"
V2_FEATURES_CSV = V2_FEATURES_DIR / "lunette_features.csv"
EMBEDDINGS_PATH = ROOT / "out" / "prior_qwen3vl8b" / "embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"


def check_v2_features() -> dict:
    """Check v2 feature coverage and stats."""
    import pandas as pd
    from experiment_a.data_loader import stable_split_tasks

    if not V2_FEATURES_CSV.exists():
        return {"error": f"v2 features not found: {V2_FEATURES_CSV}"}

    df = pd.read_csv(V2_FEATURES_CSV)

    # Get train/test split
    items_path = ROOT / "clean_data" / "swebench_verified_20251115_full" / "1d_1pl" / "items.csv"
    items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(items.index)
    train_tasks_ids, test_tasks_ids = stable_split_tasks(
        all_task_ids, test_fraction=0.2, seed=42
    )

    task_ids = set(df["task_id"].tolist())
    in_train = len([t for t in task_ids if t in train_tasks_ids])
    in_test = len([t for t in task_ids if t in test_tasks_ids])

    # Check version
    v2_count = df["_extraction_version"].value_counts().get("v2", 0) if "_extraction_version" in df.columns else 0

    return {
        "total_features": len(df),
        "train_coverage": f"{in_train}/{len(train_tasks_ids)}",
        "test_coverage": f"{in_test}/{len(test_tasks_ids)}",
        "v2_count": v2_count,
        "feature_columns": [c for c in df.columns if not c.startswith("_") and c != "task_id"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment A evaluation with v2 Lunette features"
    )
    parser.add_argument(
        "--with_embeddings", action="store_true",
        help="Include embedding predictor for comparison"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show configuration without running"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENT A EVALUATION - V2 FEATURES ONLY")
    print("=" * 60)

    # Check v2 features
    print("\nChecking v2 features...")
    stats = check_v2_features()

    if "error" in stats:
        print(f"\nERROR: {stats['error']}")
        print("\nPlease run the overnight extraction script first:")
        print("  python -m experiment_a.overnight_lunette_extraction")
        return

    print(f"  Total features: {stats['total_features']}")
    print(f"  Train coverage: {stats['train_coverage']}")
    print(f"  Test coverage: {stats['test_coverage']}")
    print(f"  v2 extraction count: {stats['v2_count']}")
    print(f"  Feature columns: {len(stats['feature_columns'])}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"\nWould run evaluation with:")
        print(f"  Lunette features: {V2_FEATURES_CSV}")
        if args.with_embeddings:
            print(f"  Embeddings: {EMBEDDINGS_PATH}")
        return

    # Build command
    cmd = [
        sys.executable, "-m", "experiment_a.train_evaluate",
        "--lunette_features_path", str(V2_FEATURES_CSV),
    ]

    if args.with_embeddings and EMBEDDINGS_PATH.exists():
        cmd.extend(["--embeddings_path", str(EMBEDDINGS_PATH)])

    print(f"\nRunning evaluation...")
    print(f"  Command: {' '.join(cmd)}")
    print()

    # Run evaluation
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
