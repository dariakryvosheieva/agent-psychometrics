#!/usr/bin/env python3
"""Compare all feature extraction versions across datasets.

For each feature version (v2, v3, v5, v6, v7, v8):
1. Augments judge features with auditor/environment features:
   - SWE-bench Verified: +3 auditor pilot features (23 total)
   - GSO: +8 GPT 5.4 auditor features (28 total)
   - TerminalBench: +8 GPT 5.4 auditor features (28 total)
   - SWE-bench Pro: no auditor features (20 total)
2. Runs full experiment to get Ridge coefficients
3. Selects top 15 features per dataset by |coefficient|
4. Runs experiment with top-15 features
5. Reports comparison table

Usage:
    source .venv/bin/activate
    python scripts/compare_feature_versions.py
    python scripts/compare_feature_versions.py --versions v7,v8
    python scripts/compare_feature_versions.py --skip-augment --skip-coeff  # rerun with existing CSVs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── Feature version definitions ──────────────────────────────────────────────

FEATURE_VERSIONS = {
    "v2": {"label": "GPT 5.4 natural", "dir": "v2_full_20features"},
    "v3": {"label": "GPT 5.4 solution", "dir": "v3_solution_level"},
    "v5": {"label": "Sonnet 4.6 solution", "dir": "v5_sonnet_solution"},
    "v6": {"label": "Sonnet 4.6 natural", "dir": "v6_anthropic_natural"},
    "v7": {"label": "Opus 4.6 solution", "dir": "v7_opus_solution"},
    "v8": {"label": "Opus 4.6 natural", "dir": "v8_opus_natural"},
}

ALL_DATASETS = ["swebench_verified", "gso", "swebench_pro", "terminalbench"]

DISPLAY_NAMES = {
    "swebench_verified": "SWE-bench Verified",
    "gso": "GSO",
    "swebench_pro": "SWE-bench Pro",
    "terminalbench": "TerminalBench",
}

# ── Auditor feature configuration ───────────────────────────────────────────
# Read from feature_registry.py: 8 environment-level features

ENVIRONMENT_FEATURES = [
    "fix_localization", "entry_point_clarity", "change_blast_radius",
    "environment_setup_complexity", "implementation_language_complexity",
    "testing_infrastructure_quality", "dependency_complexity", "codebase_scale",
]

# Per-dataset auditor paths and which columns to use
AUDITOR_CONFIG = {
    "swebench_verified": {
        "path": ROOT / "chris_output" / "auditor_pilot" / "v3_features_top3.csv",
        "columns": ["entry_point_clarity", "change_blast_radius", "fix_localization"],
    },
    "gso": {
        "path": ROOT / "chris_output" / "auditor_features" / "gso_v4_gpt54" / "auditor_features.csv",
        "columns": ENVIRONMENT_FEATURES,
    },
    "terminalbench": {
        "path": ROOT / "chris_output" / "auditor_features" / "terminalbench_v4_gpt54" / "auditor_features.csv",
        "columns": ENVIRONMENT_FEATURES,
    },
    # swebench_pro: no auditor features yet
}


# ── Helper functions ─────────────────────────────────────────────────────────

def get_judge_csv_path(version: str, dataset: str) -> Path:
    ver_dir = FEATURE_VERSIONS[version]["dir"]
    return ROOT / "chris_output" / "llm_judge_features" / ver_dir / dataset / "llm_judge_features.csv"


def get_output_dir(version: str) -> Path:
    return ROOT / "chris_output" / "llm_judge_features" / "comparison" / version


def normalize_instance_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has an 'instance_id' column."""
    if "instance_id" in df.columns:
        return df
    if "_task_id" in df.columns:
        df = df.copy()
        df["instance_id"] = df["_task_id"]
        return df
    raise KeyError("No 'instance_id' or '_task_id' column found")


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get numeric feature columns (same logic as CSVFeatureSource)."""
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("_")
        and c != "reasoning"
    ]


def load_irt_items(dataset: str) -> pd.DataFrame:
    """Load IRT item difficulties."""
    irt_path = ROOT / "data" / dataset / "irt" / "1d_1pl" / "items.csv"
    df = pd.read_csv(irt_path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "instance_id"})
    elif "item_id" in df.columns:
        df = df.rename(columns={"item_id": "instance_id"})
    return df[["instance_id", "b"]]


# ── Step 1: Build augmented CSVs ────────────────────────────────────────────

def build_augmented_csv(version: str, dataset: str) -> Optional[Path]:
    """Merge judge features + auditor features into one CSV."""
    judge_path = get_judge_csv_path(version, dataset)
    if not judge_path.exists():
        return None

    judge_df = normalize_instance_id(pd.read_csv(judge_path))

    if dataset in AUDITOR_CONFIG:
        cfg = AUDITOR_CONFIG[dataset]
        auditor_cols = cfg["columns"]

        # Skip merge if auditor columns already present in judge CSV
        missing = [c for c in auditor_cols if c not in judge_df.columns]

        if missing:
            auditor_path = cfg["path"]
            if not auditor_path.exists():
                raise FileNotFoundError(f"Auditor features not found: {auditor_path}")

            auditor_df = pd.read_csv(auditor_path)
            auditor_df = auditor_df[["instance_id"] + missing]

            nan_count = auditor_df[missing].isna().sum().sum()
            if nan_count > 0:
                raise ValueError(f"Auditor features have {nan_count} NaN values for {dataset}")

            merged = judge_df.merge(auditor_df, on="instance_id", how="left")
            n_missing = merged[missing[0]].isna().sum()
            if n_missing > 0:
                raise ValueError(f"{n_missing}/{len(merged)} tasks missing auditor features")
            result_df = merged
        else:
            result_df = judge_df
    else:
        result_df = judge_df

    out_dir = get_output_dir(version) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "augmented.csv"
    result_df.to_csv(out_path, index=False)
    return out_path


# ── Step 2: Select top 15 features ──────────────────────────────────────────

def select_top_features(
    features_df: pd.DataFrame,
    irt_df: pd.DataFrame,
    n: int = 15,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """Select top N features by Ridge coefficient magnitude on ALL data."""
    merged = features_df.merge(irt_df, on="instance_id")
    numeric_cols = get_numeric_cols(features_df)

    X = merged[numeric_cols].values
    y = merged["b"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    coef_abs = np.abs(ridge.coef_)
    ranked_indices = np.argsort(coef_abs)[::-1]
    ranked = [(numeric_cols[i], float(ridge.coef_[i])) for i in ranked_indices]
    top_n = [numeric_cols[i] for i in ranked_indices[:n]]
    return top_n, ranked


def build_top15_csv(version: str, dataset: str) -> Optional[Path]:
    """Build top-15 feature CSV from augmented CSV."""
    aug_path = get_output_dir(version) / dataset / "augmented.csv"
    if not aug_path.exists():
        return None

    features_df = pd.read_csv(aug_path)
    irt_df = load_irt_items(dataset)

    top15, ranked = select_top_features(features_df, irt_df, n=15)

    out_dir = get_output_dir(version) / dataset
    out_path = out_dir / "top15.csv"
    features_df[["instance_id"] + top15].to_csv(out_path, index=False)

    # Save coefficients for reference
    coeff_path = out_dir / "coefficients.json"
    with open(coeff_path, "w") as f:
        json.dump({"ranked": ranked, "selected_top15": top15}, f, indent=2)

    return out_path


# ── Step 3: Run experiments ──────────────────────────────────────────────────

def run_all_datasets_with_path(
    features_template: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Run experiment on all datasets with given features path template."""
    from experiment_new_tasks.config import ExperimentAConfig
    from experiment_new_tasks.pipeline import cross_validate_all_predictors
    from experiment_new_tasks.run_all_datasets import extract_metrics

    results = {}
    for dataset in ALL_DATASETS:
        expanded = features_template.replace("{dataset}", dataset)
        if not Path(expanded).exists():
            results[dataset] = None
            continue

        try:
            config = ExperimentAConfig.for_dataset(
                dataset, llm_judge_features_path=Path(expanded)
            )
            raw = cross_validate_all_predictors(config, ROOT, k=5)
            metrics = extract_metrics(raw)
            results[dataset] = metrics
        except Exception as e:
            print(f"  ERROR {dataset}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset] = None

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare feature versions")
    parser.add_argument(
        "--versions", type=str, default=None,
        help="Comma-separated versions to run (default: all available)",
    )
    parser.add_argument(
        "--skip-augment", action="store_true",
        help="Skip augmentation step (use existing augmented CSVs)",
    )
    parser.add_argument(
        "--skip-coeff", action="store_true",
        help="Skip coefficient selection (use existing top15 CSVs)",
    )
    args = parser.parse_args()

    # Auto-detect available versions
    if args.versions:
        versions = [v.strip() for v in args.versions.split(",")]
    else:
        versions = []
        for v in FEATURE_VERSIONS:
            for ds in ALL_DATASETS:
                if get_judge_csv_path(v, ds).exists():
                    versions.append(v)
                    break
    print(f"Versions: {versions}")

    comparison_dir = ROOT / "chris_output" / "llm_judge_features" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build augmented CSVs ─────────────────────────────────────
    if not args.skip_augment:
        print("\n" + "=" * 80)
        print("STEP 1: Building augmented CSVs (attach auditor features)")
        print("=" * 80)
        for version in versions:
            print(f"\n  {version} ({FEATURE_VERSIONS[version]['label']}):")
            for dataset in ALL_DATASETS:
                path = build_augmented_csv(version, dataset)
                if path:
                    df = pd.read_csv(path)
                    n = len(get_numeric_cols(df))
                    print(f"    {dataset}: {n} features, {len(df)} tasks")
                else:
                    print(f"    {dataset}: SKIP (source not found)")

    # ── Step 2: Select top 15 features ───────────────────────────────────
    if not args.skip_coeff:
        print("\n" + "=" * 80)
        print("STEP 2: Selecting top 15 features by Ridge |coefficient|")
        print("=" * 80)
        for version in versions:
            print(f"\n  {version} ({FEATURE_VERSIONS[version]['label']}):")
            for dataset in ALL_DATASETS:
                path = build_top15_csv(version, dataset)
                if path:
                    # Load coefficients to show which auditor features were selected
                    coeff_path = get_output_dir(version) / dataset / "coefficients.json"
                    with open(coeff_path) as f:
                        data = json.load(f)
                    auditor_in_top15 = [
                        f for f in data["selected_top15"]
                        if f in ENVIRONMENT_FEATURES
                    ]
                    suffix = f" (auditor: {len(auditor_in_top15)})" if auditor_in_top15 else ""
                    print(f"    {dataset}: top15 saved{suffix}")
                else:
                    print(f"    {dataset}: SKIP")

    # ── Step 3: Run experiments with top-15 features ─────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: Running experiments with top-15 features")
    print("=" * 80)

    all_results: Dict[str, Dict] = {}
    for version in versions:
        label = FEATURE_VERSIONS[version]["label"]
        template = str(get_output_dir(version) / "{dataset}" / "top15.csv")

        # Check availability
        available = [ds for ds in ALL_DATASETS
                     if Path(template.replace("{dataset}", ds)).exists()]
        if not available:
            print(f"\n  {version} ({label}): SKIP (no top15 CSVs found)")
            continue

        print(f"\n  {version} ({label}): running {len(available)} datasets...")
        results = run_all_datasets_with_path(template)
        all_results[version] = results

    # ── Step 4: Print comparison tables ──────────────────────────────────
    print("\n" + "=" * 80)
    print("RESULTS: Top-15 Features by Version")
    print("=" * 80)

    for dataset in ALL_DATASETS:
        display = DISPLAY_NAMES[dataset]
        print(f"\n### {display}")
        header = f"{'Version':<28s} {'Grouped':>8s} {'LLM Judge':>10s} {'Embedding':>10s} {'Baseline':>9s}"
        print(header)
        print("-" * len(header))
        for version in versions:
            if version not in all_results:
                continue
            m = all_results[version].get(dataset)
            if m is None:
                continue
            label = f"{version} ({FEATURE_VERSIONS[version]['label']})"
            g = f"{m['Grouped']:>8.4f}" if m.get("Grouped") else f"{'N/A':>8s}"
            l = f"{m['LLM Judge']:>10.4f}" if m.get("LLM Judge") else f"{'N/A':>10s}"
            e = f"{m['Embedding']:>10.4f}" if m.get("Embedding") else f"{'N/A':>10s}"
            b = f"{m['Baseline']:>9.4f}" if m.get("Baseline") else f"{'N/A':>9s}"
            print(f"{label:<28s} {g} {l} {e} {b}")

    # Save results
    results_path = comparison_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
