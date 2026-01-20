"""Analyze LLM judge v7 feature correlations with difficulty residual.

Quick script to see if v7 features predict residual difficulty.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.prior_model import PriorModel
from experiment_b.llm_judge.features_v7 import LLM_JUDGE_V7_FEATURE_NAMES

V7_FEATURES_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_v7_features"


def main():
    config = ExperimentConfig()

    # Load IRT difficulties
    items_path = ROOT / config.items_path
    items_df = pd.read_csv(items_path, index_col=0)
    print(f"Loaded {len(items_df)} task difficulties")

    # Train prior model
    print("Training prior model...")
    all_task_ids = list(items_df.index)
    all_difficulties = items_df["b"].values
    prior_model = PriorModel(alpha=config.prior_alpha)
    prior_model.fit(all_task_ids, all_difficulties)

    # Load v7 features by task
    task_features = defaultdict(list)

    for agent_dir in V7_FEATURES_DIR.iterdir():
        if not agent_dir.is_dir():
            continue
        for task_file in agent_dir.glob("*.json"):
            task_id = task_file.stem
            try:
                with open(task_file) as f:
                    data = json.load(f)
                task_features[task_id].append(data)
            except (json.JSONDecodeError, IOError):
                continue

    print(f"Loaded v7 features for {len(task_features)} tasks")

    # Aggregate features per task and correlate with residual
    results = []

    for task_id, agent_features in task_features.items():
        if task_id not in items_df.index:
            continue

        ground_truth_b = items_df.loc[task_id, "b"]
        prior_preds = prior_model.get_prior_predictions([task_id])
        if task_id not in prior_preds:
            continue
        prior_b = prior_preds[task_id]
        residual = ground_truth_b - prior_b

        # Aggregate v7 features across agents (mean)
        row = {
            "task_id": task_id,
            "ground_truth_b": ground_truth_b,
            "prior_b": prior_b,
            "residual": residual,
            "n_agents": len(agent_features),
        }

        for feat_name in LLM_JUDGE_V7_FEATURE_NAMES:
            values = [f.get(feat_name, 3) for f in agent_features]
            row[f"{feat_name}_mean"] = np.mean(values)
            row[f"{feat_name}_std"] = np.std(values) if len(values) > 1 else 0
            row[f"{feat_name}_min"] = np.min(values)
            row[f"{feat_name}_max"] = np.max(values)

        results.append(row)

    df = pd.DataFrame(results)
    print(f"\nAnalyzing {len(df)} tasks with v7 features")
    print(f"Agents per task: mean={df['n_agents'].mean():.1f}, range=[{df['n_agents'].min()}, {df['n_agents'].max()}]")

    # Compute correlations
    print("\n" + "=" * 70)
    print("V7 FEATURE CORRELATIONS WITH DIFFICULTY RESIDUAL")
    print("(residual = ground_truth_b - prior_b, positive = harder than expected)")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in ["task_id", "ground_truth_b", "prior_b", "residual", "n_agents"]]

    correlations = []
    for col in feature_cols:
        if df[col].std() < 1e-10:
            continue
        r, p = stats.pearsonr(df[col], df["residual"])
        correlations.append({
            "feature": col,
            "pearson_r": r,
            "p_value": p,
            "significant": p < 0.05,
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("pearson_r", key=abs, ascending=False)

    print("\nCorrelations (sorted by |r|):")
    print("-" * 70)
    for _, row in corr_df.iterrows():
        sig = "*" if row["significant"] else " "
        print(f"  {row['feature']:35s}  r={row['pearson_r']:+.3f}  p={row['p_value']:.3f} {sig}")

    print("\n* = significant at p < 0.05")

    # Show feature statistics
    print("\n" + "=" * 70)
    print("V7 FEATURE STATISTICS (aggregated across agents per task)")
    print("=" * 70)

    for feat_name in LLM_JUDGE_V7_FEATURE_NAMES:
        mean_col = f"{feat_name}_mean"
        if mean_col in df.columns:
            print(f"\n{feat_name}:")
            print(f"  task mean of agent means: {df[mean_col].mean():.2f}")
            print(f"  task std of agent means: {df[mean_col].std():.2f}")
            print(f"  range: [{df[mean_col].min():.2f}, {df[mean_col].max():.2f}]")


if __name__ == "__main__":
    main()
