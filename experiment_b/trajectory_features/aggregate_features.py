"""Aggregate trajectory features across agents per task.

This module takes raw features (one row per agent-task pair) and aggregates
them to produce per-task features suitable for difficulty prediction.

Usage:
    python -m experiment_b.trajectory_features.aggregate_features \
        --input_path chris_output/trajectory_features/raw_features.csv \
        --output_path chris_output/trajectory_features/aggregated_features.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FEATURE_NAMES


def aggregate_features(
    raw_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    agent_abilities: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Aggregate trajectory features across agents for each task.

    Produces multiple aggregation statistics for each feature:
    - {feat}_mean: Mean across all agents
    - {feat}_std: Standard deviation across agents
    - {feat}_ability_weighted: Mean weighted by agent ability

    Args:
        raw_df: DataFrame with columns: agent, task_id, resolved, and feature columns
        feature_cols: Feature columns to aggregate (default: FEATURE_NAMES)
        agent_abilities: Dict mapping agent name to IRT ability (theta)

    Returns:
        DataFrame with task_id as index and aggregated feature columns
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_NAMES if c in raw_df.columns]

    # Fail loudly if there are any NaN values in feature columns
    for col in feature_cols:
        if col in raw_df.columns:
            nan_count = raw_df[col].isna().sum()
            if nan_count > 0:
                nan_rows = raw_df[raw_df[col].isna()][['task_id', 'agent']].head(5)
                examples = [f"{r['task_id']}/{r['agent']}" for _, r in nan_rows.iterrows()]
                raise ValueError(
                    f"Found {nan_count} NaN values in feature column '{col}'. "
                    f"First examples: {examples}. "
                    f"Fix the raw features before aggregating."
                )

    aggregated_rows = []

    for task_id, task_df in raw_df.groupby("task_id"):
        row = {"task_id": task_id}

        # Count statistics
        row["n_agents"] = len(task_df)
        row["n_pass"] = task_df["resolved"].sum()
        row["n_fail"] = len(task_df) - row["n_pass"]
        row["pass_rate"] = row["n_pass"] / row["n_agents"] if row["n_agents"] > 0 else 0

        # Trajectory length statistics
        if "trajectory_length" in task_df.columns:
            row["trajectory_length_mean"] = task_df["trajectory_length"].mean()
            row["trajectory_length_std"] = task_df["trajectory_length"].std()

        # Aggregate each feature
        for feat in feature_cols:
            if feat not in task_df.columns:
                continue

            values = task_df[feat].dropna()
            if len(values) == 0:
                continue

            # Basic statistics
            row[f"{feat}_mean"] = values.mean()
            row[f"{feat}_std"] = values.std() if len(values) > 1 else 0

            # Ability-weighted mean (if abilities provided)
            if agent_abilities is not None:
                weights = task_df["agent"].map(agent_abilities).fillna(0)
                # Shift weights to be positive (add offset)
                weights = weights - weights.min() + 1
                valid_weights = weights[task_df[feat].notna()]
                # Compute weighted mean if we have values and non-zero weights
                if len(values) > 0 and valid_weights.sum() > 0:
                    row[f"{feat}_ability_weighted"] = (
                        (values * valid_weights.values).sum() / valid_weights.sum()
                    )

        aggregated_rows.append(row)

    result = pd.DataFrame(aggregated_rows)
    if "task_id" in result.columns:
        result = result.set_index("task_id")

    return result


def create_high_dim_features(
    raw_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create high-dimensional feature matrix with one column per agent-feature pair.

    Instead of aggregating, concatenate raw features for all agents.
    This creates a (n_tasks, n_agents * n_features) matrix.

    Args:
        raw_df: DataFrame with columns: agent, task_id, and feature columns
        feature_cols: Feature columns to include (default: FEATURE_NAMES)
        agents: List of agents to include (default: all in data)

    Returns:
        DataFrame with task_id as index and agent_feature columns
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_NAMES if c in raw_df.columns]

    if agents is None:
        agents = sorted(raw_df["agent"].unique())

    # Pivot: one row per task, one column per agent-feature
    rows = []
    for task_id, task_df in raw_df.groupby("task_id"):
        row = {"task_id": task_id}

        for agent in agents:
            agent_data = task_df[task_df["agent"] == agent]

            for feat in feature_cols:
                col_name = f"{agent}__{feat}"
                if len(agent_data) > 0 and feat in agent_data.columns:
                    row[col_name] = agent_data[feat].values[0]
                else:
                    row[col_name] = np.nan

            # Also include resolved status
            col_name = f"{agent}__resolved"
            if len(agent_data) > 0:
                row[col_name] = int(agent_data["resolved"].values[0])
            else:
                row[col_name] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    if "task_id" in result.columns:
        result = result.set_index("task_id")

    return result


def load_agent_abilities(
    abilities_path: str = "clean_data/swebench_verified_20251120_full/1d/abilities.csv",
) -> Dict[str, float]:
    """Load agent IRT abilities from file."""
    df = pd.read_csv(abilities_path, index_col=0)
    return df["theta"].to_dict()


def get_behavioral_columns(result_df: pd.DataFrame) -> List[str]:
    """Get only behavioral feature columns (no outcome leakage).

    Returns columns that don't leak pass/fail information:
    - trajectory_length_mean, trajectory_length_std
    - {feat}_mean, {feat}_std for each behavioral feature

    Excludes:
    - n_agents, n_pass, n_fail, pass_rate (outcome info)
    - {feat}_ability_weighted (not needed for simpler model)
    """
    behavioral_cols = []

    # Trajectory length stats
    if "trajectory_length_mean" in result_df.columns:
        behavioral_cols.append("trajectory_length_mean")
    if "trajectory_length_std" in result_df.columns:
        behavioral_cols.append("trajectory_length_std")

    # Feature mean and std only (no ability_weighted)
    for col in result_df.columns:
        if col.endswith("_mean") and not col.startswith("trajectory"):
            behavioral_cols.append(col)
        elif col.endswith("_std") and not col.startswith("trajectory"):
            behavioral_cols.append(col)

    return behavioral_cols


def main():
    parser = argparse.ArgumentParser(description="Aggregate trajectory features")
    parser.add_argument(
        "--input_path",
        type=str,
        default="chris_output/trajectory_features/raw_features.csv",
        help="Input path for raw features",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="chris_output/trajectory_features/aggregated_features.csv",
        help="Output path for aggregated features",
    )
    parser.add_argument(
        "--high_dim",
        action="store_true",
        help="Create high-dimensional features instead of aggregated",
    )
    parser.add_argument(
        "--behavioral_only",
        action="store_true",
        help="Only output behavioral features (no pass/fail info leakage)",
    )
    parser.add_argument(
        "--abilities_path",
        type=str,
        default="clean_data/swebench_verified_20251120_full/1d/abilities.csv",
        help="Path to agent abilities CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw features from {input_path}...")
    raw_df = pd.read_csv(input_path)
    print(f"  Loaded {len(raw_df)} rows ({raw_df['task_id'].nunique()} tasks, {raw_df['agent'].nunique()} agents)")

    if args.high_dim:
        print("Creating high-dimensional features...")
        result_df = create_high_dim_features(raw_df)
    else:
        print("Loading agent abilities...")
        abilities = load_agent_abilities(args.abilities_path)
        print(f"  Loaded abilities for {len(abilities)} agents")

        print("Aggregating features...")
        result_df = aggregate_features(raw_df, agent_abilities=abilities)

    print(f"  Created {len(result_df)} task rows with {len(result_df.columns)} columns")

    # Filter to behavioral-only columns if requested
    if args.behavioral_only and not args.high_dim:
        behavioral_cols = get_behavioral_columns(result_df)
        print(f"\n  Filtering to {len(behavioral_cols)} behavioral-only columns (no outcome leakage)")
        output_df = result_df[behavioral_cols]
    else:
        output_df = result_df

    output_df.to_csv(output_path)
    print(f"Saved to {output_path}")

    # Print summary statistics
    print("\n=== Aggregated Feature Summary ===")
    feature_cols = [c for c in output_df.columns if "_mean" in c and not c.startswith("trajectory")]
    for col in feature_cols[:10]:
        print(f"  {col}: mean={output_df[col].mean():.2f}, std={output_df[col].std():.2f}")


if __name__ == "__main__":
    main()
