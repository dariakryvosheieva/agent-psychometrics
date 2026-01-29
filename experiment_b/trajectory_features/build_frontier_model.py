"""Build and evaluate a model for predicting frontier task difficulty.

This script:
1. Loads all extracted trajectory features from multiple agents
2. Computes keyword features from LLM reasoning
3. Finds features significantly correlated with oracle IRT difficulty
4. Builds a Ridge regression model with cross-validation
5. Saves results to a versioned JSON file for reproducibility

Usage:
    python -m experiment_b.trajectory_features.build_frontier_model

    # Save results with a specific version tag
    python -m experiment_b.trajectory_features.build_frontier_model --version v1

    # Use stricter significance threshold
    python -m experiment_b.trajectory_features.build_frontier_model --alpha 0.01

    # Use only agents with full 500-task trajectory coverage
    python -m experiment_b.trajectory_features.build_frontier_model --full-coverage-only
"""

# Agents with full 500/500 trajectory coverage (pre-frontier only)
# These are the only agents that can be used for full production runs
FULL_COVERAGE_AGENTS = {
    # Short name -> Full agent name mapping
    "masai": "20240612_MASAI_gpt4o",
    "sweagent_sonnet": "20240620_sweagent_claude3.5sonnet",
    "amazon": "20240721_amazon-q-developer-agent-20240719-dev",
    "honeycomb": "20240820_honeycomb",
    "lingma72b": "20241002_lingma-agent_lingma-swe-gpt-72b",
    "agentless": "20241028_agentless-1.5_gpt4o",
    "openhands_sonnet": "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "navie": "20241106_navie-2-gpt4o-sonnet",
    "autocoderover": "20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022",
    "marscode": "20241125_marscode-agent-dev",
    "agentless_sonnet": "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
    "epam_sonnet": "20241212_epam-ai-run-claude-3-5-sonnet",
    "openhands": "20250415_openhands",
}

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from experiment_b.swebench.config import SWEBenchConfig
from experiment_b.shared.data_preparation import (
    identify_frontier_tasks_zero_pre,
    split_agents_by_dates,
)
from experiment_b.trajectory_features.keyword_features import (
    KEYWORD_FEATURES,
    extract_keyword_features,
)


def load_frontier_tasks(config: SWEBenchConfig) -> Tuple[List[str], pd.DataFrame]:
    """Load frontier tasks and oracle difficulties."""
    all_agents = config.all_agents
    agent_dates = config.get_agent_dates(all_agents)
    pre_frontier, post_frontier = split_agents_by_dates(
        all_agents, agent_dates, config.cutoff_date
    )
    frontier_tasks = identify_frontier_tasks_zero_pre(
        config.responses_path, pre_frontier, post_frontier
    )

    oracle_items = pd.read_csv(
        "clean_data/swebench_verified_20251120_full/1d/items.csv", index_col=0
    )
    oracle_frontier = oracle_items.loc[oracle_items.index.intersection(frontier_tasks)]

    return frontier_tasks, oracle_frontier


def load_agent_features(
    features_dir: Path,
    frontier_tasks: List[str],
    allowed_agents: set = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and combine features from all agents.

    Args:
        features_dir: Directory containing frontier_v2_* subdirectories
        frontier_tasks: List of frontier task IDs to filter to
        allowed_agents: If provided, only load features from these agents (short names)

    Returns:
        Tuple of (combined feature DataFrame, list of agent short names used)
    """
    agent_dirs = list(features_dir.glob("frontier_v2_*"))

    all_features = []
    agent_names = []

    for agent_dir in sorted(agent_dirs):
        csv_path = agent_dir / "llm_judge_features.csv"
        if not csv_path.exists():
            continue

        short_name = agent_dir.name.replace("frontier_v2_", "")

        # Filter to allowed agents if specified
        if allowed_agents is not None and short_name not in allowed_agents:
            continue

        agent_names.append(short_name)

        df = pd.read_csv(csv_path)
        if "_task_id" in df.columns:
            df = df.rename(columns={"_task_id": "task_id"})
        df = df.set_index("task_id")

        # Extract base features
        agent_features = {}
        for base_col in ["solution_stability", "progress_linearity", "conceptual_pivot_count"]:
            if base_col in df.columns:
                agent_features[f"{short_name}_{base_col}"] = df[base_col]

        # Extract keyword features from reasoning
        if "reasoning" in df.columns:
            for task_id in df.index:
                reasoning = df.loc[task_id, "reasoning"]
                kw_features = extract_keyword_features(reasoning, short_name)
                for feat_name, feat_val in kw_features.items():
                    if feat_name not in agent_features:
                        agent_features[feat_name] = pd.Series(dtype=int)
                    agent_features[feat_name].loc[task_id] = feat_val

        if agent_features:
            all_features.append(pd.DataFrame(agent_features))

    # Combine all features
    if not all_features:
        raise ValueError("No agent features found")

    combined_df = pd.concat(all_features, axis=1)
    combined_df = combined_df.loc[combined_df.index.intersection(frontier_tasks)]

    return combined_df, agent_names


def find_significant_features(
    combined_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    alpha: float = 0.05,
) -> List[Dict]:
    """Find features significantly correlated with oracle difficulty."""
    merged = combined_df.join(oracle_df[["b"]], how="inner")

    significant = []
    for col in merged.columns:
        if col == "b":
            continue

        col_data = merged[col].fillna(0)

        # Check for sufficient variance
        if "kw_" in col:
            # Keyword feature: need enough positive cases
            if col_data.sum() < 3 or col_data.sum() > len(merged) - 3:
                continue
        else:
            # Base feature: need sufficient std
            if col_data.std() < 0.3:
                continue

        r, p = stats.pearsonr(col_data, merged["b"])
        if p < alpha:
            significant.append({
                "feature": col,
                "r": float(r),
                "p": float(p),
                "n": int((~col_data.isna()).sum()),
            })

    return sorted(significant, key=lambda x: x["p"])


def build_model(
    combined_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict:
    """Build Ridge regression model and evaluate."""
    merged = combined_df.join(oracle_df[["b"]], how="inner")

    X = merged[feature_cols].fillna(0).values
    y = merged["b"].values

    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred

    # High residual tasks
    high_resid_idx = np.argsort(np.abs(residuals))[::-1][:5]
    high_residual_tasks = [
        {
            "task_id": merged.index[i],
            "actual": float(y[i]),
            "predicted": float(y_pred[i]),
            "residual": float(residuals[i]),
        }
        for i in high_resid_idx
    ]

    return {
        "n_features": len(feature_cols),
        "n_samples": len(y),
        "best_alpha": float(model.alpha_),
        "full_data_r2": float(model.score(X, y)),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "high_residual_tasks": high_residual_tasks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build frontier task difficulty prediction model"
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("chris_output/trajectory_features"),
        help="Directory containing extracted features",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: features_dir/model_results_{version}.json)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version tag for output file",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for feature selection",
    )
    parser.add_argument(
        "--full-coverage-only",
        action="store_true",
        help="Only use agents with full 500-task trajectory coverage",
    )
    args = parser.parse_args()

    # Load data
    config = SWEBenchConfig()
    print("Loading frontier tasks...")
    frontier_tasks, oracle_df = load_frontier_tasks(config)
    print(f"  Frontier tasks: {len(frontier_tasks)}")

    # Determine which agents to use
    allowed_agents = None
    if args.full_coverage_only:
        allowed_agents = set(FULL_COVERAGE_AGENTS.keys())
        print(f"\nFiltering to full-coverage agents only ({len(allowed_agents)} agents)...")

    print("\nLoading agent features...")
    combined_df, agent_names = load_agent_features(
        args.features_dir, frontier_tasks, allowed_agents
    )
    print(f"  Agents: {len(agent_names)}")
    print(f"  Total features: {len(combined_df.columns)}")

    print(f"\nFinding significant features (p < {args.alpha})...")
    significant = find_significant_features(combined_df, oracle_df, args.alpha)
    print(f"  Significant features: {len(significant)}")

    for feat in significant:
        sig = "***" if feat["p"] < 0.001 else "**" if feat["p"] < 0.01 else "*"
        print(f"    {feat['feature']}: r={feat['r']:+.3f}, p={feat['p']:.4f} {sig}")

    if not significant:
        print("\nNo significant features found. Try extracting more agents.")
        return

    # Build model
    feature_cols = [f["feature"] for f in significant]
    print("\nBuilding model...")
    model_results = build_model(combined_df, oracle_df, feature_cols)

    print(f"\nModel Performance:")
    print(f"  Features: {model_results['n_features']}")
    print(f"  Samples: {model_results['n_samples']}")
    print(f"  Best alpha: {model_results['best_alpha']}")
    print(f"  Full data R²: {model_results['full_data_r2']:.3f}")
    print(f"  CV R² (5-fold): {model_results['cv_r2_mean']:.3f} (±{model_results['cv_r2_std']:.3f})")

    # Prepare output
    version = args.version or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or args.features_dir / f"model_results_{version}.json"

    results = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "config": {
            "significance_alpha": args.alpha,
            "features_dir": str(args.features_dir),
            "n_agents": len(agent_names),
            "agents": agent_names,
        },
        "keyword_patterns": KEYWORD_FEATURES,
        "model_performance": model_results,
        "significant_features": significant,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Also update the main tracking file
    tracking_path = args.features_dir / "significant_features_by_agent.json"
    with open(tracking_path, "w") as f:
        json.dump({
            "model_performance": {
                "n_features": model_results["n_features"],
                "n_samples": model_results["n_samples"],
                "best_alpha": model_results["best_alpha"],
                "full_data_r2": round(model_results["full_data_r2"], 3),
                "cv_r2": round(model_results["cv_r2_mean"], 3),
            },
            "features": feature_cols,
            "feature_details": [
                {"col_name": f["feature"], "r": round(f["r"], 3), "p": round(f["p"], 4)}
                for f in significant
            ],
        }, f, indent=2)
    print(f"Updated tracking file: {tracking_path}")


if __name__ == "__main__":
    main()
