"""PCA ablation study for embedding-based posterior difficulty prediction.

Tests the effect of PCA dimensionality reduction combined with varying
ridge alpha values on the trajectory embedding posterior model.

Usage:
    python -m experiment_b.embeddings.pca_ablation \
        --embeddings_dir chris_output/experiment_b/trajectory_embeddings/full_difficulty \
        --output_dir chris_output/experiment_b/pca_ablation
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.metrics import roc_auc_score

# Add parent to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.data_splits import create_experiment_split
from experiment_b.prior_model import EmbeddingPriorModel
from experiment_b.embeddings.aggregator import AggregationType
from experiment_b.embeddings.posterior_model import EmbeddingPosteriorModel


def load_responses(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load response matrix from JSONL file."""
    responses = {}
    with open(responses_path) as f:
        for line in f:
            row = json.loads(line)
            agent_id = row["subject_id"]
            if "responses" in row:
                responses[agent_id] = row["responses"]
            else:
                task_id = row["item_id"]
                response = row["response"]
                if agent_id not in responses:
                    responses[agent_id] = {}
                responses[agent_id][task_id] = response
    return responses


def compute_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
    agent_ids: List[str],
) -> Dict:
    """Compute AUC using IRT formula P(success) = sigmoid(theta - beta)."""
    y_true = []
    y_scores = []

    for task_id in task_ids:
        if task_id not in predicted_difficulties:
            continue

        beta_pred = predicted_difficulties[task_id]

        for agent_id in agent_ids:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue
            if agent_id not in abilities.index:
                continue

            theta = float(abilities.loc[agent_id, "theta"])
            actual = responses[agent_id][task_id]
            prob = float(expit(theta - beta_pred))

            y_true.append(int(actual))
            y_scores.append(prob)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"error": "Insufficient data", "n_pairs": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_pairs": len(y_true),
    }


def evaluate_predictions(predictions: Dict[str, float], ground_truth: pd.Series) -> Dict:
    """Evaluate prediction quality with Pearson r and MSE."""
    common_tasks = set(predictions.keys()) & set(ground_truth.index)
    if len(common_tasks) < 3:
        return {"error": "Too few common tasks", "n": len(common_tasks)}

    pred_arr = np.array([predictions[t] for t in common_tasks])
    gt_arr = np.array([ground_truth[t] for t in common_tasks])

    r, p = stats.pearsonr(pred_arr, gt_arr)
    mse = np.mean((pred_arr - gt_arr) ** 2)

    return {
        "n": len(common_tasks),
        "pearson_r": float(r),
        "p_value": float(p),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def run_pca_ablation(
    embeddings_dir: Path,
    output_dir: Path,
    pca_components_list: List[Optional[int]],
    ridge_alphas: List[Union[float, str]],
    aggregations: List[AggregationType] = ["mean_only", "mean_std"],
) -> Dict[str, Any]:
    """Run PCA ablation study on embedding posterior.

    Args:
        embeddings_dir: Directory with trajectory embeddings
        output_dir: Directory to save results
        pca_components_list: List of PCA component counts to try (None = no PCA)
        ridge_alphas: List of ridge alphas to try ("cv" for cross-validation)
        aggregations: Aggregation strategies to test

    Returns:
        Dict with all results
    """
    print("=" * 60)
    print("PCA ABLATION STUDY FOR EMBEDDING POSTERIOR")
    print("=" * 60)

    # Paths
    items_path = ROOT / "clean_data/swebench_verified_20251120_full/1d/items.csv"
    abilities_path = ROOT / "clean_data/swebench_verified_20251120_full/1d/abilities.csv"
    responses_path = ROOT / "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
    trajectories_dir = ROOT / "trajectory_data/unified_trajs"
    prior_embeddings_path = ROOT / "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    items_df = pd.read_csv(items_path, index_col=0)
    abilities_df = pd.read_csv(abilities_path, index_col=0)
    responses = load_responses(responses_path)
    print(f"   Tasks: {len(items_df)}")
    print(f"   Agents: {len(abilities_df)}")

    # Create splits
    print("\n2. Creating data splits...")
    split = create_experiment_split(
        responses_path=responses_path,
        trajectories_dir=trajectories_dir,
        weak_threshold=0.2,
        strong_min_improvement=0.1,
        m1_fraction=0.4,
        m2_fraction=0.4,
    )
    print(f"   M1 agents: {len(split.m1_agents)}")
    print(f"   M2 agents: {len(split.m2_agents)}")
    print(f"   D_train tasks: {len(split.d_train_tasks)}")
    print(f"   D_valid tasks: {len(split.d_valid_tasks)}")

    # Train prior model (once)
    print("\n3. Training prior model...")
    all_task_ids = list(items_df.index)
    all_difficulties = items_df["b"].values
    prior_model = EmbeddingPriorModel(prior_embeddings_path, alpha=10000.0)
    prior_model.fit(all_task_ids, all_difficulties)

    # Get prior AUC on validation set (baseline)
    prior_valid_preds = prior_model.get_prior_predictions(split.d_valid_tasks)
    prior_valid_auc = compute_auc(
        prior_valid_preds, abilities_df, responses, split.d_valid_tasks, split.m2_agents
    )
    print(f"   Prior AUC (D_valid): {prior_valid_auc.get('auc', 'N/A'):.4f}")

    # Prepare for ablation
    train_difficulties = items_df.loc[split.d_train_tasks, "b"].values
    valid_gt = items_df.loc[split.d_valid_tasks, "b"]

    results = {
        "config": {
            "embeddings_dir": str(embeddings_dir),
            "n_d_train_tasks": len(split.d_train_tasks),
            "n_d_valid_tasks": len(split.d_valid_tasks),
            "n_m1_agents": len(split.m1_agents),
            "n_m2_agents": len(split.m2_agents),
            "prior_auc": prior_valid_auc.get("auc"),
        },
        "ablation_results": [],
    }

    best_auc = prior_valid_auc.get("auc", 0)
    best_config = None

    # Run ablation
    print("\n4. Running PCA ablation...")
    total_runs = len(aggregations) * len(pca_components_list) * len(ridge_alphas)
    run_count = 0

    for aggregation in aggregations:
        for pca_components in pca_components_list:
            for ridge_alpha in ridge_alphas:
                run_count += 1
                pca_label = f"PCA-{pca_components}" if pca_components else "No-PCA"
                alpha_label = f"alpha={ridge_alpha}"
                print(f"\n   [{run_count}/{total_runs}] {aggregation}, {pca_label}, {alpha_label}...")

                try:
                    # Create and train posterior model
                    posterior_model = EmbeddingPosteriorModel(
                        prior_model=prior_model,
                        aggregation=aggregation,
                        alpha=ridge_alpha,
                        abilities=None,
                        pca_components=pca_components,
                    )

                    posterior_model.fit(
                        task_ids=split.d_train_tasks,
                        ground_truth_difficulties=train_difficulties,
                        weak_agents=split.m1_agents,
                        embeddings_dir=embeddings_dir,
                    )

                    # Evaluate on validation set
                    posterior_valid_preds = posterior_model.predict(
                        split.d_valid_tasks, split.m1_agents, embeddings_dir
                    )
                    posterior_valid_auc = compute_auc(
                        posterior_valid_preds, abilities_df, responses,
                        split.d_valid_tasks, split.m2_agents
                    )
                    posterior_valid_eval = evaluate_predictions(posterior_valid_preds, valid_gt)

                    training_stats = posterior_model.get_training_stats()

                    result = {
                        "aggregation": aggregation,
                        "pca_components_requested": pca_components,
                        "pca_components_actual": posterior_model.pca_dim,
                        "ridge_alpha_requested": str(ridge_alpha),
                        "ridge_alpha_selected": posterior_model.best_alpha,
                        "embedding_dim": posterior_model.embedding_dim,
                        "auc": posterior_valid_auc.get("auc"),
                        "pearson_r": posterior_valid_eval.get("pearson_r"),
                        "mse": posterior_valid_eval.get("mse"),
                        "rmse": posterior_valid_eval.get("rmse"),
                        "n_tasks_used": training_stats.get("tasks_with_embeddings"),
                    }

                    auc = result["auc"]
                    if auc is not None:
                        delta = auc - prior_valid_auc.get("auc", 0)
                        result["delta_auc"] = delta
                        print(f"      AUC: {auc:.4f} (ΔAUC: {delta:+.4f}), r: {result.get('pearson_r', 'N/A'):.4f}")

                        if auc > best_auc:
                            best_auc = auc
                            best_config = result.copy()

                    results["ablation_results"].append(result)

                except Exception as e:
                    print(f"      Error: {e}")
                    import traceback
                    traceback.print_exc()
                    results["ablation_results"].append({
                        "aggregation": aggregation,
                        "pca_components_requested": pca_components,
                        "ridge_alpha_requested": str(ridge_alpha),
                        "error": str(e),
                    })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPrior AUC (baseline): {prior_valid_auc.get('auc', 'N/A'):.4f}")

    print(f"\n{'Aggregation':<12} {'PCA Dims':<10} {'Alpha':<12} {'AUC':>8} {'ΔAUC':>10} {'Pearson r':>12}")
    print("-" * 70)

    for r in sorted(results["ablation_results"], key=lambda x: x.get("auc", 0) or 0, reverse=True):
        if "error" in r:
            continue
        agg_str = r["aggregation"]
        pca_str = str(r["pca_components_actual"]) if r.get("pca_components_actual") else "None"
        alpha_val = r.get("ridge_alpha_selected")
        alpha_str = f"{alpha_val:.0e}" if alpha_val and alpha_val >= 1 else str(alpha_val)
        auc_str = f"{r['auc']:.4f}" if r.get("auc") else "N/A"
        delta_str = f"{r.get('delta_auc', 0):+.4f}" if r.get("delta_auc") is not None else "N/A"
        r_str = f"{r['pearson_r']:.4f}" if r.get("pearson_r") else "N/A"
        print(f"{agg_str:<12} {pca_str:<10} {alpha_str:<12} {auc_str:>8} {delta_str:>10} {r_str:>12}")

    if best_config and best_config.get("auc", 0) > prior_valid_auc.get("auc", 0):
        print(f"\nBest config (beats prior):")
        print(f"  Aggregation: {best_config['aggregation']}")
        print(f"  PCA components: {best_config.get('pca_components_actual')}")
        print(f"  Ridge alpha: {best_config.get('ridge_alpha_selected')}")
        print(f"  AUC: {best_config['auc']:.4f}")
        print(f"  ΔAUC vs prior: {best_config.get('delta_auc', 0):+.4f}")
        results["best_config"] = best_config
    else:
        print(f"\nNo configuration beats the prior baseline.")
        results["best_config"] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PCA ablation study for embedding posterior"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="chris_output/experiment_b/trajectory_embeddings/full_difficulty",
        help="Path to trajectory embeddings directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_b/pca_ablation",
        help="Output directory",
    )
    args = parser.parse_args()

    embeddings_dir = ROOT / args.embeddings_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define ablation grid
    # PCA components: None (no PCA), 25, 50, 100
    # Note: With 119 training tasks, max PCA is 118 components
    pca_components_list = [None, 25, 50, 100]

    # Ridge alphas: cross-validation and fixed values
    ridge_alphas = ["cv", 100, 1000, 10000, 100000, 1000000]

    # Aggregations: mean_only (4096 dims) vs mean_std (8192 dims)
    aggregations = ["mean_only", "mean_std"]

    results = run_pca_ablation(
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        pca_components_list=pca_components_list,
        ridge_alphas=ridge_alphas,
        aggregations=aggregations,
    )

    # Save results
    output_path = output_dir / "pca_ablation_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = convert_numpy(results)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
