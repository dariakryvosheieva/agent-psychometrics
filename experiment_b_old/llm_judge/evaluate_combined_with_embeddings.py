"""Evaluate combining LLM judge features with embeddings.

This script tests whether LLM judge features add signal on top of embeddings:
1. Embedding only (baseline): 0.7383 AUC
2. Embedding + Prior LLM judge features (9 problem features)
3. Embedding + Combined LLM judge features (13 features = 9 problem + 4 trajectory)

Usage:
    python -m experiment_b.llm_judge.evaluate_combined_with_embeddings
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_b.config import ExperimentConfig
from experiment_b.data_splits import create_experiment_split
from experiment_b.llm_judge.features_prior import (
    LLM_JUDGE_PRIOR_FEATURE_NAMES,
    load_llm_judge_prior_features_batch,
)
from experiment_b.llm_judge.features_combined import (
    LLM_JUDGE_COMBINED_FEATURE_NAMES,
    TRAJECTORY_FEATURE_NAMES,
    load_llm_judge_combined_features_batch,
)


# Directories
PRIOR_FEATURES_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_prior_features"
COMBINED_FEATURES_DIR = ROOT / "chris_output" / "experiment_b" / "llm_judge_combined_features"


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


def load_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from .npz file."""
    data = np.load(embeddings_path, allow_pickle=True)
    task_ids = [str(x) for x in data["task_ids"].tolist()]
    X = data["X"].astype(np.float32)
    return {task_id: X[i] for i, task_id in enumerate(task_ids)}


def compute_auc(
    predicted_difficulties: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    task_ids: List[str],
    agent_ids: List[str],
) -> Dict:
    """Compute AUC for predicted difficulties using IRT formula."""
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


def main():
    print("=" * 70)
    print("COMBINING EMBEDDINGS WITH LLM JUDGE FEATURES")
    print("=" * 70)

    # Load config and create splits
    config = ExperimentConfig()
    split = create_experiment_split(
        responses_path=ROOT / config.responses_path,
        trajectories_dir=ROOT / config.trajectories_dir,
        weak_threshold=config.weak_threshold,
        strong_min_improvement=config.strong_min_improvement,
        m1_fraction=config.m1_fraction,
        m2_fraction=config.m2_fraction,
    )

    print(f"\nData splits:")
    print(f"  D_train tasks: {len(split.d_train_tasks)}")
    print(f"  D_valid tasks: {len(split.d_valid_tasks)}")

    # Load ground truth
    items_path = ROOT / config.items_path
    items_df = pd.read_csv(items_path, index_col=0)
    abilities_path = items_path.parent / "abilities.csv"
    abilities_df = pd.read_csv(abilities_path, index_col=0)
    responses = load_responses(ROOT / config.responses_path)

    # Load embeddings
    # Use Daria's pre-computed embeddings
    embeddings_path = ROOT / "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = load_embeddings(embeddings_path)
    embedding_dim = len(next(iter(embeddings.values())))
    print(f"  Loaded {len(embeddings)} embeddings, dim={embedding_dim}")

    # Load LLM judge features
    print("\nLoading LLM judge features...")
    prior_features = load_llm_judge_prior_features_batch(
        list(set(split.d_train_tasks) | set(split.d_valid_tasks)),
        PRIOR_FEATURES_DIR
    )
    combined_features = load_llm_judge_combined_features_batch(
        list(set(split.d_train_tasks) | set(split.d_valid_tasks)),
        COMBINED_FEATURES_DIR
    )
    print(f"  Prior features: {len(prior_features)}")
    print(f"  Combined features: {len(combined_features)}")

    # Prepare training data
    # NOTE: The main experiment trains prior on ALL tasks, not just D_train
    # To match that baseline (0.7383), we train embedding prior on all 500 tasks
    all_task_ids = list(items_df.index)
    all_gt = items_df["b"].values

    train_task_ids = split.d_train_tasks
    train_gt = items_df.loc[train_task_ids, "b"].values
    valid_task_ids = split.d_valid_tasks

    results = {}

    # =========================================================================
    # Model 1: Embedding only (baseline) - trained on ALL tasks
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: EMBEDDING ONLY (baseline, trained on ALL 500 tasks)")
    print("=" * 70)

    # Build embedding matrix for training on ALL tasks
    all_emb_tasks = [t for t in all_task_ids if t in embeddings]
    X_all_emb = np.stack([embeddings[t] for t in all_emb_tasks])
    y_all_emb = np.array([all_gt[all_task_ids.index(t)] for t in all_emb_tasks])

    scaler_emb = StandardScaler()
    X_all_emb_scaled = scaler_emb.fit_transform(X_all_emb)

    model_emb = Ridge(alpha=10000.0)
    model_emb.fit(X_all_emb_scaled, y_all_emb)
    print(f"Trained on {len(all_emb_tasks)} tasks")

    # Predict on D_train
    train_preds_emb = {}
    for t in train_task_ids:
        if t in embeddings:
            X = scaler_emb.transform(embeddings[t].reshape(1, -1))
            train_preds_emb[t] = float(model_emb.predict(X)[0])

    train_auc_emb = compute_auc(train_preds_emb, abilities_df, responses, train_task_ids, split.m1_agents)
    print(f"D_train AUC: {train_auc_emb.get('auc', 'N/A'):.4f}")

    # Predict on D_valid
    valid_preds_emb = {}
    for t in valid_task_ids:
        if t in embeddings:
            X = scaler_emb.transform(embeddings[t].reshape(1, -1))
            valid_preds_emb[t] = float(model_emb.predict(X)[0])

    valid_auc_emb = compute_auc(valid_preds_emb, abilities_df, responses, valid_task_ids, split.m2_agents)
    print(f"D_valid AUC: {valid_auc_emb.get('auc', 'N/A'):.4f}")

    results["embedding_only"] = {
        "train_auc": train_auc_emb,
        "valid_auc": valid_auc_emb,
    }

    # =========================================================================
    # Model 2: Embedding + Prior LLM features (9 problem features)
    # Trained on ALL tasks that have LLM features
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: EMBEDDING + PRIOR LLM FEATURES (9 problem features)")
    print("=" * 70)

    # Build combined feature matrix - train on ALL tasks with LLM features
    all_prior_tasks = [t for t in all_task_ids if t in embeddings and t in prior_features]
    X_all_prior = []
    y_all_prior = []
    for t in all_prior_tasks:
        emb = embeddings[t]
        llm = prior_features[t].to_vector()
        X_all_prior.append(np.concatenate([emb, llm]))
        y_all_prior.append(all_gt[all_task_ids.index(t)])

    X_all_prior = np.array(X_all_prior)
    y_all_prior = np.array(y_all_prior)

    print(f"Training on {len(all_prior_tasks)} tasks, {X_all_prior.shape[1]} features")
    print(f"  ({embedding_dim} embedding + {len(LLM_JUDGE_PRIOR_FEATURE_NAMES)} LLM features)")

    scaler_prior = StandardScaler()
    X_all_prior_scaled = scaler_prior.fit_transform(X_all_prior)

    model_prior = Ridge(alpha=10000.0)
    model_prior.fit(X_all_prior_scaled, y_all_prior)

    # Predict on D_train
    train_preds_prior = {}
    for t in train_task_ids:
        if t in embeddings and t in prior_features:
            emb = embeddings[t]
            llm = prior_features[t].to_vector()
            X = np.concatenate([emb, llm]).reshape(1, -1)
            X_scaled = scaler_prior.transform(X)
            train_preds_prior[t] = float(model_prior.predict(X_scaled)[0])

    train_auc_prior = compute_auc(train_preds_prior, abilities_df, responses, train_task_ids, split.m1_agents)
    print(f"D_train AUC: {train_auc_prior.get('auc', 'N/A'):.4f}")

    # Predict on D_valid
    valid_preds_prior = {}
    for t in valid_task_ids:
        if t in embeddings and t in prior_features:
            emb = embeddings[t]
            llm = prior_features[t].to_vector()
            X = np.concatenate([emb, llm]).reshape(1, -1)
            X_scaled = scaler_prior.transform(X)
            valid_preds_prior[t] = float(model_prior.predict(X_scaled)[0])

    valid_auc_prior = compute_auc(valid_preds_prior, abilities_df, responses, valid_task_ids, split.m2_agents)
    print(f"D_valid AUC: {valid_auc_prior.get('auc', 'N/A'):.4f}")

    results["embedding_plus_prior"] = {
        "train_auc": train_auc_prior,
        "valid_auc": valid_auc_prior,
    }

    # =========================================================================
    # Model 3: Embedding + Combined LLM features (13 features)
    # Trained on ALL tasks that have LLM features
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: EMBEDDING + COMBINED LLM FEATURES (13 features)")
    print("=" * 70)

    # Build combined feature matrix - train on ALL tasks with LLM features
    all_combined_tasks = [t for t in all_task_ids if t in embeddings and t in combined_features]
    X_all_combined = []
    y_all_combined = []
    for t in all_combined_tasks:
        emb = embeddings[t]
        llm = combined_features[t].to_vector()
        X_all_combined.append(np.concatenate([emb, llm]))
        y_all_combined.append(all_gt[all_task_ids.index(t)])

    X_all_combined = np.array(X_all_combined)
    y_all_combined = np.array(y_all_combined)

    print(f"Training on {len(all_combined_tasks)} tasks, {X_all_combined.shape[1]} features")
    print(f"  ({embedding_dim} embedding + {len(LLM_JUDGE_COMBINED_FEATURE_NAMES)} LLM features)")

    scaler_combined = StandardScaler()
    X_all_combined_scaled = scaler_combined.fit_transform(X_all_combined)

    model_combined = Ridge(alpha=10000.0)
    model_combined.fit(X_all_combined_scaled, y_all_combined)

    # Predict on D_train
    train_preds_combined = {}
    for t in train_task_ids:
        if t in embeddings and t in combined_features:
            emb = embeddings[t]
            llm = combined_features[t].to_vector()
            X = np.concatenate([emb, llm]).reshape(1, -1)
            X_scaled = scaler_combined.transform(X)
            train_preds_combined[t] = float(model_combined.predict(X_scaled)[0])

    train_auc_combined = compute_auc(train_preds_combined, abilities_df, responses, train_task_ids, split.m1_agents)
    print(f"D_train AUC: {train_auc_combined.get('auc', 'N/A'):.4f}")

    # Predict on D_valid
    valid_preds_combined = {}
    for t in valid_task_ids:
        if t in embeddings and t in combined_features:
            emb = embeddings[t]
            llm = combined_features[t].to_vector()
            X = np.concatenate([emb, llm]).reshape(1, -1)
            X_scaled = scaler_combined.transform(X)
            valid_preds_combined[t] = float(model_combined.predict(X_scaled)[0])

    valid_auc_combined = compute_auc(valid_preds_combined, abilities_df, responses, valid_task_ids, split.m2_agents)
    print(f"D_valid AUC: {valid_auc_combined.get('auc', 'N/A'):.4f}")

    results["embedding_plus_combined"] = {
        "train_auc": train_auc_combined,
        "valid_auc": valid_auc_combined,
    }

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n| Model | Features | D_train AUC | D_valid AUC | Delta vs Emb |")
    print("|-------|----------|-------------|-------------|--------------|")

    emb_valid = results["embedding_only"]["valid_auc"].get("auc", 0)

    for name, label, n_llm in [
        ("embedding_only", "Embedding only", 0),
        ("embedding_plus_prior", "Emb + Prior (9)", 9),
        ("embedding_plus_combined", "Emb + Combined (13)", 13),
    ]:
        train_auc = results[name]["train_auc"].get("auc", 0)
        valid_auc = results[name]["valid_auc"].get("auc", 0)
        delta = valid_auc - emb_valid if name != "embedding_only" else 0
        delta_str = f"{delta:+.4f}" if delta != 0 else "—"
        print(f"| {label:24s} | {embedding_dim}+{n_llm:2d} | {train_auc:.4f} | {valid_auc:.4f} | {delta_str} |")

    # Check if trajectory features help
    prior_valid = results["embedding_plus_prior"]["valid_auc"].get("auc", 0)
    combined_valid = results["embedding_plus_combined"]["valid_auc"].get("auc", 0)

    print("\n" + "-" * 70)
    if combined_valid > emb_valid:
        print(f"=> Embedding + Combined IMPROVES over Embedding only by {combined_valid - emb_valid:+.4f}")
    else:
        print(f"=> Adding LLM features does NOT improve over Embedding only")

    if combined_valid > prior_valid:
        print(f"=> Trajectory features ADD signal: {combined_valid - prior_valid:+.4f} AUC")
    else:
        print(f"=> Trajectory features add NO signal beyond problem features")

    # Save results
    output_path = ROOT / "chris_output" / "experiment_b" / "embedding_plus_llm_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
