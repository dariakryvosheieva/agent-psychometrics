"""Evaluate combining embeddings with Experiment A's LLM judge features.

Uses Experiment A's pre-computed features (500 tasks) for fair comparison.

Usage:
    python -m experiment_b.llm_judge.evaluate_embeddings_with_exp_a_features
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


# Directories
EXP_A_FEATURES_DIR = ROOT / "chris_output" / "experiment_a" / "llm_judge_features"

# Feature names from Experiment A
LLM_JUDGE_FEATURE_NAMES = [
    "fix_in_description",
    "problem_clarity",
    "error_message_provided",
    "reproduction_steps",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
]


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


def load_exp_a_features(features_dir: Path) -> Dict[str, np.ndarray]:
    """Load Experiment A LLM judge features."""
    features = {}
    for json_file in features_dir.glob("*.json"):
        if json_file.name.startswith("compute_stats"):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            task_id = data.get("_instance_id") or json_file.stem

            # Extract feature vector (normalized to 0-1)
            vec = np.array([
                data.get("fix_in_description", 1) / 3.0,
                (data.get("problem_clarity", 3) - 1) / 4.0,
                float(data.get("error_message_provided", 0)),
                float(data.get("reproduction_steps", 0)),
                (data.get("fix_locality", 2) - 1) / 2.0,
                (data.get("domain_knowledge_required", 3) - 1) / 4.0,
                (data.get("fix_complexity", 3) - 1) / 4.0,
                (data.get("logical_reasoning_required", 3) - 1) / 4.0,
                (data.get("atypicality", 3) - 1) / 4.0,
            ])
            features[task_id] = vec
        except (json.JSONDecodeError, IOError, KeyError):
            continue
    return features


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
    print("EMBEDDINGS + EXPERIMENT A LLM JUDGE FEATURES")
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

    all_task_ids = list(items_df.index)
    all_gt = items_df["b"].values

    # Load embeddings
    embeddings_path = ROOT / "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
    print(f"\nLoading embeddings...")
    embeddings = load_embeddings(embeddings_path)
    embedding_dim = len(next(iter(embeddings.values())))
    print(f"  Loaded {len(embeddings)} embeddings, dim={embedding_dim}")

    # Load Experiment A LLM features
    print(f"\nLoading Experiment A LLM judge features...")
    llm_features = load_exp_a_features(EXP_A_FEATURES_DIR)
    print(f"  Loaded {len(llm_features)} task features")

    results = {}

    # =========================================================================
    # Model 1: Embedding only (baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: EMBEDDING ONLY (baseline)")
    print("=" * 70)

    all_emb_tasks = [t for t in all_task_ids if t in embeddings]
    X_all_emb = np.stack([embeddings[t] for t in all_emb_tasks])
    y_all_emb = np.array([all_gt[all_task_ids.index(t)] for t in all_emb_tasks])

    scaler_emb = StandardScaler()
    X_all_emb_scaled = scaler_emb.fit_transform(X_all_emb)

    model_emb = Ridge(alpha=10000.0)
    model_emb.fit(X_all_emb_scaled, y_all_emb)
    print(f"Trained on {len(all_emb_tasks)} tasks")

    # Predict
    train_preds_emb = {}
    for t in split.d_train_tasks:
        if t in embeddings:
            X = scaler_emb.transform(embeddings[t].reshape(1, -1))
            train_preds_emb[t] = float(model_emb.predict(X)[0])

    train_auc_emb = compute_auc(train_preds_emb, abilities_df, responses, split.d_train_tasks, split.m1_agents)
    print(f"D_train AUC: {train_auc_emb.get('auc', 'N/A'):.4f}")

    valid_preds_emb = {}
    for t in split.d_valid_tasks:
        if t in embeddings:
            X = scaler_emb.transform(embeddings[t].reshape(1, -1))
            valid_preds_emb[t] = float(model_emb.predict(X)[0])

    valid_auc_emb = compute_auc(valid_preds_emb, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
    print(f"D_valid AUC: {valid_auc_emb.get('auc', 'N/A'):.4f}")

    results["embedding_only"] = {
        "train_auc": train_auc_emb,
        "valid_auc": valid_auc_emb,
    }

    # =========================================================================
    # Model 2: LLM features only (9 problem features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: LLM FEATURES ONLY (9 problem features)")
    print("=" * 70)

    all_llm_tasks = [t for t in all_task_ids if t in llm_features]
    X_all_llm = np.stack([llm_features[t] for t in all_llm_tasks])
    y_all_llm = np.array([all_gt[all_task_ids.index(t)] for t in all_llm_tasks])

    scaler_llm = StandardScaler()
    X_all_llm_scaled = scaler_llm.fit_transform(X_all_llm)

    model_llm = Ridge(alpha=1.0)
    model_llm.fit(X_all_llm_scaled, y_all_llm)
    print(f"Trained on {len(all_llm_tasks)} tasks")

    # Predict
    train_preds_llm = {}
    for t in split.d_train_tasks:
        if t in llm_features:
            X = scaler_llm.transform(llm_features[t].reshape(1, -1))
            train_preds_llm[t] = float(model_llm.predict(X)[0])

    train_auc_llm = compute_auc(train_preds_llm, abilities_df, responses, split.d_train_tasks, split.m1_agents)
    print(f"D_train AUC: {train_auc_llm.get('auc', 'N/A'):.4f}")

    valid_preds_llm = {}
    for t in split.d_valid_tasks:
        if t in llm_features:
            X = scaler_llm.transform(llm_features[t].reshape(1, -1))
            valid_preds_llm[t] = float(model_llm.predict(X)[0])

    valid_auc_llm = compute_auc(valid_preds_llm, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
    print(f"D_valid AUC: {valid_auc_llm.get('auc', 'N/A'):.4f}")

    results["llm_only"] = {
        "train_auc": train_auc_llm,
        "valid_auc": valid_auc_llm,
    }

    # Print feature importance
    print("\nFeature importance (LLM only):")
    importance = dict(zip(LLM_JUDGE_FEATURE_NAMES, model_llm.coef_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_imp[:5]:
        print(f"  {name}: {coef:.4f}")

    # =========================================================================
    # Model 3: Embedding + LLM features
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: EMBEDDING + LLM FEATURES (4096 + 9)")
    print("=" * 70)

    all_combined_tasks = [t for t in all_task_ids if t in embeddings and t in llm_features]
    X_all_combined = []
    y_all_combined = []
    for t in all_combined_tasks:
        emb = embeddings[t]
        llm = llm_features[t]
        X_all_combined.append(np.concatenate([emb, llm]))
        y_all_combined.append(all_gt[all_task_ids.index(t)])

    X_all_combined = np.array(X_all_combined)
    y_all_combined = np.array(y_all_combined)

    print(f"Training on {len(all_combined_tasks)} tasks, {X_all_combined.shape[1]} features")

    scaler_combined = StandardScaler()
    X_all_combined_scaled = scaler_combined.fit_transform(X_all_combined)

    model_combined = Ridge(alpha=10000.0)
    model_combined.fit(X_all_combined_scaled, y_all_combined)

    # Predict
    train_preds_combined = {}
    for t in split.d_train_tasks:
        if t in embeddings and t in llm_features:
            X = np.concatenate([embeddings[t], llm_features[t]]).reshape(1, -1)
            X_scaled = scaler_combined.transform(X)
            train_preds_combined[t] = float(model_combined.predict(X_scaled)[0])

    train_auc_combined = compute_auc(train_preds_combined, abilities_df, responses, split.d_train_tasks, split.m1_agents)
    print(f"D_train AUC: {train_auc_combined.get('auc', 'N/A'):.4f}")

    valid_preds_combined = {}
    for t in split.d_valid_tasks:
        if t in embeddings and t in llm_features:
            X = np.concatenate([embeddings[t], llm_features[t]]).reshape(1, -1)
            X_scaled = scaler_combined.transform(X)
            valid_preds_combined[t] = float(model_combined.predict(X_scaled)[0])

    valid_auc_combined = compute_auc(valid_preds_combined, abilities_df, responses, split.d_valid_tasks, split.m2_agents)
    print(f"D_valid AUC: {valid_auc_combined.get('auc', 'N/A'):.4f}")

    results["embedding_plus_llm"] = {
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

    for name, label, n_feat in [
        ("embedding_only", "Embedding only", f"{embedding_dim}"),
        ("llm_only", "LLM only", "9"),
        ("embedding_plus_llm", "Emb + LLM", f"{embedding_dim}+9"),
    ]:
        train_auc = results[name]["train_auc"].get("auc", 0)
        valid_auc = results[name]["valid_auc"].get("auc", 0)
        delta = valid_auc - emb_valid if name != "embedding_only" else 0
        delta_str = f"{delta:+.4f}" if delta != 0 else "—"
        print(f"| {label:18s} | {n_feat:8s} | {train_auc:.4f} | {valid_auc:.4f} | {delta_str} |")

    print("\n" + "-" * 70)
    llm_valid = results["llm_only"]["valid_auc"].get("auc", 0)
    combined_valid = results["embedding_plus_llm"]["valid_auc"].get("auc", 0)

    if combined_valid > emb_valid:
        print(f"=> Embedding + LLM IMPROVES over Embedding only by {combined_valid - emb_valid:+.4f}")
    else:
        print(f"=> Adding LLM features does NOT improve over Embedding only ({combined_valid - emb_valid:+.4f})")

    # Save results
    output_path = ROOT / "chris_output" / "experiment_b" / "embedding_plus_exp_a_features.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
