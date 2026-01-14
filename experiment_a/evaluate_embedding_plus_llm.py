"""Evaluate Embedding + LLM features for Experiment A.

This adds a new entry to the Experiment A results table:
- Embedding + LLM Judge: combines 4096-dim embeddings with 9 LLM features

Usage:
    python -m experiment_a.evaluate_embedding_plus_llm
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_a.config import ExperimentAConfig
from experiment_a.data_loader import load_experiment_data


# Directories
LLM_FEATURES_DIR = ROOT / "chris_output" / "experiment_a" / "llm_judge_features"

# Feature names
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


def load_embeddings(embeddings_path: Path) -> Dict[str, np.ndarray]:
    """Load embeddings from .npz file."""
    data = np.load(embeddings_path, allow_pickle=True)
    task_ids = [str(x) for x in data["task_ids"].tolist()]
    X = data["X"].astype(np.float32)
    return {task_id: X[i] for i, task_id in enumerate(task_ids)}


def load_llm_features(features_dir: Path) -> Dict[str, np.ndarray]:
    """Load LLM judge features."""
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
    predictions: Dict[str, float],
    abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    test_tasks: List[str],
) -> Dict:
    """Compute AUC using IRT formula P(success) = sigmoid(theta - beta)."""
    y_true = []
    y_scores = []

    for task_id in test_tasks:
        if task_id not in predictions:
            continue
        beta_pred = predictions[task_id]

        for agent_id in abilities.index:
            if agent_id not in responses:
                continue
            if task_id not in responses[agent_id]:
                continue

            theta = float(abilities.loc[agent_id, "theta"])
            actual = responses[agent_id][task_id]
            prob = float(expit(theta - beta_pred))

            y_true.append(int(actual))
            y_scores.append(prob)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"error": "Insufficient data", "n_pairs": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)
    return {"auc": float(auc), "n_pairs": len(y_true)}


def compute_pearson_r(
    predictions: Dict[str, float],
    items: pd.DataFrame,
    test_tasks: List[str],
) -> Dict:
    """Compute Pearson correlation with ground truth."""
    common = [t for t in test_tasks if t in predictions and t in items.index]
    if len(common) < 3:
        return {"error": "Insufficient data"}

    pred = np.array([predictions[t] for t in common])
    true = np.array([items.loc[t, "b"] for t in common])

    r, p = stats.pearsonr(pred, true)
    return {"pearson_r": float(r), "p_value": float(p), "n": len(common)}


def main():
    print("=" * 70)
    print("EXPERIMENT A: EMBEDDING + LLM JUDGE EVALUATION")
    print("=" * 70)

    # Load config and data using Experiment A setup
    config = ExperimentAConfig()

    # Resolve paths
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    print("\nLoading data...")
    data = load_experiment_data(
        abilities_path=abilities_path,
        items_path=items_path,
        responses_path=responses_path,
        test_fraction=config.test_fraction,
        split_seed=config.split_seed,
    )
    print(f"  Agents: {data.n_agents}")
    print(f"  Tasks: {data.n_tasks}")
    print(f"  Train tasks: {data.n_train_tasks}")
    print(f"  Test tasks: {data.n_test_tasks}")

    # Get training data
    train_tasks = data.train_tasks
    test_tasks = data.test_tasks
    train_b = data.items.loc[train_tasks, "b"].values

    # Load embeddings (use the known path)
    embeddings_path = ROOT / "out/prior_qwen3vl8b/embeddings__Qwen__Qwen3-VL-8B-Instruct__pool-lasttoken__qs-sol-instr__qs_sol_instr_b7008f2d__idnorm_instance-v1__princeton-nlp_SWE-bench_Verified__test__n500__maxlen8192__seed0.npz"
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = load_embeddings(embeddings_path)
    embedding_dim = len(next(iter(embeddings.values())))
    print(f"  Loaded {len(embeddings)} embeddings, dim={embedding_dim}")

    # Load LLM features
    print(f"\nLoading LLM judge features...")
    llm_features = load_llm_features(LLM_FEATURES_DIR)
    print(f"  Loaded {len(llm_features)} task features")

    results = {}

    # =========================================================================
    # Model 1: Oracle (ground truth b)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ORACLE (ground truth b)")
    print("=" * 70)

    oracle_preds = {t: float(data.items.loc[t, "b"]) for t in test_tasks}
    oracle_auc = compute_auc(oracle_preds, data.abilities, data.responses, test_tasks)
    oracle_r = compute_pearson_r(oracle_preds, data.items, test_tasks)
    print(f"AUC: {oracle_auc.get('auc', 'N/A'):.4f}")
    print(f"Pearson r: {oracle_r.get('pearson_r', 'N/A'):.4f}")
    results["oracle"] = {"auc": oracle_auc, "pearson_r": oracle_r}

    # =========================================================================
    # Model 2: Embedding only
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: EMBEDDING ONLY (Qwen3-VL-8B)")
    print("=" * 70)

    train_emb_tasks = [t for t in train_tasks if t in embeddings]
    X_train = np.stack([embeddings[t] for t in train_emb_tasks])
    y_train = np.array([train_b[train_tasks.index(t)] for t in train_emb_tasks])

    scaler_emb = StandardScaler()
    X_train_scaled = scaler_emb.fit_transform(X_train)

    ridge_alpha = 10000.0  # Standard alpha for embedding predictor
    model_emb = Ridge(alpha=ridge_alpha)
    model_emb.fit(X_train_scaled, y_train)
    print(f"Trained on {len(train_emb_tasks)} tasks, alpha={ridge_alpha}")

    emb_preds = {}
    for t in test_tasks:
        if t in embeddings:
            X = scaler_emb.transform(embeddings[t].reshape(1, -1))
            emb_preds[t] = float(model_emb.predict(X)[0])

    emb_auc = compute_auc(emb_preds, data.abilities, data.responses, test_tasks)
    emb_r = compute_pearson_r(emb_preds, data.items, test_tasks)
    print(f"AUC: {emb_auc.get('auc', 'N/A'):.4f}")
    print(f"Pearson r: {emb_r.get('pearson_r', 'N/A'):.4f}")
    results["embedding"] = {"auc": emb_auc, "pearson_r": emb_r}

    # =========================================================================
    # Model 3: LLM Judge only (9 features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: LLM JUDGE ONLY (9 semantic features)")
    print("=" * 70)

    train_llm_tasks = [t for t in train_tasks if t in llm_features]
    X_train_llm = np.stack([llm_features[t] for t in train_llm_tasks])
    y_train_llm = np.array([train_b[train_tasks.index(t)] for t in train_llm_tasks])

    scaler_llm = StandardScaler()
    X_train_llm_scaled = scaler_llm.fit_transform(X_train_llm)

    model_llm = Ridge(alpha=1.0)
    model_llm.fit(X_train_llm_scaled, y_train_llm)
    print(f"Trained on {len(train_llm_tasks)} tasks")

    llm_preds = {}
    for t in test_tasks:
        if t in llm_features:
            X = scaler_llm.transform(llm_features[t].reshape(1, -1))
            llm_preds[t] = float(model_llm.predict(X)[0])

    llm_auc = compute_auc(llm_preds, data.abilities, data.responses, test_tasks)
    llm_r = compute_pearson_r(llm_preds, data.items, test_tasks)
    print(f"AUC: {llm_auc.get('auc', 'N/A'):.4f}")
    print(f"Pearson r: {llm_r.get('pearson_r', 'N/A'):.4f}")
    results["llm_judge"] = {"auc": llm_auc, "pearson_r": llm_r}

    # Print feature coefficients
    print("\nFeature importance:")
    importance = dict(zip(LLM_JUDGE_FEATURE_NAMES, model_llm.coef_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in sorted_imp:
        print(f"  {name}: {coef:+.4f}")

    # =========================================================================
    # Model 4: Embedding + LLM Judge (4096 + 9 features)
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL 4: EMBEDDING + LLM JUDGE (4096 + 9 features)")
    print("=" * 70)

    train_combined_tasks = [t for t in train_tasks if t in embeddings and t in llm_features]
    X_train_combined = []
    y_train_combined = []
    for t in train_combined_tasks:
        X_train_combined.append(np.concatenate([embeddings[t], llm_features[t]]))
        y_train_combined.append(train_b[train_tasks.index(t)])

    X_train_combined = np.array(X_train_combined)
    y_train_combined = np.array(y_train_combined)

    print(f"Training on {len(train_combined_tasks)} tasks, {X_train_combined.shape[1]} features")

    scaler_combined = StandardScaler()
    X_train_combined_scaled = scaler_combined.fit_transform(X_train_combined)

    model_combined = Ridge(alpha=ridge_alpha)
    model_combined.fit(X_train_combined_scaled, y_train_combined)

    combined_preds = {}
    for t in test_tasks:
        if t in embeddings and t in llm_features:
            X = np.concatenate([embeddings[t], llm_features[t]]).reshape(1, -1)
            X_scaled = scaler_combined.transform(X)
            combined_preds[t] = float(model_combined.predict(X_scaled)[0])

    combined_auc = compute_auc(combined_preds, data.abilities, data.responses, test_tasks)
    combined_r = compute_pearson_r(combined_preds, data.items, test_tasks)
    print(f"AUC: {combined_auc.get('auc', 'N/A'):.4f}")
    print(f"Pearson r: {combined_r.get('pearson_r', 'N/A'):.4f}")
    results["embedding_plus_llm"] = {"auc": combined_auc, "pearson_r": combined_r}

    # =========================================================================
    # Model 5-8: PCA embeddings at various dimensions + LLM features
    # =========================================================================
    print("\n" + "=" * 70)
    print("PCA EMBEDDING EXPERIMENTS")
    print("=" * 70)

    # Test various PCA dimensions
    pca_dims = [16, 32, 64, 128, 256]

    for n_components in pca_dims:
        print(f"\n--- PCA dim={n_components} ---")

        # Fit PCA on training embeddings
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        variance_explained = sum(pca.explained_variance_ratio_)
        print(f"Variance explained: {variance_explained:.3f}")

        # Model A: PCA embedding only
        scaler_pca = StandardScaler()
        X_train_pca_scaled = scaler_pca.fit_transform(X_train_pca)

        model_pca = Ridge(alpha=1.0)  # Lower alpha for lower-dim features
        model_pca.fit(X_train_pca_scaled, y_train)

        pca_preds = {}
        for t in test_tasks:
            if t in embeddings:
                X_pca = pca.transform(embeddings[t].reshape(1, -1))
                X_scaled = scaler_pca.transform(X_pca)
                pca_preds[t] = float(model_pca.predict(X_scaled)[0])

        pca_auc = compute_auc(pca_preds, data.abilities, data.responses, test_tasks)
        pca_r = compute_pearson_r(pca_preds, data.items, test_tasks)
        print(f"PCA only - AUC: {pca_auc.get('auc', 'N/A'):.4f}, r: {pca_r.get('pearson_r', 'N/A'):.3f}")
        results[f"pca_{n_components}"] = {"auc": pca_auc, "pearson_r": pca_r}

        # Model B: PCA embedding + LLM features
        train_pca_llm_tasks = [t for t in train_tasks if t in embeddings and t in llm_features]
        X_train_pca_llm = []
        y_train_pca_llm = []
        for t in train_pca_llm_tasks:
            emb_pca = pca.transform(embeddings[t].reshape(1, -1)).flatten()
            X_train_pca_llm.append(np.concatenate([emb_pca, llm_features[t]]))
            y_train_pca_llm.append(train_b[train_tasks.index(t)])

        X_train_pca_llm = np.array(X_train_pca_llm)
        y_train_pca_llm = np.array(y_train_pca_llm)

        scaler_pca_llm = StandardScaler()
        X_train_pca_llm_scaled = scaler_pca_llm.fit_transform(X_train_pca_llm)

        model_pca_llm = Ridge(alpha=1.0)
        model_pca_llm.fit(X_train_pca_llm_scaled, y_train_pca_llm)

        pca_llm_preds = {}
        for t in test_tasks:
            if t in embeddings and t in llm_features:
                emb_pca = pca.transform(embeddings[t].reshape(1, -1)).flatten()
                X = np.concatenate([emb_pca, llm_features[t]]).reshape(1, -1)
                X_scaled = scaler_pca_llm.transform(X)
                pca_llm_preds[t] = float(model_pca_llm.predict(X_scaled)[0])

        pca_llm_auc = compute_auc(pca_llm_preds, data.abilities, data.responses, test_tasks)
        pca_llm_r = compute_pearson_r(pca_llm_preds, data.items, test_tasks)
        delta = pca_llm_auc.get('auc', 0) - pca_auc.get('auc', 0)
        print(f"PCA + LLM - AUC: {pca_llm_auc.get('auc', 'N/A'):.4f}, r: {pca_llm_r.get('pearson_r', 'N/A'):.3f} (delta: {delta:+.4f})")
        results[f"pca_{n_components}_plus_llm"] = {"auc": pca_llm_auc, "pearson_r": pca_llm_r}

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY - EXPERIMENT A RESULTS")
    print("=" * 70)

    print("\n| Method | AUC | Pearson r | Notes |")
    print("|--------|-----|-----------|-------|")

    rows = [
        ("Oracle (true b)", results["oracle"]["auc"].get("auc", 0), results["oracle"]["pearson_r"].get("pearson_r", 0), "Upper bound"),
        ("Embedding (Qwen3-VL-8B)", results["embedding"]["auc"].get("auc", 0), results["embedding"]["pearson_r"].get("pearson_r", 0), "4096-dim embeddings"),
        ("LLM Judge", results["llm_judge"]["auc"].get("auc", 0), results["llm_judge"]["pearson_r"].get("pearson_r", 0), "9 semantic features"),
        ("**Embedding + LLM**", results["embedding_plus_llm"]["auc"].get("auc", 0), results["embedding_plus_llm"]["pearson_r"].get("pearson_r", 0), "4096 + 9 features"),
    ]

    for method, auc, r, notes in rows:
        print(f"| {method:24s} | {auc:.4f} | {r:.3f} | {notes} |")

    # PCA summary
    print("\n" + "-" * 70)
    print("PCA EXPERIMENTS:")
    print("-" * 70)
    print("\n| PCA dim | Variance | PCA only AUC | PCA+LLM AUC | Delta |")
    print("|---------|----------|--------------|-------------|-------|")
    for n_components in pca_dims:
        pca_key = f"pca_{n_components}"
        pca_llm_key = f"pca_{n_components}_plus_llm"
        pca_auc = results[pca_key]["auc"].get("auc", 0)
        pca_llm_auc = results[pca_llm_key]["auc"].get("auc", 0)
        delta = pca_llm_auc - pca_auc
        # Compute variance explained (stored in results or re-compute would need saving)
        print(f"| {n_components:7d} |   —      | {pca_auc:.4f}       | {pca_llm_auc:.4f}      | {delta:+.4f} |")

    # Delta analysis
    emb_auc_val = results["embedding"]["auc"].get("auc", 0)
    combined_auc_val = results["embedding_plus_llm"]["auc"].get("auc", 0)
    delta = combined_auc_val - emb_auc_val

    print(f"\n=> Delta (Emb+LLM vs Emb): {delta:+.4f} AUC")

    if delta > 0:
        print("   LLM features ADD signal on top of embeddings!")
    else:
        print("   LLM features do NOT add signal beyond embeddings")

    # Save results
    output_path = ROOT / "chris_output" / "experiment_a" / "embedding_plus_llm_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
