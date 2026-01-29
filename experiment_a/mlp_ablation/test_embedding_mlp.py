"""Ablation study: MLP on embedding features (5120-dim).

For embeddings, the problem is likely reversed from LLM Judge:
- LLM Judge (9 params): Agent abilities (131 params) dominate → agents memorize
- Embeddings (5120 params): Feature weights dominate → features overfit

This script tests various strategies to prevent embedding overfitting:
1. Strong feature regularization (feature_weight_decay)
2. IRT initialization + regularization combo
3. Two-stage training with strong regularization

Usage:
    python -m experiment_a.mlp_ablation.test_embedding_mlp

    # Quick test with fewer configs
    python -m experiment_a.mlp_ablation.test_embedding_mlp --quick
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import MLPPredictor
from experiment_a.shared.baselines import (
    OraclePredictor,
    ConstantPredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold

ROOT = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="MLP on embeddings ablation")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    # Build embedding source
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    # Load task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    # Generate folds
    k_folds = args.k_folds
    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    # Create fold data loader
    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # Training parameters
    n_epochs = 500
    learning_rate = 0.01

    configs: List[CVPredictorConfig] = []

    # === Baselines ===
    configs.append(
        CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true β)",
        )
    )
    configs.append(
        CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant",
            display_name="Constant (mean β)",
        )
    )

    # Ridge baseline (what we're trying to match)
    configs.append(
        CVPredictorConfig(
            predictor=DifficultyPredictorAdapter(
                FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
            ),
            name="ridge",
            display_name="Ridge (Embedding)",
        )
    )

    # === MLP Experiments ===

    # Feature regularization values to test
    if args.quick:
        feature_wds = [0.01, 1.0, 10.0]
    else:
        feature_wds = [0.01, 0.1, 1.0, 10.0, 100.0]

    # 1. Baseline: MLP learned from scratch with varying feature_weight_decay
    for wd in feature_wds:
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=False,
                    two_stage=False,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    feature_weight_decay=wd,
                    weight_decay=0.01,  # Agent weight decay
                    verbose=True,
                ),
                name=f"mlp_baseline_wd{wd}",
                display_name=f"MLP baseline (wd={wd})",
            )
        )

    # 2. Frozen IRT abilities with varying feature_weight_decay
    for wd in feature_wds:
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=True,
                    two_stage=False,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    feature_weight_decay=wd,
                    verbose=True,
                ),
                name=f"mlp_frozen_wd{wd}",
                display_name=f"MLP frozen IRT (wd={wd})",
            )
        )

    # 3. Two-stage training with varying feature_weight_decay
    for wd in feature_wds:
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=False,
                    two_stage=True,
                    stage1_epochs=n_epochs // 2,
                    stage2_agent_lr_scale=0.1,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    feature_weight_decay=wd,
                    weight_decay=0.01,  # Agent weight decay
                    verbose=True,
                ),
                name=f"mlp_twostage_wd{wd}",
                display_name=f"MLP two-stage (wd={wd})",
            )
        )

    # 4. Test dropout for embeddings (only with frozen IRT to isolate effect)
    if not args.quick:
        for dropout in [0.3, 0.5]:
            configs.append(
                CVPredictorConfig(
                    predictor=MLPPredictor(
                        embedding_source,
                        freeze_abilities=True,
                        two_stage=False,
                        learning_rate=learning_rate,
                        n_epochs=n_epochs,
                        feature_weight_decay=1.0,
                        dropout=dropout,
                        verbose=True,
                    ),
                    name=f"mlp_frozen_dropout{dropout}",
                    display_name=f"MLP frozen + dropout={dropout}",
                )
            )

    # 5. PCA dimensionality reduction experiments
    # PCA is fit on training data within each CV fold (no data leakage)
    pca_dims = [128, 256] if args.quick else [128, 256, 512]
    for pca_dim in pca_dims:
        # PCA with baseline training
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=False,
                    two_stage=False,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    feature_weight_decay=0.1,  # Lower reg for reduced dim
                    weight_decay=0.01,
                    pca_dim=pca_dim,
                    verbose=True,
                ),
                name=f"mlp_pca{pca_dim}_baseline",
                display_name=f"MLP PCA-{pca_dim} baseline",
            )
        )

        # PCA with frozen IRT
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=True,
                    two_stage=False,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    feature_weight_decay=0.1,
                    pca_dim=pca_dim,
                    verbose=True,
                ),
                name=f"mlp_pca{pca_dim}_frozen",
                display_name=f"MLP PCA-{pca_dim} frozen IRT",
            )
        )

        # PCA with two-stage
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=False,
                    two_stage=True,
                    stage1_epochs=n_epochs // 2,
                    stage2_agent_lr_scale=0.1,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    feature_weight_decay=0.1,
                    weight_decay=0.01,
                    pca_dim=pca_dim,
                    verbose=True,
                ),
                name=f"mlp_pca{pca_dim}_twostage",
                display_name=f"MLP PCA-{pca_dim} two-stage",
            )
        )

    # 6. Early stopping experiments (prevents overfitting)
    if not args.quick:
        # Early stopping with frozen IRT abilities
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=True,
                    learning_rate=learning_rate,
                    n_epochs=1000,  # More epochs since early stopping will cut it short
                    feature_weight_decay=1.0,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name="mlp_frozen_earlystop",
                display_name="MLP frozen + early stopping",
            )
        )

        # Early stopping with strong regularization (no IRT init)
        # Tests if early stopping alone can prevent feature overfitting
        configs.append(
            CVPredictorConfig(
                predictor=MLPPredictor(
                    embedding_source,
                    freeze_abilities=False,
                    learning_rate=learning_rate,
                    n_epochs=1000,
                    feature_weight_decay=10.0,
                    weight_decay=0.01,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name="mlp_baseline_earlystop_wd10",
                display_name="MLP baseline + early stop (wd=10)",
            )
        )

    # Run CV
    results = {}

    print("\n" + "=" * 80)
    print("MLP ON EMBEDDINGS ABLATION")
    print(f"Feature dim: {embedding_source.feature_dim}, Epochs: {n_epochs}, LR: {learning_rate}")
    print("=" * 80)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        cv_result = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
        )

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
        }

        # Get train AUC if available (for overfitting diagnosis)
        if hasattr(pc.predictor, 'get_train_auc') and pc.predictor.get_train_auc() is not None:
            results[pc.name]["train_auc"] = pc.predictor.get_train_auc()

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} ± {cv_result.std_auc:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<35} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 70)

    # Sort by test AUC descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc", None)
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<35} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<35} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Highlight key comparisons
    print("\n" + "-" * 70)
    print("KEY COMPARISONS:")
    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best MLP
        best_mlp = max(
            [(name, r) for name, r in results.items() if name.startswith("mlp_")],
            key=lambda x: x[1]["mean_auc"],
            default=None
        )
        if best_mlp:
            best_name, best_r = best_mlp
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best MLP: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embedding_mlp_results.json"

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
