"""Ablation study: Full MLP on embedding features.

Tests the FullMLPPredictor which takes [agent_one_hot | task_features] as
concatenated input through a hidden layer. This is the PRIMARY MLP experiment
for trying to beat Ridge regression on embeddings.

IMPORTANT: This uses FullMLPPredictor, NOT the IRT-style MLPPredictor.
The IRT-style MLP (MLPPredictor) was only a sanity check to verify that
the IRT formula works - it cannot exceed IRT performance by design.
To beat Ridge, we need a more flexible architecture that can learn
arbitrary agent-task interactions.

Key insight from initial experiments: The model was UNDERFITTING (train ≈ test AUC),
not overfitting. Strong regularization (wd=100-1000) made things worse.
This suggests the hidden layer bottleneck (128 units for 5120-dim embeddings)
was too restrictive.

Ablation dimensions:
1. Hidden size: [128, 256, 512, 1024, 2048] - MOST IMPORTANT (address underfitting)
2. Weight decay: [0.01, 0.1, 1.0, 10.0] - moderate values since not overfitting
3. Early stopping: prevents overtraining
4. All experiments use IRT ability initialization for stable agent representations

Usage:
    python -m experiment_a.mlp_ablation.test_full_mlp

    # Quick test with fewer configs
    python -m experiment_a.mlp_ablation.test_full_mlp --quick

    # Parallel execution on SLURM
    sbatch experiment_a/mlp_ablation/slurm_full_mlp.sh
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv, CVPredictor
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import FullMLPPredictor
from experiment_a.shared.baselines import (
    OraclePredictor,
    ConstantPredictor,
    DifficultyPredictorAdapter,
)
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold


def extract_train_auc(predictor: CVPredictor, fold_idx: int) -> float | None:
    """Extract train AUC from fitted predictor for diagnostics."""
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None

ROOT = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Full MLP on embeddings ablation")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--part", type=int, choices=[1, 2], default=None,
                        help="Run only part 1 or 2 of configs (for parallel execution)")
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

    # Ridge baseline
    configs.append(
        CVPredictorConfig(
            predictor=DifficultyPredictorAdapter(
                FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
            ),
            name="ridge",
            display_name="Ridge (Embedding)",
        )
    )

    # === Full MLP Experiments ===
    # Goal: Beat Ridge (0.823 AUC) by learning agent-task interactions
    # All experiments use IRT ability initialization for stable agent representations
    # Key insight: Ridge succeeds with alpha=1000-10000, so we need very strong regularization

    # Ablation parameters
    # NOTE: Results show underfitting (train ≈ test), not overfitting
    # So we try larger hidden sizes and lower regularization
    if args.quick:
        hidden_sizes = [256, 512]
        weight_decays = [0.1, 1.0]
    else:
        hidden_sizes = [128, 256, 512, 1024, 2048]
        weight_decays = [0.01, 0.1, 1.0, 10.0]

    # 1. Strong regularization sweep (PRIMARY EXPERIMENT)
    # All use IRT init since it provides stable agent representations
    for wd in weight_decays:
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=128,
                    weight_decay=wd,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    verbose=True,
                ),
                name=f"full_mlp_wd{wd}_irt",
                display_name=f"FullMLP (wd={wd}, IRT)",
            )
        )

    # 2. Hidden size ablation with moderate regularization
    # Using wd=1.0 since results showed underfitting, not overfitting
    for hidden_size in hidden_sizes:
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=hidden_size,
                    weight_decay=1.0,  # Moderate reg (underfitting)
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    verbose=True,
                ),
                name=f"full_mlp_h{hidden_size}_wd1_irt",
                display_name=f"FullMLP (h={hidden_size}, wd=1, IRT)",
            )
        )

    # 3. Early stopping with larger hidden sizes
    # Test if larger capacity + early stopping helps with underfitting
    for hidden_size in [512, 1024]:
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=hidden_size,
                    weight_decay=0.1,  # Low reg for larger models
                    learning_rate=learning_rate,
                    n_epochs=1000,  # More epochs since early stopping will cut it
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"full_mlp_h{hidden_size}_wd0.1_irt_earlystop",
                display_name=f"FullMLP (h={hidden_size}, wd=0.1, IRT, early stop)",
            )
        )

    # 4. Best config candidates (combining larger capacity with good settings)
    if not args.quick:
        # Very large hidden + low regularization
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=2048,
                    weight_decay=0.1,
                    learning_rate=learning_rate,
                    n_epochs=1000,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name="full_mlp_best_v1",
                display_name="FullMLP (h=2048, wd=0.1, IRT, early stop)",
            )
        )

        # Large hidden + minimal regularization
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=1024,
                    weight_decay=0.01,
                    learning_rate=learning_rate,
                    n_epochs=1000,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name="full_mlp_best_v2",
                display_name="FullMLP (h=1024, wd=0.01, IRT, early stop)",
            )
        )

        # Large hidden + dropout (alternative regularization)
        configs.append(
            CVPredictorConfig(
                predictor=FullMLPPredictor(
                    embedding_source,
                    hidden_size=1024,
                    dropout=0.3,
                    weight_decay=0.1,
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    verbose=True,
                ),
                name="full_mlp_dropout",
                display_name="FullMLP (h=1024, dropout=0.3, wd=0.1, IRT)",
            )
        )

    # Filter configs by part if specified (for parallel execution)
    if args.part is not None:
        # Split configs roughly in half
        # Part 1: Baselines + weight decay sweep + small hidden sizes
        # Part 2: Large hidden sizes + early stopping + best configs
        part1_keywords = ["oracle", "constant", "ridge", "full_mlp_wd0.01", "full_mlp_wd0.1", "full_mlp_wd1.0", "full_mlp_wd10.0", "full_mlp_h128", "full_mlp_h256"]
        part2_keywords = ["full_mlp_h512", "full_mlp_h1024", "full_mlp_h2048", "earlystop", "full_mlp_best", "full_mlp_dropout"]

        if args.part == 1:
            configs = [c for c in configs if any(kw in c.name for kw in part1_keywords)]
        else:
            configs = [c for c in configs if any(kw in c.name for kw in part2_keywords)]

        print(f"\n*** Running PART {args.part} only ({len(configs)} configs) ***")

    # Run CV
    results = {}

    print("\n" + "=" * 80)
    print("FULL MLP ON EMBEDDINGS ABLATION")
    print(f"Feature dim: {embedding_source.feature_dim}, Epochs: {n_epochs}, LR: {learning_rate}")
    print(f"Configs to run: {len(configs)}")
    print("=" * 80)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        cv_result = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            diagnostics_extractor=extract_train_auc,
        )

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
        }

        # Get mean train AUC from fold diagnostics
        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))
                results[pc.name]["fold_train_aucs"] = valid_train_aucs

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} ± {cv_result.std_auc:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 80)

    # Sort by test AUC descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc", None)
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Highlight key comparisons
    print("\n" + "-" * 80)
    print("KEY COMPARISONS:")
    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

        # Find best Full MLP
        best_full = max(
            [(name, r) for name, r in results.items() if name.startswith("full_mlp")],
            key=lambda x: x[1]["mean_auc"],
            default=None
        )
        if best_full:
            best_name, best_r = best_full
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best FullMLP: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.part is not None:
        output_path = output_dir / f"full_mlp_results_part{args.part}.json"
    else:
        output_path = output_dir / "full_mlp_results.json"

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
