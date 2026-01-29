"""Test Daria-style training setup with mini-batches.

Matches Daria's predict_agent_task_success.py as closely as possible:
- Mini-batch training (batch_size=256)
- 10 epochs, no early stopping
- AdamW with lr=0.001, weight_decay=0.01
- Dropout=0.1
- Hidden=[256, 128]
- Random initialization

Usage:
    python -m experiment_a.mlp_ablation.test_daria_style
    python -m experiment_a.mlp_ablation.test_daria_style --quick
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import AgentEmbeddingPredictor
from experiment_a.shared.baselines import OraclePredictor, DifficultyPredictorAdapter
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold


def extract_train_auc(predictor, fold_idx):
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None


ROOT = Path(__file__).parent.parent.parent


def main():
    main_start = time.time()
    print(f"Script starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Test Daria-style training")
    parser.add_argument("--quick", action="store_true", help="Quick test with 2 folds")
    args = parser.parse_args()

    config = ExperimentAConfig()
    k_folds = 2 if args.quick else 5

    embeddings_path = ROOT / config.embeddings_path
    items_path = ROOT / config.items_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"Loading embeddings...")
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    folds = k_fold_split_tasks(all_task_ids, k=k_folds, seed=config.split_seed)

    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=ROOT / config.abilities_path,
            items_path=ROOT / config.items_path,
            responses_path=ROOT / config.responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    configs: List[CVPredictorConfig] = []

    # Baselines
    configs.append(CVPredictorConfig(
        predictor=OraclePredictor(),
        name="oracle",
        display_name="Oracle (true beta)",
    ))
    configs.append(CVPredictorConfig(
        predictor=DifficultyPredictorAdapter(
            FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
        ),
        name="ridge",
        display_name="Ridge (Embedding)",
    ))

    # Daria-style: mini-batch, 10 epochs, no early stopping, random init
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=64,
            hidden_sizes=[256, 128],
            dropout=0.1,
            learning_rate=0.001,
            weight_decay=0.01,
            n_epochs=10,
            batch_size=256,  # Mini-batch!
            init_from_irt=False,  # Random init
            early_stopping=False,  # No early stopping
            verbose=True,
        ),
        name="daria_style_random",
        display_name="Daria-style (mini-batch, random)",
    ))

    # Daria-style with IRT init for comparison
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=64,
            hidden_sizes=[256, 128],
            dropout=0.1,
            learning_rate=0.001,
            weight_decay=0.01,
            n_epochs=10,
            batch_size=256,
            init_from_irt=True,
            early_stopping=False,
            verbose=True,
        ),
        name="daria_style_irt",
        display_name="Daria-style (mini-batch, IRT)",
    ))

    # Full-batch baseline (our previous best random: high LR)
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=64,
            hidden_sizes=[256, 128],
            dropout=0.1,
            learning_rate=0.01,  # Higher LR
            weight_decay=0.01,
            n_epochs=100,
            batch_size=None,  # Full-batch
            init_from_irt=False,
            early_stopping=True,
            val_fraction=0.1,
            patience=20,
            verbose=True,
        ),
        name="fullbatch_highLR_random",
        display_name="Full-batch, LR=0.01, random",
    ))

    # Mini-batch with higher LR (hybrid approach)
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=64,
            hidden_sizes=[256, 128],
            dropout=0.1,
            learning_rate=0.01,  # Higher LR
            weight_decay=0.01,
            n_epochs=20,
            batch_size=256,
            init_from_irt=False,
            early_stopping=False,
            verbose=True,
        ),
        name="minibatch_highLR_random",
        display_name="Mini-batch, LR=0.01, random",
    ))

    # Mini-batch with larger embedding dim
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=128,  # Larger embedding
            hidden_sizes=[256, 128],
            dropout=0.1,
            learning_rate=0.001,
            weight_decay=0.01,
            n_epochs=10,
            batch_size=256,
            init_from_irt=False,
            early_stopping=False,
            verbose=True,
        ),
        name="minibatch_emb128_random",
        display_name="Mini-batch, EmbDim=128, random",
    ))

    print(f"\n*** Running {len(configs)} configs ***")

    results = {}

    print("\n" + "=" * 85)
    print(f"DARIA-STYLE TEST (starting at {time.strftime('%H:%M:%S')})")
    print("=" * 85)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        config_start = time.time()

        cv_result = run_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            diagnostics_extractor=extract_train_auc,
        )
        config_elapsed = time.time() - config_start

        results[pc.name] = {
            "display_name": pc.display_name,
            "mean_auc": cv_result.mean_auc,
            "std_auc": cv_result.std_auc,
            "fold_aucs": cv_result.fold_aucs,
            "elapsed_seconds": config_elapsed,
        }

        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")
        print(f"   Time: {config_elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 85)
    print("RESULTS SUMMARY")
    print("=" * 85)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_auc"], reverse=True)

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        display_name = r["display_name"][:43]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output_dir = ROOT / "chris_output/experiment_a/mlp_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "daria_style_test.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s")


if __name__ == "__main__":
    main()
