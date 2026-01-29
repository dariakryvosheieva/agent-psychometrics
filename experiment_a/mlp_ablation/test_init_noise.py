"""Test whether adding noise to IRT initialization breaks symmetry and improves learning.

The hypothesis is that broadcasting IRT ability to all 64 embedding dims creates
symmetry that the optimizer doesn't break. Adding noise should allow the model
to learn diverse agent representations.

Usage:
    python -m experiment_a.mlp_ablation.test_init_noise
    python -m experiment_a.mlp_ablation.test_init_noise --quick
    sbatch experiment_a/mlp_ablation/slurm_test_init_noise.sh
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from experiment_ab_shared.feature_source import EmbeddingFeatureSource
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv, CVPredictor
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import AgentEmbeddingPredictor
from experiment_a.shared.baselines import OraclePredictor, DifficultyPredictorAdapter
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold


def extract_train_auc(predictor: CVPredictor, fold_idx: int) -> float | None:
    """Extract train AUC from fitted predictor for diagnostics."""
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None


def extract_embedding_stats(predictor: CVPredictor, fold_idx: int) -> dict | None:
    """Extract embedding statistics to check for symmetry breaking."""
    if not hasattr(predictor, '_model') or predictor._model is None:
        return None
    if not hasattr(predictor._model, 'agent_embedding'):
        return None

    with torch.no_grad():
        emb = predictor._model.agent_embedding.weight.cpu().numpy()

    # Compute stats
    within_agent_std = emb.std(axis=1).mean()  # Avg std across dims for each agent
    between_dim_std = emb.std(axis=0).mean()  # Avg std across agents for each dim

    return {
        "within_agent_std": float(within_agent_std),
        "between_dim_std": float(between_dim_std),
        "embedding_shape": emb.shape,
    }


ROOT = Path(__file__).parent.parent.parent


def main():
    main_start = time.time()
    print(f"Script starting at: {time.strftime('%H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Test IRT init noise")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    items_path = ROOT / config.items_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"Loading embeddings at: {time.strftime('%H:%M:%S')}")
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions")

    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    folds = k_fold_split_tasks(all_task_ids, k=args.k_folds, seed=config.split_seed)

    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=ROOT / config.abilities_path,
            items_path=ROOT / config.items_path,
            responses_path=ROOT / config.responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=args.k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # Training parameters
    learning_rate = 0.01
    weight_decay = 0.2
    n_epochs = 1000

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

    # Test configurations
    if args.quick:
        noise_scales = [0.0, 0.1]
    else:
        noise_scales = [0.0, 0.01, 0.1, 0.5, 1.0]

    # 1. Random init (no IRT)
    configs.append(CVPredictorConfig(
        predictor=AgentEmbeddingPredictor(
            embedding_source,
            agent_emb_dim=64,
            hidden_sizes=[64, 64],
            dropout=0.0,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            init_from_irt=False,  # Random init
            init_noise_scale=0.0,
            early_stopping=True,
            val_fraction=0.1,
            patience=30,
            verbose=True,
        ),
        name="random_init",
        display_name="AgentEmb (random init)",
    ))

    # 2. IRT init with various noise scales
    for noise_scale in noise_scales:
        name = f"irt_noise_{noise_scale}".replace(".", "p")
        display = f"AgentEmb (IRT + noise σ={noise_scale})"
        if noise_scale == 0:
            display = "AgentEmb (IRT broadcast, no noise)"

        configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=64,
                hidden_sizes=[64, 64],
                dropout=0.0,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_epochs=n_epochs,
                init_from_irt=True,
                init_noise_scale=noise_scale,
                early_stopping=True,
                val_fraction=0.1,
                patience=30,
                verbose=True,
            ),
            name=name,
            display_name=display,
        ))

    print(f"\n*** Running {len(configs)} configs ***")

    # Run CV
    results = {}

    print("\n" + "=" * 85)
    print(f"INIT NOISE TEST (starting CV at {time.strftime('%H:%M:%S')})")
    print("=" * 85)

    for i, pc in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {pc.display_name}")
        print("-" * 60)

        config_start = time.time()

        # Custom CV to also extract embedding stats
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
        }

        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")
        print(f"   Config time: {config_elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 85)
    print("INIT NOISE TEST RESULTS")
    print("=" * 85)
    print(f"\n{'Method':<45} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["mean_auc"],
        reverse=True
    )

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        display_name = r["display_name"][:43]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<45} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<45} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Key comparisons
    print(f"\n{'-' * 85}")
    print("KEY COMPARISONS:")

    if "ridge" in results:
        ridge_auc = results["ridge"]["mean_auc"]
        print(f"  Ridge baseline: {ridge_auc:.4f}")

    if "irt_noise_0" in results:
        print(f"  IRT broadcast (no noise): {results['irt_noise_0']['mean_auc']:.4f}")

    if "random_init" in results:
        print(f"  Random init: {results['random_init']['mean_auc']:.4f}")

    # Find best noise scale
    noise_results = [(n, r) for n, r in results.items() if n.startswith("irt_noise")]
    if noise_results:
        best_name, best_r = max(noise_results, key=lambda x: x[1]["mean_auc"])
        print(f"  Best noise config: {best_r['display_name']}: {best_r['mean_auc']:.4f}")

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

    output_dir = ROOT / "chris_output/experiment_a/mlp_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "init_noise_test.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s")


if __name__ == "__main__":
    main()
