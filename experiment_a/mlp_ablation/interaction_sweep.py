"""Architecture sweep Part 5: New interaction architectures.

Tests alternative ways to combine agent and task information:
1. Two-Tower: Separate agent/task encoders, dot product for compatibility
2. Bilinear: Agent-task bilinear interaction matrix
3. NCF: Neural Collaborative Filtering (GMF + MLP)
4. Multiplicative: Element-wise product instead of concatenation
5. Agent Embedding: Learned low-dim agent embeddings instead of one-hot

Hypothesis: These architectures explicitly model agent-task interactions
differently than simple concatenation, potentially capturing patterns
that help beat Ridge regression.

Usage:
    python -m experiment_a.mlp_ablation.interaction_sweep
    python -m experiment_a.mlp_ablation.interaction_sweep --quick
    python -m experiment_a.mlp_ablation.interaction_sweep --part 1  # Two-Tower, Bilinear, Multiplicative
    python -m experiment_a.mlp_ablation.interaction_sweep --part 2  # NCF, Agent Embedding
    sbatch experiment_a/mlp_ablation/slurm_interaction_sweep.sh
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
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv, CVPredictor
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import (
    InteractionPredictor,
    AgentEmbeddingPredictor,
    FullMLPPredictor,
)
from experiment_a.shared.baselines import (
    OraclePredictor,
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
    main_start = time.time()
    print(f"Script starting at: {time.strftime('%H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Interaction architecture sweep")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--part", type=int, choices=[1, 2], help="Run specific part (1=Two-Tower/Bilinear/Multiplicative, 2=NCF/AgentEmb)")
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    items_path = ROOT / config.items_path

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    print(f"Loading embeddings at: {time.strftime('%H:%M:%S')}")
    embedding_source = EmbeddingFeatureSource(embeddings_path)
    print(f"Loaded embeddings: {embedding_source.feature_dim} dimensions (took {time.time() - main_start:.1f}s)")

    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    folds = k_fold_split_tasks(all_task_ids, k=args.k_folds, seed=config.split_seed)

    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            abilities_path=ROOT / config.abilities_path,
            items_path=ROOT / config.items_path,
            responses_path=ROOT / config.responses_path,
        )

    # Training parameters (use best from previous sweeps)
    learning_rate = 0.01
    weight_decay = 0.2
    n_epochs = 1000

    configs: List[CVPredictorConfig] = []

    # === Baselines ===
    configs.append(CVPredictorConfig(
        predictor=OraclePredictor(),
        name="oracle",
        display_name="Oracle (true β)",
    ))
    configs.append(CVPredictorConfig(
        predictor=DifficultyPredictorAdapter(
            FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
        ),
        name="ridge",
        display_name="Ridge (Embedding)",
    ))

    # Reference: Best MLP from previous sweep
    configs.append(CVPredictorConfig(
        predictor=FullMLPPredictor(
            embedding_source,
            hidden_size=64,
            dropout=0.0,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            init_from_irt=True,
            early_stopping=True,
            val_fraction=0.1,
            patience=30,
            verbose=True,
        ),
        name="fullmlp_64",
        display_name="FullMLP (h=64, baseline)",
    ))

    # === New Interaction Architectures ===

    if args.quick:
        emb_dims = [32]
        hidden_dims = [64]
    else:
        emb_dims = [16, 32, 64]
        hidden_dims = [32, 64, 128]

    # 1. Two-Tower: Dot product compatibility
    for emb_dim in emb_dims:
        for hidden_dim in hidden_dims:
            configs.append(CVPredictorConfig(
                predictor=InteractionPredictor(
                    embedding_source,
                    model_type="two_tower",
                    emb_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    dropout=0.0,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"two_tower_e{emb_dim}_h{hidden_dim}",
                display_name=f"TwoTower (emb={emb_dim}, h={hidden_dim})",
            ))

    # 2. Bilinear: Explicit agent-task interaction matrix
    for emb_dim in emb_dims:
        configs.append(CVPredictorConfig(
            predictor=InteractionPredictor(
                embedding_source,
                model_type="bilinear",
                emb_dim=emb_dim,
                hidden_dim=64,  # Not used for bilinear
                dropout=0.0,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_epochs=n_epochs,
                init_from_irt=True,
                early_stopping=True,
                val_fraction=0.1,
                patience=30,
                verbose=True,
            ),
            name=f"bilinear_e{emb_dim}",
            display_name=f"Bilinear (emb={emb_dim})",
        ))

    # 3. NCF: Neural Collaborative Filtering
    for emb_dim in emb_dims:
        for hidden_dim in hidden_dims:
            configs.append(CVPredictorConfig(
                predictor=InteractionPredictor(
                    embedding_source,
                    model_type="ncf",
                    emb_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    dropout=0.0,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"ncf_e{emb_dim}_h{hidden_dim}",
                display_name=f"NCF (emb={emb_dim}, h={hidden_dim})",
            ))

    # 4. Multiplicative: Element-wise product
    for hidden_dim in hidden_dims:
        configs.append(CVPredictorConfig(
            predictor=InteractionPredictor(
                embedding_source,
                model_type="multiplicative",
                emb_dim=32,  # Not used
                hidden_dim=hidden_dim,
                dropout=0.0,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_epochs=n_epochs,
                init_from_irt=True,
                early_stopping=True,
                val_fraction=0.1,
                patience=30,
                verbose=True,
            ),
            name=f"multiplicative_h{hidden_dim}",
            display_name=f"Multiplicative (h={hidden_dim})",
        ))

    # 5. Agent Embedding: Learned low-dim agent embeddings
    for agent_emb_dim in [16, 32, 64]:
        for hidden_sizes in [[64, 32], [128, 64], [64, 64]]:
            h_str = "-".join(map(str, hidden_sizes))
            configs.append(CVPredictorConfig(
                predictor=AgentEmbeddingPredictor(
                    embedding_source,
                    agent_emb_dim=agent_emb_dim,
                    hidden_sizes=hidden_sizes,
                    dropout=0.0,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    n_epochs=n_epochs,
                    init_from_irt=True,
                    early_stopping=True,
                    val_fraction=0.1,
                    patience=30,
                    verbose=True,
                ),
                name=f"agent_emb_{agent_emb_dim}_h{h_str}",
                display_name=f"AgentEmb (d={agent_emb_dim}, h={h_str})",
            ))

    # Filter by part if specified (for parallel SLURM execution)
    if args.part is not None:
        baseline_names = ["oracle", "ridge", "fullmlp_64"]
        if args.part == 1:
            # Part 1: Baselines + Two-Tower + Bilinear + Multiplicative
            part_prefixes = ["two_tower", "bilinear", "multiplicative"]
            configs = [c for c in configs if c.name in baseline_names or any(c.name.startswith(p) for p in part_prefixes)]
            print(f"Running Part 1: Baselines, Two-Tower, Bilinear, Multiplicative")
        elif args.part == 2:
            # Part 2: NCF + Agent Embedding
            part_prefixes = ["ncf", "agent_emb"]
            configs = [c for c in configs if any(c.name.startswith(p) for p in part_prefixes)]
            print(f"Running Part 2: NCF, Agent Embedding")

    if args.quick:
        # Filter to just a few representative configs
        quick_names = ["oracle", "ridge", "fullmlp_64",
                       "two_tower_e32_h64", "bilinear_e32",
                       "ncf_e32_h64", "multiplicative_h64",
                       "agent_emb_32_h64-32"]
        configs = [c for c in configs if c.name in quick_names]

    print(f"\n*** Running {len(configs)} configs ***")

    # Run CV
    results = {}
    cv_start = time.time()

    print("\n" + "=" * 85)
    print(f"INTERACTION ARCHITECTURE SWEEP (starting CV at {time.strftime('%H:%M:%S')})")
    print(f"Fixed: weight_decay={weight_decay}, init_from_irt=True, early_stopping=True")
    print(f"Configs to run: {len(configs)}")
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
        }

        if cv_result.fold_diagnostics:
            valid_train_aucs = [t for t in cv_result.fold_diagnostics if t is not None]
            if valid_train_aucs:
                results[pc.name]["train_auc"] = float(np.mean(valid_train_aucs))

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} ± {cv_result.std_auc:.4f}")
        print(f"   Config time: {config_elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 85)
    print("INTERACTION ARCHITECTURE SWEEP RESULTS")
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
        display_name = r["display_name"]

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

        # Find best interaction model (exclude baselines)
        interaction_results = [(n, r) for n, r in results.items()
                               if n not in ["oracle", "ridge", "fullmlp_64"]]
        if interaction_results:
            best_name, best_r = max(interaction_results, key=lambda x: x[1]["mean_auc"])
            delta = best_r["mean_auc"] - ridge_auc
            print(f"  Best interaction: {best_r['display_name']}: {best_r['mean_auc']:.4f} ({delta:+.4f} vs Ridge)")

    if "fullmlp_64" in results:
        print(f"  FullMLP baseline: {results['fullmlp_64']['mean_auc']:.4f}")

    if "oracle" in results:
        print(f"  Oracle upper bound: {results['oracle']['mean_auc']:.4f}")

    # Group by architecture type
    print(f"\n{'-' * 85}")
    print("BY ARCHITECTURE TYPE:")

    for arch_type in ["two_tower", "bilinear", "ncf", "multiplicative", "agent_emb"]:
        arch_results = [(n, r) for n, r in results.items() if n.startswith(arch_type)]
        if arch_results:
            best_name, best_r = max(arch_results, key=lambda x: x[1]["mean_auc"])
            print(f"  Best {arch_type}: {best_r['display_name']}: {best_r['mean_auc']:.4f}")

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

    # Use part-specific filename if running a specific part
    if args.part is not None:
        output_path = output_dir / f"interaction_sweep_part{args.part}.json"
    else:
        output_path = output_dir / "interaction_sweep.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s")


if __name__ == "__main__":
    main()
