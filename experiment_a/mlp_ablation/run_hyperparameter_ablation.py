"""Hyperparameter ablation for AgentEmbeddingPredictor.

Tests different combinations of:
- Learning rate
- Weight decay
- Dropout
- Hidden layer dimensions
- Init strategy (random vs IRT)

Usage:
    # Run all configs on GPU 0
    python -m experiment_a.mlp_ablation.run_hyperparameter_ablation --gpu 0

    # Run configs split across 2 GPUs (for cluster with 2 GPUs)
    python -m experiment_a.mlp_ablation.run_hyperparameter_ablation --gpu 0 --total_gpus 2 &
    python -m experiment_a.mlp_ablation.run_hyperparameter_ablation --gpu 1 --total_gpus 2 &
    wait

    # Quick test
    python -m experiment_a.mlp_ablation.run_hyperparameter_ablation --quick --gpu 0
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""
    name: str
    display_name: str
    learning_rate: float
    weight_decay: float
    dropout: float
    hidden_sizes: List[int]
    n_epochs: int
    init_from_irt: bool
    init_noise_scale: float = 0.0
    agent_emb_dim: int = 64


def generate_ablation_configs(quick: bool = False) -> List[AblationConfig]:
    """Generate all ablation configurations."""
    configs = []

    if quick:
        # Quick test - just a few key configs
        configs.extend([
            AblationConfig("baseline_random", "Baseline random", 0.001, 0.01, 0.1, [64, 64], 100, False),
            AblationConfig("baseline_irt", "Baseline IRT", 0.001, 0.01, 0.1, [64, 64], 100, True),
            AblationConfig("high_reg_random", "High reg random", 0.001, 0.1, 0.3, [64, 64], 100, False),
        ])
        return configs

    # ============================================================
    # SECTION 1: Systematic single-factor sweeps
    # ============================================================

    # Learning rate sweep (with moderate regularization)
    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            configs.append(AblationConfig(
                f"lr_{lr}_{init_name}".replace(".", "p"),
                f"LR={lr}, {init_name}",
                lr, 0.01, 0.1, [64, 64], 100, init_irt
            ))

    # Weight decay sweep
    for wd in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            configs.append(AblationConfig(
                f"wd_{wd}_{init_name}".replace(".", "p"),
                f"WD={wd}, {init_name}",
                0.001, wd, 0.1, [64, 64], 100, init_irt
            ))

    # Dropout sweep
    for dropout in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            configs.append(AblationConfig(
                f"drop_{dropout}_{init_name}".replace(".", "p"),
                f"Dropout={dropout}, {init_name}",
                0.001, 0.01, dropout, [64, 64], 100, init_irt
            ))

    # Hidden architecture sweep
    hidden_configs = [
        ([32], "32"),
        ([64], "64"),
        ([128], "128"),
        ([256], "256"),
        ([32, 32], "32x32"),
        ([64, 64], "64x64"),
        ([128, 64], "128x64"),
        ([256, 128], "256x128"),
        ([128, 128], "128x128"),
        ([64, 64, 64], "64x64x64"),
        ([128, 64, 32], "128x64x32"),
    ]
    for hidden, hidden_str in hidden_configs:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            configs.append(AblationConfig(
                f"hidden_{hidden_str}_{init_name}",
                f"Hidden={hidden_str}, {init_name}",
                0.001, 0.01, 0.1, hidden, 100, init_irt
            ))

    # Epochs sweep (to understand training dynamics)
    for epochs in [20, 50, 100, 200, 500]:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            configs.append(AblationConfig(
                f"epochs_{epochs}_{init_name}",
                f"Epochs={epochs}, {init_name}",
                0.001, 0.01, 0.1, [64, 64], epochs, init_irt
            ))

    # Agent embedding dimension sweep
    for emb_dim in [8, 16, 32, 64, 128]:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            configs.append(AblationConfig(
                f"embdim_{emb_dim}_{init_name}",
                f"EmbDim={emb_dim}, {init_name}",
                0.001, 0.01, 0.1, [64, 64], 100, init_irt,
                agent_emb_dim=emb_dim
            ))

    # ============================================================
    # SECTION 2: Promising combinations of best single-factor choices
    # Based on hypotheses about what works well:
    # - LR: 0.001 and 0.0005 likely best
    # - WD: 0.01, 0.05, 0.1 likely best range
    # - Dropout: 0.1, 0.2, 0.3 likely best range
    # - Hidden: 64x64, 128x64, 256x128 likely good
    # ============================================================

    # Combine best LRs with best WDs
    best_lrs = [0.0005, 0.001]
    best_wds = [0.01, 0.05, 0.1]
    best_dropouts = [0.1, 0.2, 0.3]

    for lr in best_lrs:
        for wd in best_wds:
            for init_irt in [False, True]:
                init_name = "irt" if init_irt else "random"
                name = f"combo_lr{lr}_wd{wd}_{init_name}".replace(".", "p")
                configs.append(AblationConfig(
                    name, f"LR={lr},WD={wd}, {init_name}",
                    lr, wd, 0.1, [64, 64], 100, init_irt
                ))

    # Combine best LRs with best dropouts
    for lr in best_lrs:
        for dropout in best_dropouts:
            for init_irt in [False, True]:
                init_name = "irt" if init_irt else "random"
                name = f"combo_lr{lr}_drop{dropout}_{init_name}".replace(".", "p")
                configs.append(AblationConfig(
                    name, f"LR={lr},Drop={dropout}, {init_name}",
                    lr, 0.01, dropout, [64, 64], 100, init_irt
                ))

    # Combine best WDs with best dropouts
    for wd in best_wds:
        for dropout in best_dropouts:
            for init_irt in [False, True]:
                init_name = "irt" if init_irt else "random"
                name = f"combo_wd{wd}_drop{dropout}_{init_name}".replace(".", "p")
                configs.append(AblationConfig(
                    name, f"WD={wd},Drop={dropout}, {init_name}",
                    0.001, wd, dropout, [64, 64], 100, init_irt
                ))

    # Triple combinations: best LR + best WD + best dropout
    for lr in best_lrs:
        for wd in best_wds:
            for dropout in best_dropouts:
                for init_irt in [False, True]:
                    init_name = "irt" if init_irt else "random"
                    name = f"triple_lr{lr}_wd{wd}_drop{dropout}_{init_name}".replace(".", "p")
                    configs.append(AblationConfig(
                        name, f"LR={lr},WD={wd},D={dropout}, {init_name}",
                        lr, wd, dropout, [64, 64], 100, init_irt
                    ))

    # Best regularization combos with different architectures
    best_hiddens = [([64, 64], "64x64"), ([128, 64], "128x64"), ([256, 128], "256x128")]
    reg_combos = [(0.05, 0.2), (0.1, 0.2), (0.1, 0.3)]

    for hidden, hidden_str in best_hiddens:
        for wd, dropout in reg_combos:
            for init_irt in [False, True]:
                init_name = "irt" if init_irt else "random"
                name = f"arch_{hidden_str}_wd{wd}_drop{dropout}_{init_name}".replace(".", "p")
                configs.append(AblationConfig(
                    name, f"H={hidden_str},WD={wd},D={dropout}, {init_name}",
                    0.001, wd, dropout, hidden, 100, init_irt
                ))

    # ============================================================
    # SECTION 3: Daria-style configurations
    # ============================================================

    # Exactly match Daria's setup
    configs.append(AblationConfig(
        "daria_exact_random",
        "Daria exact (random)",
        0.001, 0.01, 0.1, [256, 128], 100, False
    ))
    configs.append(AblationConfig(
        "daria_exact_irt",
        "Daria exact (IRT)",
        0.001, 0.01, 0.1, [256, 128], 100, True
    ))

    # Daria-style with best reg combos
    for wd, dropout in reg_combos:
        for init_irt in [False, True]:
            init_name = "irt" if init_irt else "random"
            name = f"daria_wd{wd}_drop{dropout}_{init_name}".replace(".", "p")
            configs.append(AblationConfig(
                name, f"Daria+WD={wd},D={dropout}, {init_name}",
                0.001, wd, dropout, [256, 128], 100, init_irt
            ))

    # ============================================================
    # SECTION 4: IRT noise scale sweep (for IRT init only)
    # ============================================================

    for noise in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]:
        configs.append(AblationConfig(
            f"irt_noise_{noise}".replace(".", "p"),
            f"IRT + noise={noise}",
            0.001, 0.01, 0.1, [64, 64], 100, True,
            init_noise_scale=noise
        ))

    # IRT noise with best reg combos
    for noise in [0.01, 0.1]:
        for wd, dropout in [(0.05, 0.2), (0.1, 0.3)]:
            name = f"irt_noise{noise}_wd{wd}_drop{dropout}".replace(".", "p")
            configs.append(AblationConfig(
                name, f"IRT+noise={noise},WD={wd},D={dropout}",
                0.001, wd, dropout, [64, 64], 100, True,
                init_noise_scale=noise
            ))

    # ============================================================
    # SECTION 5: Extreme configurations to find boundaries
    # ============================================================

    # Very heavy regularization
    configs.append(AblationConfig(
        "extreme_reg_random", "Extreme reg (random)",
        0.001, 0.3, 0.5, [64, 64], 100, False
    ))
    configs.append(AblationConfig(
        "extreme_reg_irt", "Extreme reg (IRT)",
        0.001, 0.3, 0.5, [64, 64], 100, True
    ))

    # Very low learning rate with longer training
    configs.append(AblationConfig(
        "very_low_lr_random", "Very low LR (random)",
        0.00005, 0.01, 0.1, [64, 64], 300, False
    ))
    configs.append(AblationConfig(
        "very_low_lr_irt", "Very low LR (IRT)",
        0.00005, 0.01, 0.1, [64, 64], 300, True
    ))

    # Minimal model (to see if complexity matters)
    configs.append(AblationConfig(
        "minimal_random", "Minimal (random)",
        0.001, 0.01, 0.0, [32], 100, False
    ))
    configs.append(AblationConfig(
        "minimal_irt", "Minimal (IRT)",
        0.001, 0.01, 0.0, [32], 100, True
    ))

    # Large model with heavy regularization
    configs.append(AblationConfig(
        "large_reg_random", "Large+Reg (random)",
        0.001, 0.1, 0.3, [512, 256, 128], 100, False
    ))
    configs.append(AblationConfig(
        "large_reg_irt", "Large+Reg (IRT)",
        0.001, 0.1, 0.3, [512, 256, 128], 100, True
    ))

    return configs


def extract_train_auc(predictor: CVPredictor, fold_idx: int) -> float | None:
    """Extract train AUC from fitted predictor."""
    if hasattr(predictor, 'get_train_auc'):
        return predictor.get_train_auc()
    return None


ROOT = Path(__file__).parent.parent.parent


def main():
    main_start = time.time()
    print(f"Script starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    parser = argparse.ArgumentParser(description="Hyperparameter ablation")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--total_gpus", type=int, default=1, help="Total GPUs (for splitting work)")
    args = parser.parse_args()

    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("CUDA not available, using CPU")

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

    # Generate ablation configs
    all_ablation_configs = generate_ablation_configs(quick=args.quick)

    # Split configs across GPUs
    my_configs = [c for i, c in enumerate(all_ablation_configs) if i % args.total_gpus == args.gpu]

    print(f"\n*** GPU {args.gpu}: Running {len(my_configs)}/{len(all_ablation_configs)} configs ***")

    # Build predictor configs
    predictor_configs: List[CVPredictorConfig] = []

    # Add baselines (only on GPU 0)
    if args.gpu == 0:
        predictor_configs.append(CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true beta)",
        ))
        predictor_configs.append(CVPredictorConfig(
            predictor=DifficultyPredictorAdapter(
                FeatureBasedPredictor(embedding_source, alphas=config.ridge_alphas)
            ),
            name="ridge",
            display_name="Ridge (Embedding)",
        ))

    # Add ablation configs
    for ac in my_configs:
        predictor_configs.append(CVPredictorConfig(
            predictor=AgentEmbeddingPredictor(
                embedding_source,
                agent_emb_dim=ac.agent_emb_dim,
                hidden_sizes=ac.hidden_sizes,
                dropout=ac.dropout,
                learning_rate=ac.learning_rate,
                weight_decay=ac.weight_decay,
                n_epochs=ac.n_epochs,
                init_from_irt=ac.init_from_irt,
                init_noise_scale=ac.init_noise_scale,
                early_stopping=True,
                val_fraction=0.1,
                patience=20,
                verbose=True,
            ),
            name=ac.name,
            display_name=ac.display_name,
        ))

    print(f"\n*** Total configs on GPU {args.gpu}: {len(predictor_configs)} ***")

    # Run CV
    results = {}

    print("\n" + "=" * 85)
    print(f"HYPERPARAMETER ABLATION - GPU {args.gpu} (starting CV at {time.strftime('%H:%M:%S')})")
    print("=" * 85)

    for i, pc in enumerate(predictor_configs, 1):
        print(f"\n[{i}/{len(predictor_configs)}] {pc.display_name}")
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
        print(f"   Config time: {config_elapsed:.1f}s")

    # Print summary
    print("\n" + "=" * 85)
    print(f"ABLATION RESULTS - GPU {args.gpu}")
    print("=" * 85)
    print(f"\n{'Method':<50} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 85)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["mean_auc"],
        reverse=True
    )

    for name, r in sorted_results:
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc")
        display_name = r["display_name"][:48]

        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{display_name:<50} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{display_name:<50} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

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
    output_path = output_dir / f"hyperparameter_ablation_gpu{args.gpu}.json"

    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total time: {time.time() - main_start:.1f}s ({(time.time() - main_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
