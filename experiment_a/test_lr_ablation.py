"""Quick ablation: test different agent learning rate scales.

Tests whether slowing down agent learning can solve gradient competition
without freezing abilities to IRT values.

Usage:
    python -m experiment_a.test_lr_ablation
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import build_feature_sources
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import MLPPredictor
from experiment_a.shared.baselines import OraclePredictor, ConstantPredictor
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold

ROOT = Path(__file__).parent.parent


def main():
    config = ExperimentAConfig()

    # Resolve paths
    llm_judge_path = ROOT / config.llm_judge_features_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    # Build feature sources (just LLM Judge)
    feature_source_list = build_feature_sources(
        llm_judge_path=llm_judge_path,
        verbose=True,
    )
    source = feature_source_list[0][1]  # LLM Judge source

    # Load task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    # Generate folds
    k_folds = 5
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

    # Test different agent_lr_scale values
    lr_scales = [1.0, 0.1, 0.01, 0.001]

    configs: List[CVPredictorConfig] = []

    for scale in lr_scales:
        predictor = MLPPredictor(
            source,
            freeze_abilities=False,  # Not freezing - testing LR approach
            agent_lr_scale=scale,
            learning_rate=0.01,
            n_epochs=500,
            verbose=True,
        )
        configs.append(
            CVPredictorConfig(
                predictor=predictor,
                name=f"mlp_lr_scale_{scale}",
                display_name=f"MLP (agent_lr_scale={scale})",
            )
        )

    # Add frozen IRT baseline for comparison
    configs.append(
        CVPredictorConfig(
            predictor=MLPPredictor(
                source,
                freeze_abilities=True,
                learning_rate=0.01,
                n_epochs=500,
                verbose=True,
            ),
            name="mlp_frozen_irt",
            display_name="MLP (frozen IRT)",
        )
    )

    # Add baselines
    configs.append(
        CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true b)",
        )
    )
    configs.append(
        CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant",
            display_name="Constant (mean b)",
        )
    )

    # Run CV
    results = {}

    print("\n" + "=" * 80)
    print("LEARNING RATE ABLATION STUDY (LLM Judge)")
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

        # Get train AUC if available
        if hasattr(pc.predictor, 'get_train_auc') and pc.predictor.get_train_auc() is not None:
            results[pc.name]["train_auc"] = pc.predictor.get_train_auc()

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<30} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
    print("-" * 65)

    for name, r in results.items():
        test_auc = r["mean_auc"]
        train_auc = r.get("train_auc", None)
        if train_auc is not None:
            gap = train_auc - test_auc
            print(f"{r['display_name']:<30} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
        else:
            print(f"{r['display_name']:<30} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Save results
    output_dir = ROOT / "chris_output/experiment_a/mlp_lr_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lr_ablation_results.json"

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
