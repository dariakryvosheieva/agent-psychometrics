"""Run MLP ablation study to debug overfitting/underfitting issues.

Tests 4 conditions per feature source:
- A) Baseline: learned abilities, weak regularization (current broken)
- B) Frozen IRT: IRT abilities frozen, weak regularization
- C) Strong reg: learned abilities, strong regularization
- D) Both fixes: IRT abilities frozen, strong regularization

Usage:
    python -m experiment_a.run_mlp_ablation
    python -m experiment_a.run_mlp_ablation --source embedding  # Single source
    python -m experiment_a.run_mlp_ablation --source llm_judge
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from experiment_ab_shared.feature_source import (
    build_feature_sources,
    GroupedFeatureSource,
    RegularizedFeatureSource,
)
from experiment_ab_shared.dataset import _load_binary_responses
from experiment_a.shared.cross_validation import k_fold_split_tasks, run_cv
from experiment_a.shared.pipeline import CVPredictorConfig
from experiment_a.shared.mlp_predictor import MLPPredictor
from experiment_a.shared.baselines import OraclePredictor, ConstantPredictor
from experiment_a.swebench.config import ExperimentAConfig
from experiment_ab_shared import load_dataset_for_fold

ROOT = Path(__file__).parent.parent


def build_ablation_configs(
    source_by_name: Dict[str, Any],
    sources_to_test: List[str],
) -> List[CVPredictorConfig]:
    """Build MLP ablation configurations."""
    configs: List[CVPredictorConfig] = []

    # Ablation conditions: (freeze_abilities, feature_weight_decay, suffix)
    ablation_conditions = [
        (False, 0.01, "baseline"),      # A) Current broken behavior
        (True, 0.01, "frozen_irt"),     # B) Frozen IRT abilities
        (False, 1.0, "strong_reg"),     # C) Strong regularization
        (True, 1.0, "both_fixes"),      # D) Both fixes
    ]

    for source_name in sources_to_test:
        if source_name not in source_by_name:
            print(f"Warning: Source '{source_name}' not found, skipping")
            continue

        source = source_by_name[source_name]
        for freeze, feat_wd, suffix in ablation_conditions:
            predictor = MLPPredictor(
                source,
                freeze_abilities=freeze,
                feature_weight_decay=feat_wd,
                verbose=True,
            )
            short_name = source_name.replace(" ", "_").lower()
            configs.append(
                CVPredictorConfig(
                    predictor=predictor,
                    name=f"mlp_{short_name}_{suffix}",
                    display_name=f"MLP {source_name} ({suffix.replace('_', ' ')})",
                )
            )

    return configs


def main():
    parser = argparse.ArgumentParser(description="Run MLP ablation study")
    parser.add_argument(
        "--source",
        type=str,
        choices=["embedding", "llm_judge", "grouped", "all"],
        default="all",
        help="Which feature source to test (default: all)",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chris_output/experiment_a/mlp_ablation",
        help="Output directory",
    )
    args = parser.parse_args()

    config = ExperimentAConfig()

    # Resolve paths
    embeddings_path = ROOT / config.embeddings_path
    llm_judge_path = ROOT / config.llm_judge_features_path
    abilities_path = ROOT / config.abilities_path
    items_path = ROOT / config.items_path
    responses_path = ROOT / config.responses_path

    # Build feature sources
    feature_source_list = build_feature_sources(
        embeddings_path=embeddings_path,
        llm_judge_path=llm_judge_path,
        verbose=True,
    )
    source_by_name = {name: source for name, source in feature_source_list}

    # Add grouped source if we have multiple sources
    if len(feature_source_list) >= 2:
        feature_sources = [source for _, source in feature_source_list]
        grouped_source = GroupedFeatureSource([
            RegularizedFeatureSource(src) for src in feature_sources
        ])
        source_by_name["Grouped"] = grouped_source

    # Determine which sources to test
    if args.source == "all":
        sources_to_test = list(source_by_name.keys())
    elif args.source == "embedding":
        sources_to_test = ["Embedding"]
    elif args.source == "llm_judge":
        sources_to_test = ["LLM Judge"]
    elif args.source == "grouped":
        sources_to_test = ["Grouped"]
    else:
        sources_to_test = [args.source]

    print(f"\nTesting sources: {sources_to_test}")

    # Build ablation configs
    ablation_configs = build_ablation_configs(source_by_name, sources_to_test)

    # Add baselines for comparison
    ablation_configs.append(
        CVPredictorConfig(
            predictor=OraclePredictor(),
            name="oracle",
            display_name="Oracle (true b)",
        )
    )
    ablation_configs.append(
        CVPredictorConfig(
            predictor=ConstantPredictor(),
            name="constant",
            display_name="Constant (mean b)",
        )
    )

    # Load task IDs
    full_items = pd.read_csv(items_path, index_col=0)
    all_task_ids = list(full_items.index)

    print(f"\nTotal tasks: {len(all_task_ids)}")
    print(f"K-folds: {args.k_folds}")

    # Generate folds
    folds = k_fold_split_tasks(all_task_ids, k=args.k_folds, seed=config.split_seed)

    # Create fold data loader
    def load_fold_data(train_tasks, test_tasks, fold_idx):
        return load_dataset_for_fold(
            abilities_path=abilities_path,
            items_path=items_path,
            responses_path=responses_path,
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            fold_idx=fold_idx,
            k_folds=args.k_folds,
            split_seed=config.split_seed,
            is_binomial=False,
            irt_cache_dir=Path("chris_output/experiment_a/irt_splits"),
        )

    # Run CV for each config
    results = {}

    print("\n" + "=" * 80)
    print("RUNNING MLP ABLATION STUDY")
    print("=" * 80)

    for i, pc in enumerate(ablation_configs, 1):
        print(f"\n[{i}/{len(ablation_configs)}] {pc.display_name}")
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

        # Get train AUC if available (for MLP predictors)
        if hasattr(pc.predictor, 'get_train_auc') and pc.predictor.get_train_auc() is not None:
            results[pc.name]["train_auc"] = pc.predictor.get_train_auc()

        print(f"   Mean AUC: {cv_result.mean_auc:.4f} +/- {cv_result.std_auc:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    # Group by source
    for source_name in sources_to_test:
        short_name = source_name.replace(" ", "_").lower()
        print(f"\n{source_name}:")
        print(f"{'Condition':<20} {'Test AUC':>10} {'Train AUC':>12} {'Gap':>10}")
        print("-" * 55)

        for suffix in ["baseline", "frozen_irt", "strong_reg", "both_fixes"]:
            key = f"mlp_{short_name}_{suffix}"
            if key in results:
                r = results[key]
                test_auc = r["mean_auc"]
                train_auc = r.get("train_auc", None)
                if train_auc is not None:
                    gap = train_auc - test_auc
                    print(f"{suffix:<20} {test_auc:>10.4f} {train_auc:>12.4f} {gap:>+10.4f}")
                else:
                    print(f"{suffix:<20} {test_auc:>10.4f} {'N/A':>12} {'N/A':>10}")

    # Print baselines
    print(f"\nBaselines:")
    print(f"{'Method':<20} {'Test AUC':>10}")
    print("-" * 32)
    for key in ["oracle", "constant"]:
        if key in results:
            print(f"{results[key]['display_name']:<20} {results[key]['mean_auc']:>10.4f}")

    # Save results
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_results.json"

    # Convert numpy types for JSON serialization
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
