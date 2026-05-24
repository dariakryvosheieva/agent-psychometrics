"""Evaluation pipeline for Experiment New Agents."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from experiment_new_agents.config import ExperimentNewAgentsConfig
from experiment_new_agents.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    evaluate_predictor_cv,
    stable_k_fold_split_agent_pairs,
)
from experiment_new_agents.dataset import (
    load_agent_model_scaffold_map,
    load_dataset_for_agent_fold,
    load_tagged_responses,
)
from experiment_new_agents.difficulty_predictors import (
    ConstantPredictor,
    ModelScaffoldPredictor,
    OraclePredictor,
)
from experiment_new_tasks.bootstrap import bootstrap_seed_mean_differences
from experiment_new_tasks.results import (
    finite_auc_values,
    print_repeated_cv_summary,
    sample_std,
)


@dataclass
class CVPredictorConfig:
    predictor: CVPredictor
    name: str
    display_name: str


def build_cv_predictors() -> List[CVPredictorConfig]:
    return [
        CVPredictorConfig(OraclePredictor(), "oracle", "Oracle (full agent IRT)"),
        CVPredictorConfig(
            ModelScaffoldPredictor(),
            "model_scaffold",
            "Model + Scaffold IRT",
        ),
        CVPredictorConfig(
            ConstantPredictor(),
            "constant_baseline",
            "Empirical task solve rate",
        ),
    ]


def cross_validate_all_predictors(
    config: ExperimentNewAgentsConfig,
    root: Path,
    *,
    dataset: str,
    k: int = 5,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print(f"EXPERIMENT NEW AGENTS: {k}-FOLD CV - {config.display_name}")
    print("=" * 60)

    responses_path = root / config.responses_path
    tagged = load_tagged_responses(responses_path, dataset)
    agent_to_ms_pair = load_agent_model_scaffold_map(responses_path, dataset, tagged)
    agent_keys = sorted(agent_to_ms_pair.keys())

    folds = stable_k_fold_split_agent_pairs(
        agent_keys,
        agent_to_ms_pair,
        k=k,
        seed=config.split_seed,
    )
    print(f"\nTotal scoreable agents: {len(agent_keys)}")
    print(f"Total model/scaffold pairs: {len(set(agent_to_ms_pair.values()))}")
    print(f"Agents per fold (test): ~{len(agent_keys) // k}")

    def load_fold_data(
        train_agents: List[str],
        test_agents: List[str],
        fold_idx: int,
        load_train_irt: bool,
    ):
        return load_dataset_for_agent_fold(
            dataset=dataset,
            responses_path=responses_path,
            train_agents=train_agents,
            test_agents=test_agents,
            fold_idx=fold_idx,
            k_folds=k,
            split_seed=config.split_seed,
            irt_cache_dir=root / config.irt_cache_dir,
            oracle_cache_dir=root / config.oracle_cache_dir,
            irt_model=config.irt_model,
            irt_epochs=config.irt_epochs,
            irt_device=config.irt_device,
            irt_lr=config.irt_lr,
            theta_combine=config.theta_combine,
            load_train_irt=load_train_irt,
        )

    predictor_configs = build_cv_predictors()
    cv_results: Dict[str, CrossValidationResult] = {}
    for i, pc in enumerate(predictor_configs, 1):
        print(f"\n{i}. {pc.display_name}:")
        extractor = diagnostics_extractors.get(pc.name) if diagnostics_extractors else None
        cv_results[pc.name] = evaluate_predictor_cv(
            pc.predictor,
            folds,
            load_fold_data,
            verbose=True,
            diagnostics_extractor=extractor,
        )
        result = cv_results[pc.name]
        if result.mean_auc is not None:
            print(f"   Mean AUC: {result.mean_auc:.4f} +/- {result.std_auc:.4f}")
        else:
            print("   Mean AUC: N/A")

    return {
        "config": config.to_dict(),
        "dataset": dataset,
        "k_folds": k,
        "cv_results": {name: asdict(result) for name, result in cv_results.items()},
    }


def cross_validate_all_predictors_repeated_seeds(
    config: ExperimentNewAgentsConfig,
    root: Path,
    *,
    dataset: str,
    k: int = 5,
    fold_seeds: Optional[List[int]] = None,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
    target_n_fold_seeds: Optional[int] = None,
) -> Dict[str, Any]:
    if fold_seeds is None:
        fold_seeds = list(range(20))
    if not fold_seeds:
        raise ValueError("At least one fold seed is required")

    target = int(target_n_fold_seeds) if target_n_fold_seeds is not None else len(fold_seeds)
    if target < 1:
        raise ValueError("target_n_fold_seeds must be >= 1")

    per_seed_results: List[Dict[str, Any]] = []
    skipped_fold_seeds: List[Dict[str, Any]] = []
    for seed_idx, fold_seed in enumerate(fold_seeds, 1):
        if len(per_seed_results) >= target:
            break
        print(
            f"\nSeed candidate {seed_idx}/{len(fold_seeds)} "
            f"(split_seed={int(fold_seed)}, accepted={len(per_seed_results)}/{target})"
        )
        seeded_config = replace(config, split_seed=int(fold_seed))
        try:
            seed_results = cross_validate_all_predictors(
                seeded_config,
                root,
                dataset=dataset,
                k=k,
                diagnostics_extractors=diagnostics_extractors,
            )
        except RuntimeError as exc:
            skipped_fold_seeds.append(
                {"split_seed": int(fold_seed), "reason": str(exc)}
            )
            print(f"Skipping split_seed={int(fold_seed)}: {exc}")
            continue
        per_seed_results.append({"fold_seed": int(fold_seed), "results": seed_results})

    if not per_seed_results:
        raise RuntimeError(f"No valid fold seeds. Skipped: {skipped_fold_seeds[:10]}")

    first_cv_results = per_seed_results[0]["results"]["cv_results"]
    if "constant_baseline" not in first_cv_results:
        raise ValueError("constant_baseline result is required for paired comparisons")

    cv_results: Dict[str, Dict[str, Any]] = {}
    for method_name in first_cv_results.keys():
        seed_mean_aucs: List[float] = []
        seed_mean_differences: List[float] = []
        seed_fold_aucs: List[List[float]] = []
        for per_seed in per_seed_results:
            fold_seed = int(per_seed["fold_seed"])
            seed_cv_results = per_seed["results"]["cv_results"]
            method_fold_aucs = finite_auc_values(
                seed_cv_results[method_name]["fold_aucs"],
                context=f"method {method_name!r}, seed {fold_seed}",
            )
            baseline_fold_aucs = finite_auc_values(
                seed_cv_results["constant_baseline"]["fold_aucs"],
                context=f"baseline, seed {fold_seed}",
            )
            seed_mean_aucs.append(float(np.mean(method_fold_aucs)))
            seed_mean_differences.append(
                float(np.mean([a - b for a, b in zip(method_fold_aucs, baseline_fold_aucs)]))
            )
            seed_fold_aucs.append(method_fold_aucs)

        bootstrap = bootstrap_seed_mean_differences(
            seed_mean_differences,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        cv_results[method_name] = {
            "mean_auc": float(np.mean(seed_mean_aucs)),
            "std_auc": sample_std(seed_mean_aucs),
            "seed_mean_aucs": seed_mean_aucs,
            "seed_fold_aucs": seed_fold_aucs,
            "seed_mean_differences_vs_baseline": seed_mean_differences,
            "mean_difference_vs_baseline": float(np.mean(seed_mean_differences)),
            "bootstrap_difference_vs_baseline": asdict(bootstrap),
            "k": k,
            "n_fold_seeds": len(per_seed_results),
        }

    print_repeated_cv_summary(
        display_name=config.display_name,
        cv_results=cv_results,
        n_fold_seeds=len(per_seed_results),
        k_folds=k,
    )

    return {
        "config": config.to_dict(),
        "dataset": dataset,
        "k_folds": k,
        "fold_seeds": [int(result["fold_seed"]) for result in per_seed_results],
        "skipped_fold_seeds": skipped_fold_seeds,
        "n_fold_seeds": len(per_seed_results),
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "cv_results": cv_results,
        "per_seed_results": per_seed_results,
    }
