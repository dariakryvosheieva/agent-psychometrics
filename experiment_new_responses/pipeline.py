"""Evaluation pipeline for Experiment New Responses."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from experiment_new_responses.config import ExperimentNewResponsesConfig
from experiment_new_responses.cross_validation import (
    CVPredictor,
    CrossValidationResult,
    evaluate_predictor_cv,
    stable_k_fold_split_observations,
)
from experiment_new_responses.dataset import (
    all_observation_keys_from_responses,
    all_item_ids_from_responses,
    load_agent_model_scaffold_map,
    load_dataset_for_observation_fold,
    load_many_agent_model_scaffold_maps,
    make_observation_key,
    parse_observation_key,
    load_tagged_responses,
)
from experiment_new_responses.difficulty_predictors import (
    ModelScaffoldPredictor,
    OraclePredictor,
    StandardIrtPredictor,
)
from experiment_new_responses.train_irt_split import (
    get_or_train_model_scaffold_observation_split_irt,
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
        CVPredictorConfig(OraclePredictor(), "oracle", "Oracle (full IRT)"),
        CVPredictorConfig(
            ModelScaffoldPredictor(),
            "model_scaffold",
            "Model + Scaffold IRT",
        ),
        CVPredictorConfig(
            StandardIrtPredictor(),
            "standard_irt_baseline",
            "Standard IRT baseline",
        ),
    ]


def cross_validate_all_predictors(
    config: ExperimentNewResponsesConfig,
    root: Path,
    *,
    dataset: str,
    k: int = 5,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Any]:
    print("=" * 60)
    print(f"EXPERIMENT NEW RESPONSES: {k}-FOLD CV - {config.display_name}")
    print("=" * 60)

    responses_path = root / config.responses_path
    tagged = load_tagged_responses(responses_path, dataset)
    agent_to_ms_pair = load_agent_model_scaffold_map(responses_path, dataset, tagged)
    observation_keys = [
        key for key in all_observation_keys_from_responses(tagged)
        if "::".join(key.split("::", 2)[:2]) in agent_to_ms_pair
    ]
    if not observation_keys:
        raise RuntimeError("No scoreable observations have model/scaffold mappings")

    folds = stable_k_fold_split_observations(
        observation_keys,
        k=k,
        seed=config.split_seed,
    )
    print(f"\nTotal observations: {len(observation_keys)}")
    print(f"Observations per fold (test): ~{len(observation_keys) // k}")

    def load_fold_data(
        train_observations: List[str],
        test_observations: List[str],
        fold_idx: int,
        load_train_irt: bool,
    ):
        return load_dataset_for_observation_fold(
            dataset=dataset,
            responses_path=responses_path,
            train_observations=train_observations,
            test_observations=test_observations,
            fold_idx=fold_idx,
            k_folds=k,
            split_seed=config.split_seed,
            irt_cache_dir=root / config.irt_cache_dir,
            oracle_cache_dir=root / config.oracle_cache_dir,
            irt_epochs=config.irt_epochs,
            irt_device=config.irt_device,
            irt_lr=config.irt_lr,
            irt_model=config.irt_model,
            load_train_irt=load_train_irt,
            theta_combine=config.theta_combine,
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
    config: ExperimentNewResponsesConfig,
    root: Path,
    *,
    dataset: str,
    k: int = 5,
    fold_seeds: Optional[List[int]] = None,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    diagnostics_extractors: Optional[Dict[str, Callable]] = None,
    target_n_fold_seeds: Optional[int] = None,
    baseline_method: str = "standard_irt_baseline",
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
    if baseline_method not in first_cv_results:
        raise ValueError(f"{baseline_method} result is required for paired comparisons")

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
                seed_cv_results[baseline_method]["fold_aucs"],
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


def cross_validate_four_benchmark_model_scaffold_repeated_seeds(
    *,
    dataset_to_responses_path: Dict[str, Path],
    output_dir: Path,
    k: int = 5,
    fold_seeds: Optional[List[int]] = None,
    target_n_fold_seeds: Optional[int] = None,
    irt_epochs: int = 2000,
    irt_device: str = "cuda",
    irt_lr: float = 0.01,
    irt_model: str = "1d_1pl",
    theta_combine: str = "sum",
) -> Dict[str, Any]:
    """Run all-benchmark observation CV for model+scaffold IRT only."""

    if fold_seeds is None:
        fold_seeds = list(range(20))
    if not fold_seeds:
        raise ValueError("At least one fold seed is required")
    target = int(target_n_fold_seeds) if target_n_fold_seeds is not None else len(fold_seeds)
    if target < 1:
        raise ValueError("target_n_fold_seeds must be >= 1")

    tagged_by_dataset = {
        dataset: load_tagged_responses(path, dataset)
        for dataset, path in dataset_to_responses_path.items()
    }
    all_responses_tagged = [
        row for dataset in dataset_to_responses_path.keys()
        for row in tagged_by_dataset[dataset]
    ]
    agent_to_ms_pair = load_many_agent_model_scaffold_maps(
        dataset_to_responses_path,
        tagged_by_dataset,
    )
    all_items = all_item_ids_from_responses(all_responses_tagged)
    all_observations = [
        key for key in all_observation_keys_from_responses(all_responses_tagged)
        if "::".join(key.split("::", 2)[:2]) in agent_to_ms_pair
    ]
    if not all_observations:
        raise RuntimeError("No scoreable observations have model/scaffold mappings")

    response_lookup: Dict[str, int] = {}
    for benchmark, subject_id, responses in all_responses_tagged:
        for task_id, actual in responses.items():
            response_lookup[make_observation_key(benchmark, subject_id, str(task_id))] = int(actual)

    per_seed_results: List[Dict[str, Any]] = []
    skipped_fold_seeds: List[Dict[str, Any]] = []
    for seed_idx, fold_seed in enumerate(fold_seeds, 1):
        if len(per_seed_results) >= target:
            break
        print(
            f"\nFour-benchmark seed candidate {seed_idx}/{len(fold_seeds)} "
            f"(split_seed={int(fold_seed)}, accepted={len(per_seed_results)}/{target})"
        )
        try:
            folds = stable_k_fold_split_observations(
                all_observations,
                k=k,
                seed=int(fold_seed),
            )
            fold_aucs: List[Optional[float]] = []
            fold_n_obs: List[int] = []
            for fold_idx, (train_observations, test_observations) in enumerate(folds):
                train_model, train_scaffold, train_items = (
                    get_or_train_model_scaffold_observation_split_irt(
                        all_responses_tagged=all_responses_tagged,
                        agent_to_ms_pair=agent_to_ms_pair,
                        train_observations=set(train_observations),
                        all_item_ids=all_items,
                        output_base=output_dir / "irt_splits" / "model_scaffold",
                        split_seed=int(fold_seed),
                        fold_idx=fold_idx,
                        k_folds=k,
                        irt_model=irt_model,
                        epochs=irt_epochs,
                        device=irt_device,
                        lr=irt_lr,
                    )
                )
                y_true: List[int] = []
                y_scores: List[float] = []
                for observation_key in test_observations:
                    benchmark, subject_id, task_id = parse_observation_key(observation_key)
                    agent_key = f"{benchmark}::{subject_id}"
                    if observation_key not in response_lookup:
                        raise ValueError(
                            f"Held-out observation is missing from responses: {observation_key!r}"
                        )
                    if agent_key not in agent_to_ms_pair:
                        raise ValueError(f"Agent {agent_key!r} has no model/scaffold mapping")
                    model, scaffold = agent_to_ms_pair[agent_key]
                    if model not in train_model.index:
                        raise ValueError(f"Model {model!r} was not observed in training")
                    if scaffold not in train_scaffold.index:
                        raise ValueError(f"Scaffold {scaffold!r} was not observed in training")
                    if task_id not in train_items.index:
                        raise ValueError(f"Task {task_id!r} has no train-fold IRT difficulty")
                    from experiment_new_responses.difficulty_predictors import combine_theta

                    theta = combine_theta(
                        train_model.loc[model, "theta"],
                        train_scaffold.loc[scaffold, "theta"],
                        combine=theta_combine,
                    )
                    from experiment_new_responses.cross_validation import probability_from_theta

                    y_true.append(int(response_lookup[observation_key]))
                    y_scores.append(
                        probability_from_theta(theta, train_items.loc[task_id, "b"])
                    )
                auc = None
                if len(y_true) >= 2 and len(set(y_true)) >= 2:
                    from sklearn.metrics import roc_auc_score

                    auc = float(roc_auc_score(y_true, y_scores))
                fold_aucs.append(auc)
                fold_n_obs.append(len(y_true))
                auc_text = f"{auc:.4f}" if auc is not None else "N/A"
                print(f"      Fold {fold_idx + 1}: AUC = {auc_text}")
        except RuntimeError as exc:
            skipped_fold_seeds.append({"split_seed": int(fold_seed), "reason": str(exc)})
            print(f"Skipping split_seed={int(fold_seed)}: {exc}")
            continue

        valid_fold_aucs = finite_auc_values(
            fold_aucs,
            context=f"four-benchmark model_scaffold, seed {int(fold_seed)}",
        )
        per_seed_results.append(
            {
                "fold_seed": int(fold_seed),
                "fold_aucs": valid_fold_aucs,
                "fold_n_obs": fold_n_obs,
                "mean_auc": float(np.mean(valid_fold_aucs)),
            }
        )

    if not per_seed_results:
        raise RuntimeError(f"No valid fold seeds. Skipped: {skipped_fold_seeds[:10]}")

    seed_mean_aucs = [float(result["mean_auc"]) for result in per_seed_results]
    seed_fold_aucs = [result["fold_aucs"] for result in per_seed_results]
    cv_results = {
        "model_scaffold": {
            "mean_auc": float(np.mean(seed_mean_aucs)),
            "std_auc": sample_std(seed_mean_aucs),
            "seed_mean_aucs": seed_mean_aucs,
            "seed_fold_aucs": seed_fold_aucs,
            "k": k,
            "n_fold_seeds": len(per_seed_results),
        }
    }
    print(
        "\nFour-benchmark model+scaffold summary: "
        f"mean={cv_results['model_scaffold']['mean_auc']:.4f} "
        f"sd={cv_results['model_scaffold']['std_auc']:.4f} "
        f"over {len(per_seed_results)} seeds x {k} folds"
    )
    return {
        "config": {
            "responses_paths": {k: str(v) for k, v in dataset_to_responses_path.items()},
            "output_dir": str(output_dir),
            "irt_epochs": int(irt_epochs),
            "irt_device": str(irt_device),
            "irt_lr": float(irt_lr),
            "irt_model": str(irt_model),
            "theta_combine": str(theta_combine),
        },
        "dataset": "four_benchmark",
        "k_folds": k,
        "fold_seeds": [int(result["fold_seed"]) for result in per_seed_results],
        "skipped_fold_seeds": skipped_fold_seeds,
        "n_fold_seeds": len(per_seed_results),
        "n_bootstrap": 0,
        "cv_results": cv_results,
        "per_seed_results": per_seed_results,
    }
