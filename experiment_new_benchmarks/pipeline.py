"""Pipeline for held-out benchmark generalization."""

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

from experiment_new_benchmarks.config import (
    ALL_DATASETS,
    ExperimentNewBenchmarksConfig,
    display_name_for_dataset,
)
from experiment_new_benchmarks.feature_sources import (
    build_embedding_source,
    build_judge_source,
    prefixed_task_id,
)
from experiment_new_responses.dataset import (
    TaggedResponses,
    all_observation_keys_from_responses,
    benchmark_key_for_dataset,
    load_agent_model_scaffold_map,
    load_tagged_responses,
    make_observation_key,
    parse_observation_key,
)
from experiment_new_responses.difficulty_predictors import combine_theta
from swebench_irt.model_scaffold_combine import theta_combine_weights_from_attrs
from experiment_new_responses.train_irt_split import (
    get_or_train_model_scaffold_observation_split_irt,
)
from experiment_new_tasks.bootstrap import paired_clustered_auc_bootstrap
from experiment_new_tasks.feature_predictor import (
    FeatureBasedPredictor,
    GroupedRidgePredictor,
)
from experiment_new_tasks.feature_source import GroupedFeatureSource, TaskFeatureSource
from experiment_new_tasks.results import sample_std


METHOD_DISPLAY_NAMES = {
    "oracle": "Oracle",
    "embedding": "Embedding",
    "judge": "LLM-as-a-Judge",
    "combined": "Combined",
    "baseline": "Baseline",
}


def evaluate_heldout_benchmark(
    config: ExperimentNewBenchmarksConfig,
    root: Path,
    *,
    heldout_dataset: str,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    """Train on all benchmarks except ``heldout_dataset`` and evaluate there."""

    if heldout_dataset not in ALL_DATASETS:
        raise ValueError(f"Unknown heldout_dataset={heldout_dataset!r}. Valid: {ALL_DATASETS}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    train_datasets = [dataset for dataset in ALL_DATASETS if dataset != heldout_dataset]
    heldout_benchmark = benchmark_key_for_dataset(heldout_dataset)
    train_benchmarks = {benchmark_key_for_dataset(dataset) for dataset in train_datasets}

    print("=" * 80)
    print(
        "EXPERIMENT NEW BENCHMARKS: "
        f"train={', '.join(train_datasets)} heldout={heldout_dataset}"
    )
    print("=" * 80)

    tagged_by_dataset = _load_prefixed_tagged_by_dataset(config, root)
    all_tagged = [
        row for dataset in ALL_DATASETS for row in tagged_by_dataset[dataset]
    ]
    agent_to_ms_pair = _load_agent_model_scaffold_maps(config, root, tagged_by_dataset)

    train_tagged = [
        row for row in all_tagged if row[0] in train_benchmarks
    ]
    all_train_item_ids = {
        str(task_id) for _, _, responses in train_tagged for task_id in responses.keys()
    }
    train_observations = [
        key
        for key in all_observation_keys_from_responses(train_tagged)
        if "::".join(key.split("::", 2)[:2]) in agent_to_ms_pair
    ]
    if not train_observations:
        raise RuntimeError("No training observations have model/scaffold mappings")

    print(f"\nTraining observations: {len(train_observations)}")
    print(f"Training items: {len(all_train_item_ids)}")
    print(f"Training model/scaffold agents: {len(agent_to_ms_pair)}")

    train_model, train_scaffold, train_items = get_or_train_model_scaffold_observation_split_irt(
        all_responses_tagged=train_tagged,
        agent_to_ms_pair=agent_to_ms_pair,
        train_observations=set(train_observations),
        all_item_ids=all_train_item_ids,
        output_base=root / config.irt_cache_dir / heldout_dataset / "model_scaffold_train",
        split_seed=config.split_seed,
        fold_idx=0,
        k_folds=1,
        irt_model=config.irt_model,
        theta_combine=config.theta_combine,
        epochs=config.irt_epochs,
        device=config.irt_device,
        lr=config.irt_lr,
    )
    oracle_model, oracle_scaffold, oracle_items = get_or_train_model_scaffold_observation_split_irt(
        all_responses_tagged=all_tagged,
        agent_to_ms_pair=agent_to_ms_pair,
        train_observations=set(all_observation_keys_from_responses(all_tagged)),
        all_item_ids={
            str(task_id) for _, _, responses in all_tagged for task_id in responses.keys()
        },
        output_base=root / config.irt_cache_dir / heldout_dataset / "model_scaffold_oracle",
        split_seed=config.split_seed,
        fold_idx=0,
        k_folds=1,
        irt_model=config.irt_model,
        theta_combine=config.theta_combine,
        epochs=config.irt_epochs,
        device=config.irt_device,
        lr=config.irt_lr,
    )

    test_records, skip_meta = _build_test_records(
        tagged_by_dataset[heldout_dataset],
        agent_to_ms_pair=agent_to_ms_pair,
        train_model_ids=set(str(x) for x in train_model.index),
        train_scaffold_ids=set(str(x) for x in train_scaffold.index),
        train_tagged=train_tagged,
    )
    if not test_records:
        raise RuntimeError(
            f"Held-out benchmark {heldout_dataset!r} has no scoreable responses after "
            "skipping unseen LLM/scaffold pairs"
        )

    print(
        f"Held-out observations scored: {len(test_records)} "
        f"(skipped_unseen_model={skip_meta['obs_skipped_unseen_model']}, "
        f"skipped_unseen_scaffold={skip_meta['obs_skipped_unseen_scaffold']})"
    )

    train_task_ids = [str(task_id) for task_id in train_items.index]
    test_task_ids = sorted({str(record["task_id"]) for record in test_records})
    baseline_predictions = _score_baseline(test_records)

    method_predictions: Dict[str, List[Dict[str, Any]]] = {
        "baseline": baseline_predictions,
        "oracle": _score_with_item_difficulties(
            test_records,
            model_abilities=oracle_model,
            scaffold_abilities=oracle_scaffold,
            item_difficulties=oracle_items["b"].to_dict(),
            theta_combine=config.theta_combine,
        ),
    }

    sources = _build_feature_sources(config, root, datasets=ALL_DATASETS)
    for method_name, source in sources.items():
        print(f"\nFitting {method_name} difficulty model...")
        predictor = _make_feature_predictor(method_name, source, config)
        predictor.fit(train_task_ids, train_items.loc[train_task_ids, "b"].values)
        predicted_difficulties = predictor.predict(test_task_ids)
        if method_name == "judge":
            _write_judge_predictions(
                root / config.output_dir / heldout_dataset / "predictions.csv",
                predicted_difficulties,
                heldout_benchmark=heldout_benchmark,
            )
        method_predictions[method_name] = _score_with_item_difficulties(
            test_records,
            model_abilities=train_model,
            scaffold_abilities=train_scaffold,
            item_difficulties=predicted_difficulties,
            theta_combine=config.theta_combine,
        )

    cv_results: Dict[str, Dict[str, Any]] = {}
    for method_name in ["baseline", "embedding", "judge", "combined", "oracle"]:
        if method_name not in method_predictions:
            continue
        records = method_predictions[method_name]
        baseline_records = _baseline_records_for(records, baseline_predictions)
        bootstrap = paired_clustered_auc_bootstrap(
            records,
            baseline_records,
            cluster_key="task_id",
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        cv_results[method_name] = {
            "mean_auc": float(bootstrap.auc),
            "std_auc": 0.0,
            "fold_aucs": [float(bootstrap.auc)],
            "mean_difference_vs_baseline": float(bootstrap.delta_auc),
            "bootstrap_difference_vs_baseline": asdict(bootstrap),
            "n_observations": int(len(records)),
            "n_items": int(len({str(record["task_id"]) for record in records})),
            "n_agents": int(len({str(record["agent_id"]) for record in records})),
            "k": 1,
            "n_fold_seeds": 1,
        }

    _print_summary(heldout_dataset, cv_results)

    return {
        "config": config.to_dict(),
        "heldout_dataset": heldout_dataset,
        "heldout_benchmark": heldout_benchmark,
        "train_datasets": train_datasets,
        "train_benchmarks": sorted(train_benchmarks),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
        "skip_meta": skip_meta,
        "cv_results": cv_results,
    }


def _load_prefixed_tagged_by_dataset(
    config: ExperimentNewBenchmarksConfig,
    root: Path,
) -> Dict[str, TaggedResponses]:
    tagged_by_dataset: Dict[str, TaggedResponses] = {}
    for dataset in ALL_DATASETS:
        responses_path = root / config.responses_paths[dataset]
        tagged = load_tagged_responses(responses_path, dataset)
        tagged_by_dataset[dataset] = [
            (
                benchmark,
                subject_id,
                {
                    prefixed_task_id(benchmark, str(task_id)): int(value)
                    for task_id, value in responses.items()
                },
            )
            for benchmark, subject_id, responses in tagged
        ]
    return tagged_by_dataset


def _load_agent_model_scaffold_maps(
    config: ExperimentNewBenchmarksConfig,
    root: Path,
    tagged_by_dataset: Mapping[str, TaggedResponses],
) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for dataset in ALL_DATASETS:
        mapping.update(
            load_agent_model_scaffold_map(
                root / config.responses_paths[dataset],
                dataset,
                tagged_by_dataset[dataset],
            )
        )
    if not mapping:
        raise RuntimeError("No agents could be mapped to model/scaffold pairs")
    return mapping


def _build_feature_sources(
    config: ExperimentNewBenchmarksConfig,
    root: Path,
    *,
    datasets: List[str],
) -> Dict[str, TaskFeatureSource]:
    embeddings = build_embedding_source(
        {dataset: root / path for dataset, path in config.embeddings_paths.items()},
        datasets,
    )
    judge = build_judge_source(
        {dataset: root / path for dataset, path in config.llm_judge_features_paths.items()},
        datasets,
    )
    sources: Dict[str, TaskFeatureSource] = {}
    if embeddings is not None:
        sources["embedding"] = embeddings
    if judge is not None:
        sources["judge"] = judge
    if embeddings is not None and judge is not None:
        sources["combined"] = GroupedFeatureSource([embeddings, judge])
    return sources


def _make_feature_predictor(
    method_name: str,
    source: TaskFeatureSource,
    config: ExperimentNewBenchmarksConfig,
):
    if method_name == "combined":
        return GroupedRidgePredictor(source)  # type: ignore[arg-type]
    return FeatureBasedPredictor(source, alphas=list(config.ridge_alphas))


def _write_judge_predictions(
    path: Path,
    predicted_difficulties: Mapping[str, float],
    *,
    heldout_benchmark: str,
) -> None:
    """Write held-out LLM-judge item difficulty predictions for downstream CAT."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        writer.writeheader()
        for task_id in sorted(predicted_difficulties):
            writer.writerow(
                {
                    "item_id": _unprefix_task_id(task_id, heldout_benchmark),
                    "diff_pred": float(predicted_difficulties[task_id]),
                    "split": "ood",
                    "fold": "",
                }
            )
    print(f"Wrote LLM-as-a-Judge predictions: {path}")


def _unprefix_task_id(task_id: str, benchmark: str) -> str:
    prefix = f"{benchmark}::"
    task_id_s = str(task_id)
    if not task_id_s.startswith(prefix):
        raise ValueError(
            f"Expected held-out task ID {task_id_s!r} to start with benchmark prefix {prefix!r}"
        )
    return task_id_s[len(prefix) :]


def _build_test_records(
    tagged: TaggedResponses,
    *,
    agent_to_ms_pair: Mapping[str, Tuple[str, str]],
    train_model_ids: set[str],
    train_scaffold_ids: set[str],
    train_tagged: TaggedResponses,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    p_success_by_model, baseline_meta = _empirical_success_by_model(
        train_tagged,
        agent_to_ms_pair,
    )
    records: List[Dict[str, Any]] = []
    meta = {
        "obs_total": 0,
        "obs_scored": 0,
        "obs_skipped_no_mapping": 0,
        "obs_skipped_unseen_model": 0,
        "obs_skipped_unseen_scaffold": 0,
        "obs_skipped_no_baseline": 0,
        **baseline_meta,
    }
    for benchmark, subject_id, responses in tagged:
        agent_key = f"{benchmark}::{subject_id}"
        for task_id, y_true in responses.items():
            meta["obs_total"] += 1
            if agent_key not in agent_to_ms_pair:
                meta["obs_skipped_no_mapping"] += 1
                continue
            model, scaffold = agent_to_ms_pair[agent_key]
            if model not in train_model_ids:
                meta["obs_skipped_unseen_model"] += 1
                continue
            if scaffold not in train_scaffold_ids:
                meta["obs_skipped_unseen_scaffold"] += 1
                continue
            if model not in p_success_by_model:
                meta["obs_skipped_no_baseline"] += 1
                continue
            records.append(
                {
                    "observation_key": make_observation_key(benchmark, subject_id, task_id),
                    "agent_id": agent_key,
                    "subject_id": str(subject_id),
                    "task_id": str(task_id),
                    "benchmark": str(benchmark),
                    "model": str(model),
                    "scaffold": str(scaffold),
                    "y_true": int(y_true),
                    "baseline_score": float(p_success_by_model[model]),
                }
            )
            meta["obs_scored"] += 1
    return records, meta


def _empirical_success_by_model(
    train_tagged: TaggedResponses,
    agent_to_ms_pair: Mapping[str, Tuple[str, str]],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    successes: Dict[str, int] = {}
    trials: Dict[str, int] = {}
    skipped_no_mapping = 0
    for benchmark, subject_id, responses in train_tagged:
        agent_key = f"{benchmark}::{subject_id}"
        if agent_key not in agent_to_ms_pair:
            skipped_no_mapping += len(responses)
            continue
        model, _ = agent_to_ms_pair[agent_key]
        successes[model] = successes.get(model, 0) + sum(int(v) for v in responses.values())
        trials[model] = trials.get(model, 0) + len(responses)
    if not trials:
        raise RuntimeError("Could not compute empirical success rates for any training LLM")
    return (
        {model: float(successes.get(model, 0) / n) for model, n in trials.items()},
        {
            "baseline_models": int(len(trials)),
            "baseline_train_observations": int(sum(trials.values())),
            "baseline_train_obs_skipped_no_mapping": int(skipped_no_mapping),
        },
    )


def _score_baseline(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "agent_id": str(record["agent_id"]),
            "task_id": str(record["task_id"]),
            "y_true": int(record["y_true"]),
            "y_score": float(record["baseline_score"]),
            "model": str(record["model"]),
            "scaffold": str(record["scaffold"]),
        }
        for record in records
    ]


def _score_with_item_difficulties(
    records: Sequence[Mapping[str, Any]],
    *,
    model_abilities,
    scaffold_abilities,
    item_difficulties: Mapping[str, float],
    theta_combine: str,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for record in records:
        model = str(record["model"])
        scaffold = str(record["scaffold"])
        task_id = str(record["task_id"])
        if model not in model_abilities.index:
            raise ValueError(f"Model {model!r} has no ability")
        if scaffold not in scaffold_abilities.index:
            raise ValueError(f"Scaffold {scaffold!r} has no ability")
        if task_id not in item_difficulties:
            raise ValueError(f"Task {task_id!r} has no predicted difficulty")
        theta = combine_theta(
            model_abilities.loc[model, "theta"],
            scaffold_abilities.loc[scaffold, "theta"],
            combine=theta_combine,
            model_id=model,
            **theta_combine_weights_from_attrs(
                model_abilities,
                scaffold_abilities,
                combine=theta_combine,
            ),
        )
        scored.append(
            {
                "agent_id": str(record["agent_id"]),
                "task_id": task_id,
                "y_true": int(record["y_true"]),
                "y_score": float(sigmoid(float(theta) - float(item_difficulties[task_id]))),
                "model": model,
                "scaffold": scaffold,
            }
        )
    return scored


def _baseline_records_for(
    records: Sequence[Mapping[str, Any]],
    baseline_predictions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_key = {
        (str(record["agent_id"]), str(record["task_id"])): record
        for record in baseline_predictions
    }
    baseline_records: List[Dict[str, Any]] = []
    for record in records:
        key = (str(record["agent_id"]), str(record["task_id"]))
        if key not in baseline_by_key:
            raise ValueError(f"Missing baseline prediction for {key}")
        baseline_records.append(dict(baseline_by_key[key]))
    return baseline_records


def _print_summary(heldout_dataset: str, cv_results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 95)
    print(f"SUMMARY: held-out {display_name_for_dataset(heldout_dataset)}")
    print("=" * 95)
    print(
        f"\n{'Method':<18} {'AUC':>10} {'Delta':>10} "
        f"{'95% CI':>23} {'p-value':>10} {'N':>10}"
    )
    print("-" * 95)
    for method_name, result in sorted(
        cv_results.items(),
        key=lambda item: item[1]["mean_auc"],
        reverse=True,
    ):
        bootstrap = result["bootstrap_difference_vs_baseline"]
        p_value = bootstrap["p_value"]
        p_str = f"<{p_value:.4f}" if bootstrap["p_value_is_upper_bound"] else f"{p_value:.4f}"
        ci_str = f"[{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}]"
        print(
            f"{METHOD_DISPLAY_NAMES.get(method_name, method_name):<18} "
            f"{result['mean_auc']:>10.4f} "
            f"{result['mean_difference_vs_baseline']:>10.4f} "
            f"{ci_str:>23} {p_str:>10} {result['n_observations']:>10}"
        )

