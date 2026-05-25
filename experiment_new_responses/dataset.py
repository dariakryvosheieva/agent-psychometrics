"""Dataset loading for observation-pair holdout cross-validation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd

from experiment_new_responses.train_irt_split import (
    get_or_train_model_scaffold_observation_split_irt,
    get_or_train_observation_split_irt,
    get_or_train_oracle_irt,
)
from swebench_irt.split_agents_model_scaffold import (
    GSO_ASSUMED_SCAFFOLD,
    _canonical_model,
    _canonical_scaffold,
    _model_for_subject,
    _scaffold_for_subject,
)


ResponseValue = int
TaggedResponses = List[Tuple[str, str, Dict[str, int]]]


@dataclass
class ResponseExperimentData:
    responses: TaggedResponses
    agent_to_ms_pair: Dict[str, Tuple[str, str]]
    train_model_abilities: pd.DataFrame
    train_scaffold_abilities: pd.DataFrame
    train_abilities: pd.DataFrame
    train_items: pd.DataFrame
    full_abilities: pd.DataFrame
    full_items: pd.DataFrame
    train_observations: List[str]
    test_observations: List[str]
    all_item_ids: List[str]
    theta_combine: str = "sum"

    def get_train_agent_abilities(self) -> List[float]:
        return [float(v) for v in self.train_abilities["ability"].values]


def benchmark_key_for_dataset(dataset: str) -> str:
    dataset_norm = str(dataset).strip().lower()
    if dataset_norm == "swebench_verified":
        return "verified"
    if dataset_norm == "swebench_pro":
        return "pro"
    if dataset_norm == "terminalbench":
        return "terminal_bench"
    if dataset_norm == "gso":
        return "gso"
    raise ValueError(f"Unsupported new-responses dataset: {dataset!r}")


def make_observation_key(benchmark: str, subject_id: str, task_id: str) -> str:
    return f"{benchmark}::{subject_id}::{task_id}"


def parse_observation_key(observation_key: str) -> Tuple[str, str, str]:
    parts = str(observation_key).split("::", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed observation key: {observation_key!r}")
    return parts[0], parts[1], parts[2]


def load_tagged_responses(responses_path: Path, dataset: str) -> TaggedResponses:
    benchmark = benchmark_key_for_dataset(dataset)
    tagged: TaggedResponses = []
    with open(responses_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            subject_id = str(record["subject_id"])
            responses: Dict[str, int] = {}
            for key, value in dict(record["responses"]).items():
                if isinstance(value, dict):
                    raise ValueError(
                        "experiment_new_responses currently expects binary response "
                        f"cells; got binomial cell for {subject_id!r}, {key!r}: {value!r}"
                    )
                responses[str(key)] = int(value)
            tagged.append((benchmark, subject_id, responses))
    if not tagged:
        raise RuntimeError(f"Parsed 0 agents from {responses_path}")
    return tagged


def all_item_ids_from_responses(
    tagged: Sequence[Tuple[str, str, Dict[str, int]]],
) -> Set[str]:
    items: Set[str] = set()
    for _, _, responses in tagged:
        items.update(str(item_id) for item_id in responses.keys())
    if not items:
        raise RuntimeError("Response matrix contains 0 item IDs")
    return items


def all_observation_keys_from_responses(
    tagged: Sequence[Tuple[str, str, Dict[str, int]]],
) -> List[str]:
    observations: List[str] = []
    for benchmark, subject_id, responses in tagged:
        for item_id in responses.keys():
            observations.append(make_observation_key(benchmark, subject_id, str(item_id)))
    if not observations:
        raise RuntimeError("Response matrix contains 0 observations")
    return sorted(observations)


def load_agent_model_scaffold_map(
    responses_path: Path,
    dataset: str,
    tagged: Sequence[Tuple[str, str, Dict[str, int]]],
) -> Dict[str, Tuple[str, str]]:
    benchmark = benchmark_key_for_dataset(dataset)
    mapping: Dict[str, Tuple[str, str]] = {}

    if benchmark == "terminal_bench":
        with open(responses_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                subject_id = str(record["subject_id"])
                model_raw = str(record.get("model", "") or "").strip()
                scaffold_raw = str(record.get("agent", "") or "").strip()
                if not model_raw or not scaffold_raw or scaffold_raw.isdigit():
                    date_model = str(record.get("date", "") or "").strip()
                    if date_model and model_raw:
                        model_raw, scaffold_raw = date_model, model_raw
                if "," in model_raw or model_raw.lower() == "multiple":
                    continue
                if not model_raw or not scaffold_raw:
                    continue
                mapping[f"{benchmark}::{subject_id}"] = (
                    _canonical_model(model_raw),
                    _canonical_scaffold(scaffold_raw),
                )
    else:
        for bench, subject_id, _ in tagged:
            if bench == "gso":
                mapping[f"{bench}::{subject_id}"] = (
                    _canonical_model(subject_id),
                    GSO_ASSUMED_SCAFFOLD,
                )
                continue
            treat_as_pro = bench == "pro"
            model = _model_for_subject(subject_id, treat_as_pro=treat_as_pro)
            scaffold = _scaffold_for_subject(subject_id, treat_as_pro=treat_as_pro)
            if model is None or scaffold is None:
                continue
            mapping[f"{bench}::{subject_id}"] = (str(model), str(scaffold))

    if not mapping:
        raise RuntimeError("No agents could be mapped to model/scaffold pairs")
    return mapping


def load_many_tagged_responses(
    dataset_to_responses_path: Dict[str, Path],
) -> TaggedResponses:
    tagged: TaggedResponses = []
    for dataset, responses_path in dataset_to_responses_path.items():
        tagged.extend(load_tagged_responses(responses_path, dataset))
    if not tagged:
        raise RuntimeError("Parsed 0 agents across response files")
    return tagged


def load_many_agent_model_scaffold_maps(
    dataset_to_responses_path: Dict[str, Path],
    tagged_by_dataset: Dict[str, TaggedResponses],
) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for dataset, responses_path in dataset_to_responses_path.items():
        mapping.update(
            load_agent_model_scaffold_map(
                responses_path,
                dataset,
                tagged_by_dataset[dataset],
            )
        )
    if not mapping:
        raise RuntimeError("No agents could be mapped across response files")
    return mapping


def load_dataset_for_observation_fold(
    *,
    dataset: str,
    responses_path: Path,
    train_observations: List[str],
    test_observations: List[str],
    fold_idx: int,
    k_folds: int,
    split_seed: int,
    irt_cache_dir: Path,
    oracle_cache_dir: Path,
    irt_epochs: int,
    irt_device: str,
    irt_lr: float,
    irt_model: str,
    load_train_irt: bool = True,
    theta_combine: str = "sum",
) -> ResponseExperimentData:
    tagged = load_tagged_responses(responses_path, dataset)
    agent_to_ms_pair = load_agent_model_scaffold_map(responses_path, dataset, tagged)
    all_items = all_item_ids_from_responses(tagged)

    full_abilities, full_items = get_or_train_oracle_irt(
        all_responses_tagged=tagged,
        all_item_ids=all_items,
        output_dir=oracle_cache_dir,
        epochs=irt_epochs,
        device=irt_device,
        seed=0,
    )
    if load_train_irt:
        train_model_abilities, train_scaffold_abilities, _ = (
            get_or_train_model_scaffold_observation_split_irt(
                all_responses_tagged=tagged,
                agent_to_ms_pair=agent_to_ms_pair,
                train_observations=set(train_observations),
                all_item_ids=all_items,
                output_base=irt_cache_dir / "model_scaffold",
                split_seed=split_seed,
                fold_idx=fold_idx,
                k_folds=k_folds,
                irt_model=irt_model,
                theta_combine=theta_combine,
                epochs=irt_epochs,
                device=irt_device,
                lr=irt_lr,
            )
        )
        train_abilities, train_items = get_or_train_observation_split_irt(
            all_responses_tagged=tagged,
            train_observations=set(train_observations),
            all_item_ids=all_items,
            output_base=irt_cache_dir / "standard",
            split_seed=split_seed,
            fold_idx=fold_idx,
            k_folds=k_folds,
            epochs=irt_epochs,
            device=irt_device,
        )
    else:
        train_model_abilities = pd.DataFrame(columns=["theta"])
        train_scaffold_abilities = pd.DataFrame(columns=["theta"])
        train_abilities = pd.DataFrame(columns=["ability"])
        train_items = pd.DataFrame(columns=["b"])

    return ResponseExperimentData(
        responses=tagged,
        agent_to_ms_pair=agent_to_ms_pair,
        train_model_abilities=train_model_abilities,
        train_scaffold_abilities=train_scaffold_abilities,
        train_abilities=train_abilities,
        train_items=train_items,
        full_abilities=full_abilities,
        full_items=full_items,
        train_observations=list(train_observations),
        test_observations=list(test_observations),
        all_item_ids=sorted(all_items),
        theta_combine=theta_combine,
    )
