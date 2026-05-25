"""Dataset loading for agent-pair holdout cross-validation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd

from swebench_irt.split_agents_model_scaffold import (
    _canonical_model,
    _canonical_scaffold,
    _model_for_subject,
    _scaffold_for_subject,
)

from experiment_new_agents.train_irt_split import (
    get_or_train_agent_split_irt,
    get_or_train_oracle_irt,
)
from swebench_irt.model_scaffold_combine import combine_theta


ResponseValue = int
TaggedResponses = List[Tuple[str, str, Dict[str, int]]]


@dataclass
class AgentExperimentData:
    responses: TaggedResponses
    agent_to_ms_pair: Dict[str, Tuple[str, str]]
    train_model_abilities: pd.DataFrame
    train_scaffold_abilities: pd.DataFrame
    train_items: pd.DataFrame
    full_abilities: pd.DataFrame
    full_items: pd.DataFrame
    train_agents: List[str]
    test_agents: List[str]
    all_item_ids: List[str]
    theta_combine: str = "sum"

    def get_train_agent_abilities(self) -> List[float]:
        values: List[float] = []
        for agent_key in self.train_agents:
            model, scaffold = self.agent_to_ms_pair[agent_key]
            values.append(
                combine_theta(
                    float(self.train_model_abilities.loc[model, "theta"]),
                    float(self.train_scaffold_abilities.loc[scaffold, "theta"]),
                    combine=self.theta_combine,
                )
            )
        return values


def benchmark_key_for_dataset(dataset: str) -> str:
    dataset_norm = str(dataset).strip().lower()
    if dataset_norm == "swebench_verified":
        return "verified"
    if dataset_norm == "terminalbench":
        return "terminal_bench"
    raise ValueError(f"Unsupported new-agents dataset: {dataset!r}")


def load_tagged_responses(responses_path: Path, dataset: str) -> TaggedResponses:
    benchmark = benchmark_key_for_dataset(dataset)
    tagged: TaggedResponses = []
    with open(responses_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            subject_id = str(record["subject_id"])
            responses = {str(k): int(v) for k, v in dict(record["responses"]).items()}
            tagged.append((benchmark, subject_id, responses))
    if not tagged:
        raise RuntimeError(f"Parsed 0 agents from {responses_path}")
    return tagged


def build_agent_model_scaffold_map(
    tagged: Sequence[Tuple[str, str, Dict[str, int]]],
) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for benchmark, subject_id, _ in tagged:
        treat_as_pro = benchmark == "pro"
        model = _model_for_subject(subject_id, treat_as_pro=treat_as_pro)
        scaffold = _scaffold_for_subject(subject_id, treat_as_pro=treat_as_pro)
        if model is None or scaffold is None:
            continue
        mapping[f"{benchmark}::{subject_id}"] = (str(model), str(scaffold))
    if not mapping:
        raise RuntimeError("No agents could be mapped to model/scaffold pairs")
    return mapping


def load_agent_model_scaffold_map(
    responses_path: Path,
    dataset: str,
    tagged: Sequence[Tuple[str, str, Dict[str, int]]],
) -> Dict[str, Tuple[str, str]]:
    """Map agents to canonical (LLM, scaffold) pairs.

    Terminal-Bench response rows include leaderboard metadata; use it when
    present because the subject id alone is not always splittable.
    """

    if benchmark_key_for_dataset(dataset) != "terminal_bench":
        return build_agent_model_scaffold_map(tagged)

    mapping: Dict[str, Tuple[str, str]] = {}
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
            mapping[f"terminal_bench::{subject_id}"] = (
                _canonical_model(model_raw),
                _canonical_scaffold(scaffold_raw),
            )

    if not mapping:
        raise RuntimeError(
            f"No Terminal-Bench agents in {responses_path} had usable model/scaffold metadata"
        )
    return mapping


def all_item_ids_from_responses(
    tagged: Sequence[Tuple[str, str, Dict[str, int]]],
) -> Set[str]:
    items: Set[str] = set()
    for _, _, responses in tagged:
        items.update(str(item_id) for item_id in responses.keys())
    if not items:
        raise RuntimeError("Response matrix contains 0 item IDs")
    return items


def load_dataset_for_agent_fold(
    *,
    dataset: str,
    responses_path: Path,
    train_agents: List[str],
    test_agents: List[str],
    fold_idx: int,
    k_folds: int,
    split_seed: int,
    irt_cache_dir: Path,
    oracle_cache_dir: Path,
    irt_model: str,
    irt_epochs: int,
    irt_device: str,
    irt_lr: float,
    theta_combine: str,
    load_train_irt: bool = True,
) -> AgentExperimentData:
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
        train_model_abilities, train_scaffold_abilities, train_items = (
            get_or_train_agent_split_irt(
                all_responses_tagged=tagged,
                agent_to_ms_pair=agent_to_ms_pair,
                train_agents=set(train_agents),
                all_item_ids=all_items,
                output_base=irt_cache_dir,
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
    else:
        train_model_abilities = pd.DataFrame(columns=["theta"])
        train_scaffold_abilities = pd.DataFrame(columns=["theta"])
        train_items = pd.DataFrame(columns=["b"])

    return AgentExperimentData(
        responses=tagged,
        agent_to_ms_pair=agent_to_ms_pair,
        train_model_abilities=train_model_abilities,
        train_scaffold_abilities=train_scaffold_abilities,
        train_items=train_items,
        full_abilities=full_abilities,
        full_items=full_items,
        train_agents=list(train_agents),
        test_agents=list(test_agents),
        all_item_ids=sorted(all_items),
        theta_combine=theta_combine,
    )
