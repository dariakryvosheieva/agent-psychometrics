"""Cross-validation utilities for holding out agent-task observations."""

import hashlib
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

from experiment_new_responses.dataset import (
    ResponseExperimentData,
    make_observation_key,
    parse_observation_key,
)
from experiment_new_tasks.cross_validation import CrossValidationResult


class CVPredictor(Protocol):
    def fit(self, data: ResponseExperimentData, train_observation_ids: List[str]) -> None:
        ...

    def predict_probability(
        self, data: ResponseExperimentData, agent_id: str, task_id: str
    ) -> float:
        ...


def stable_k_fold_split_observations(
    observation_keys: Sequence[str],
    *,
    k: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """Split observed (agent, task) cells into deterministic folds.

    Each fold is accepted only if every held-out cell's agent and task both
    remain present in the training observations, which is required by the
    train-only IRT predictor.
    """

    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    unique_observations = sorted(set(str(key) for key in observation_keys))
    if len(unique_observations) < k:
        raise RuntimeError(
            f"Not enough observations ({len(unique_observations)}) for {k}-fold CV"
        )

    shuffled: List[Tuple[float, str]] = []
    for observation_key in unique_observations:
        h = hashlib.md5(f"{observation_key}::{int(seed)}".encode("utf-8")).hexdigest()
        shuffled.append((int(h[:8], 16) / float(16**8), observation_key))
    shuffled.sort()

    observation_folds: List[List[str]] = [[] for _ in range(k)]
    for idx, (_, observation_key) in enumerate(shuffled):
        observation_folds[idx % k].append(observation_key)

    all_set = set(unique_observations)
    folds: List[Tuple[List[str], List[str]]] = []
    for fold_idx, test_observations in enumerate(observation_folds, start=1):
        test_set = set(test_observations)
        train_set = all_set - test_set
        train_agents = set()
        train_tasks = set()
        for observation_key in train_set:
            benchmark, subject_id, task_id = parse_observation_key(observation_key)
            train_agents.add(f"{benchmark}::{subject_id}")
            train_tasks.add(task_id)

        invalid: List[str] = []
        for observation_key in test_observations:
            benchmark, subject_id, task_id = parse_observation_key(observation_key)
            if f"{benchmark}::{subject_id}" not in train_agents or task_id not in train_tasks:
                invalid.append(observation_key)
        if invalid:
            raise RuntimeError(
                f"Fold {fold_idx} has test observations without train marginal coverage. "
                f"First 5: {invalid[:5]}"
            )
        folds.append((sorted(train_set), sorted(test_set)))
    return folds


def _response_lookup(data: ResponseExperimentData) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for benchmark, subject_id, responses in data.responses:
        for task_id, actual in responses.items():
            lookup[make_observation_key(benchmark, subject_id, str(task_id))] = int(actual)
    return lookup


def _run_single_fold(
    predictor: CVPredictor,
    fold_idx: int,
    train_observations: List[str],
    test_observations: List[str],
    load_fold_data: Callable[[List[str], List[str], int, bool], ResponseExperimentData],
    diagnostics_extractor: Optional[Callable[[CVPredictor, int], Any]],
) -> Dict[str, Any]:
    data = load_fold_data(
        train_observations,
        test_observations,
        fold_idx,
        bool(getattr(predictor, "requires_train_irt", True)),
    )
    predictor.fit(data, train_observations)

    diagnostics = None
    if diagnostics_extractor is not None:
        diagnostics = diagnostics_extractor(predictor, fold_idx)

    responses_by_observation = _response_lookup(data)
    y_true: List[int] = []
    y_scores: List[float] = []
    for observation_key in test_observations:
        benchmark, subject_id, task_id = parse_observation_key(observation_key)
        if observation_key not in responses_by_observation:
            raise ValueError(f"Held-out observation is missing from responses: {observation_key!r}")
        prob = predictor.predict_probability(data, f"{benchmark}::{subject_id}", task_id)
        y_true.append(int(responses_by_observation[observation_key]))
        y_scores.append(float(prob))

    auc = None
    if len(y_true) >= 2 and len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))

    return {"fold_idx": fold_idx, "auc": auc, "diagnostics": diagnostics}


def evaluate_predictor_cv(
    predictor: CVPredictor,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int, bool], ResponseExperimentData],
    verbose: bool = True,
    diagnostics_extractor: Optional[Callable[[CVPredictor, int], Any]] = None,
) -> CrossValidationResult:
    fold_aucs: List[Optional[float]] = []
    fold_diagnostics: List[Any] = []
    for fold_idx, (train_observations, test_observations) in enumerate(folds):
        result = _run_single_fold(
            predictor,
            fold_idx,
            train_observations,
            test_observations,
            load_fold_data,
            diagnostics_extractor,
        )
        fold_aucs.append(result["auc"])
        if diagnostics_extractor is not None:
            fold_diagnostics.append(result["diagnostics"])
        if verbose:
            auc = result["auc"]
            auc_text = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"      Fold {fold_idx + 1}: AUC = {auc_text}")

    valid = [auc for auc in fold_aucs if auc is not None]
    return CrossValidationResult(
        mean_auc=float(np.mean(valid)) if valid else None,
        std_auc=float(np.std(valid)) if valid else None,
        fold_aucs=fold_aucs,
        k=len(folds),
        fold_diagnostics=fold_diagnostics if diagnostics_extractor is not None else None,
    )


def probability_from_theta(theta: float, difficulty: float) -> float:
    return float(sigmoid(float(theta) - float(difficulty)))
