"""Cross-validation utilities for holding out unseen agent pairs."""

import hashlib
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

from experiment_new_agents.dataset import AgentExperimentData
from experiment_new_tasks.cross_validation import CrossValidationResult


class CVPredictor(Protocol):
    def fit(self, data: AgentExperimentData, train_agent_ids: List[str]) -> None:
        ...

    def predict_probability(
        self, data: AgentExperimentData, agent_id: str, task_id: str
    ) -> float:
        ...


def stable_k_fold_split_agent_pairs(
    agent_keys: Sequence[str],
    agent_to_ms_pair: Dict[str, Tuple[str, str]],
    *,
    k: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """Split by model/scaffold pair, not agent id.

    Grouping by pair ensures a test pair is never jointly observed in training.
    The split is accepted only if every test pair has both its model and scaffold
    present individually among training pairs.
    """

    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    pair_to_agents: Dict[Tuple[str, str], List[str]] = {}
    for agent_key in sorted(str(a) for a in agent_keys):
        if agent_key not in agent_to_ms_pair:
            continue
        pair_to_agents.setdefault(agent_to_ms_pair[agent_key], []).append(agent_key)
    if len(pair_to_agents) < k:
        raise RuntimeError(
            f"Not enough model/scaffold pairs ({len(pair_to_agents)}) for {k}-fold CV"
        )

    model_counts: Dict[str, int] = {}
    scaffold_counts: Dict[str, int] = {}
    for model, scaffold in pair_to_agents:
        model_counts[model] = model_counts.get(model, 0) + 1
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1
    eligible_pairs = [
        pair for pair in pair_to_agents
        if model_counts[pair[0]] > 1 and scaffold_counts[pair[1]] > 1
    ]
    if len(eligible_pairs) < k:
        raise RuntimeError(
            "Not enough model/scaffold pairs with both marginals observed "
            f"({len(eligible_pairs)}) for {k}-fold CV"
        )

    shuffled: List[Tuple[float, Tuple[str, str]]] = []
    for pair in eligible_pairs:
        key = f"{pair[0]}::{pair[1]}::{int(seed)}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        shuffled.append((int(h[:8], 16) / float(16**8), pair))
    shuffled.sort()

    pair_folds: List[List[Tuple[str, str]]] = [[] for _ in range(k)]
    for idx, (_, pair) in enumerate(shuffled):
        pair_folds[idx % k].append(pair)

    all_pairs = set(pair_to_agents)
    folds: List[Tuple[List[str], List[str]]] = []
    for fold_idx, test_pairs_list in enumerate(pair_folds, start=1):
        test_pairs = set(test_pairs_list)
        train_pairs = all_pairs - test_pairs
        train_models = {model for model, _ in train_pairs}
        train_scaffolds = {scaffold for _, scaffold in train_pairs}
        invalid_pairs = [
            pair for pair in sorted(test_pairs)
            if pair[0] not in train_models or pair[1] not in train_scaffolds
        ]
        if invalid_pairs:
            raise RuntimeError(
                f"Fold {fold_idx} has test pairs without train marginal coverage: "
                f"{invalid_pairs[:5]}"
            )

        train_agents: List[str] = []
        test_agents: List[str] = []
        for pair, agents in pair_to_agents.items():
            if pair in test_pairs:
                test_agents.extend(agents)
            else:
                train_agents.extend(agents)
        folds.append((sorted(train_agents), sorted(test_agents)))
    return folds


def _run_single_fold(
    predictor: CVPredictor,
    fold_idx: int,
    train_agents: List[str],
    test_agents: List[str],
    load_fold_data: Callable[[List[str], List[str], int, bool], AgentExperimentData],
    diagnostics_extractor: Optional[Callable[[CVPredictor, int], Any]],
) -> Dict[str, Any]:
    data = load_fold_data(
        train_agents,
        test_agents,
        fold_idx,
        bool(getattr(predictor, "requires_train_irt", True)),
    )
    predictor.fit(data, train_agents)

    diagnostics = None
    if diagnostics_extractor is not None:
        diagnostics = diagnostics_extractor(predictor, fold_idx)

    test_agent_set = set(test_agents)
    y_true: List[int] = []
    y_scores: List[float] = []
    for benchmark, subject_id, responses in data.responses:
        agent_key = f"{benchmark}::{subject_id}"
        if agent_key not in test_agent_set:
            continue
        for task_id, actual in responses.items():
            prob = predictor.predict_probability(data, agent_key, str(task_id))
            y_true.append(int(actual))
            y_scores.append(float(prob))

    auc = None
    if len(y_true) >= 2 and len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))

    return {"fold_idx": fold_idx, "auc": auc, "diagnostics": diagnostics}


def evaluate_predictor_cv(
    predictor: CVPredictor,
    folds: List[Tuple[List[str], List[str]]],
    load_fold_data: Callable[[List[str], List[str], int, bool], AgentExperimentData],
    verbose: bool = True,
    diagnostics_extractor: Optional[Callable[[CVPredictor, int], Any]] = None,
) -> CrossValidationResult:
    fold_aucs: List[Optional[float]] = []
    fold_diagnostics: List[Any] = []
    for fold_idx, (train_agents, test_agents) in enumerate(folds):
        result = _run_single_fold(
            predictor,
            fold_idx,
            train_agents,
            test_agents,
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
