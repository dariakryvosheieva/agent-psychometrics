"""Shared difficulty-prediction helpers."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils import embeddings as embedding_utils


DIFFICULTY_INSTRUCTION = embedding_utils.DIFFICULTY_INSTRUCTION
EMBEDDING_TEXT_FORMAT = embedding_utils.EMBEDDING_TEXT_FORMAT_WITH_SOLUTION
normalize_swebench_item_id = embedding_utils.normalize_swebench_item_id
prompt_signature = embedding_utils.prompt_signature
_candidate_embedding_roots = embedding_utils._candidate_embedding_roots
load_compatible_embeddings_cache = embedding_utils.load_compatible_embeddings_cache
find_compatible_embeddings_cache = embedding_utils.find_compatible_embeddings_cache


def probability_from_theta(theta: float, difficulty: float) -> float:
    return float(sigmoid(float(theta) - float(difficulty)))


JUDGE_FEATURE_NAMES: List[str] = [
    "atypicality",
    "codebase_scale",
    "codebase_scope",
    "debugging_complexity",
    "domain_knowledge_required",
    "error_specificity",
    "fix_localization",
    "implementation_language_complexity",
    "logical_reasoning_required",
    "side_effect_risk",
    "similar_issue_likelihood",
    "solution_complexity",
    "solution_hint",
    "test_edge_case_coverage",
    "verification_difficulty",
]

_JUDGE_INDEX_CACHE: Dict[Tuple[str, bool], Dict[str, str]] = {}
_JUDGE_CSV_HEADER_CACHE: Dict[str, List[str]] = {}
_JUDGE_CSV_CACHE: Dict[Tuple[str, bool, Tuple[str, ...]], Dict[str, np.ndarray]] = {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_subject_responses_jsonl(path: str) -> Iterator[Tuple[str, Dict[str, int]]]:
    p = str(path or "").strip()
    if not p:
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            sid = str(obj.get("subject_id", "") or "").strip()
            resp = obj.get("responses", {}) or {}
            if not sid or not isinstance(resp, dict):
                continue
            out: Dict[str, int] = {}
            for raw_id, value in resp.items():
                tid = normalize_swebench_item_id(str(raw_id))
                if not tid:
                    continue
                try:
                    out[tid] = int(value)
                except Exception:
                    out[tid] = 1 if value else 0
            if out:
                yield sid, out


def load_all_responses(path: str) -> List[Tuple[str, Dict[str, int]]]:
    return [(sid, resp) for sid, resp in iter_subject_responses_jsonl(path) if resp]


def _iter_jsonl(path: str) -> Iterator[dict]:
    p = str(path or "").strip()
    if not p:
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def iter_subject_responses_jsonl_generic(
    path: str, *, normalize_item_ids: bool
) -> Iterator[Tuple[str, Dict[str, int]]]:
    p = str(path or "").strip()
    if not p:
        return
    if not os.path.exists(p):
        raise FileNotFoundError(f"Agent results JSONL not found: {p}")
    for obj in _iter_jsonl(p):
        sid = str(obj.get("subject_id", "") or "").strip()
        resp = obj.get("responses", {}) or {}
        if not sid or not isinstance(resp, dict):
            continue
        out: Dict[str, int] = {}
        for raw_id, value in resp.items():
            tid_raw = str(raw_id or "").strip()
            if not tid_raw:
                continue
            tid = normalize_swebench_item_id(tid_raw) if normalize_item_ids else tid_raw
            if not tid:
                continue
            try:
                out[tid] = int(value)
            except Exception:
                out[tid] = 1 if value else 0
        if out:
            yield sid, out


def iter_subject_responses_jsonl_terminal(path: str) -> Iterator[Tuple[str, Dict[str, int]]]:
    return iter_subject_responses_jsonl_generic(path, normalize_item_ids=False)


def load_all_responses_terminal(path: str) -> List[Tuple[str, Dict[str, int]]]:
    return [(sid, resp) for sid, resp in iter_subject_responses_jsonl_terminal(path) if resp]


def load_all_responses_generic(*, path: str, normalize_item_ids: bool) -> List[Tuple[str, Dict[str, int]]]:
    return [
        (sid, resp)
        for sid, resp in iter_subject_responses_jsonl_generic(str(path), normalize_item_ids=bool(normalize_item_ids))
        if resp
    ]


def _compute_binary_auroc(scores: List[float], labels: List[int]) -> float:
    if not scores or len(set(int(x) for x in labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _looks_like_csv_path(path: str) -> bool:
    return str(path or "").strip().lower().endswith(".csv")


def _load_judge_csv_feature_names(features_csv: str) -> List[str]:
    root = os.path.abspath(str(features_csv))
    if root in _JUDGE_CSV_HEADER_CACHE:
        return list(_JUDGE_CSV_HEADER_CACHE[root])
    if not os.path.exists(root):
        raise FileNotFoundError(f"Judge features CSV not found: {features_csv!r}")
    if not os.path.isfile(root):
        raise ValueError(f"Expected a judge features CSV file path, got: {features_csv!r}")

    with open(root, "r", encoding="utf-8", newline="") as f:
        header = [str(x).strip() for x in next(csv.reader(f), []) if str(x).strip()]
    if not header or "instance_id" not in header:
        raise ValueError(f"Judge features CSV missing required header column 'instance_id': {features_csv!r}")
    feats = [h for h in header if h != "instance_id"]
    _JUDGE_CSV_HEADER_CACHE[root] = feats
    return list(feats)


def _load_judge_csv_vectors(
    features_csv: str,
    *,
    feature_names: Sequence[str],
    normalize_item_ids: bool = True,
) -> Dict[str, np.ndarray]:
    root = os.path.abspath(str(features_csv))
    key = (root, bool(normalize_item_ids), tuple(str(x) for x in feature_names))
    if key in _JUDGE_CSV_CACHE:
        return _JUDGE_CSV_CACHE[key]
    if not os.path.exists(root):
        raise FileNotFoundError(f"Judge features CSV not found: {features_csv!r}")
    if not os.path.isfile(root):
        raise ValueError(f"Expected a judge features CSV file path, got: {features_csv!r}")

    out: Dict[str, np.ndarray] = {}
    with open(root, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "instance_id" not in fieldnames:
            raise ValueError(f"Judge features CSV missing required column 'instance_id': {features_csv!r}")
        missing_cols = [k for k in feature_names if k not in fieldnames]
        if missing_cols:
            raise ValueError(
                f"Judge features CSV missing columns {missing_cols!r} (have {fieldnames!r}): {features_csv!r}"
            )
        for row in reader:
            iid_raw = str(row.get("instance_id") or "").strip()
            if not iid_raw:
                continue
            iid = (normalize_swebench_item_id(iid_raw) or iid_raw) if normalize_item_ids else iid_raw
            xs: List[float] = []
            ok = True
            for name in feature_names:
                value = row.get(str(name), "")
                if value is None or str(value).strip() == "":
                    ok = False
                    break
                try:
                    xs.append(float(str(value).strip()))
                except Exception:
                    ok = False
                    break
            if ok:
                out[iid] = np.asarray(xs, dtype=np.float32)

    _JUDGE_CSV_CACHE[key] = out
    return out


def _build_judge_index(features_dir: str, *, normalize_item_ids: bool = True) -> Dict[str, str]:
    root = os.path.abspath(str(features_dir))
    key = (root, bool(normalize_item_ids))
    if key in _JUDGE_INDEX_CACHE:
        return _JUDGE_INDEX_CACHE[key]

    if _looks_like_csv_path(features_dir):
        if not os.path.exists(root):
            raise FileNotFoundError(f"Judge features CSV not found: {features_dir!r}")
        if not os.path.isfile(root):
            raise ValueError(f"Expected a judge features CSV file path, got: {features_dir!r}")
        _JUDGE_INDEX_CACHE[key] = {}
        return {}

    idx: Dict[str, str] = {}
    try:
        names = [x for x in os.listdir(root) if x.endswith(".json")]
    except Exception:
        names = []
    for filename in names:
        stem = filename[:-5]
        norm = (normalize_swebench_item_id(stem) or stem) if normalize_item_ids else str(stem).strip()
        if norm:
            idx.setdefault(norm, os.path.join(root, filename))
    _JUDGE_INDEX_CACHE[key] = idx
    return idx


def _load_judge_vector(
    task_id: str,
    *,
    features_dir: str,
    feature_names: Sequence[str],
    index: Dict[str, str],
    normalize_item_ids: bool = True,
) -> Optional[np.ndarray]:
    tid = str(task_id or "").strip()
    if not tid:
        return None

    if _looks_like_csv_path(features_dir):
        vectors = _load_judge_csv_vectors(
            features_dir,
            feature_names=feature_names,
            normalize_item_ids=normalize_item_ids,
        )
        norm = (normalize_swebench_item_id(tid) or tid) if normalize_item_ids else tid
        return vectors.get(norm, None)

    path = os.path.join(str(features_dir), f"{tid}.json")
    if not os.path.exists(path):
        norm = (normalize_swebench_item_id(tid) or tid) if normalize_item_ids else tid
        path = index.get(norm, "")
        if not path or not os.path.exists(path):
            return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None

    xs: List[float] = []
    for name in feature_names:
        value = obj.get(name, None)
        if value is None:
            return None
        try:
            xs.append(float(value))
        except Exception:
            return None
    return np.asarray(xs, dtype=np.float32)


def _parse_alpha_list(spec: str) -> np.ndarray:
    try:
        xs = [float(x.strip()) for x in str(spec or "").split(",") if x.strip()]
    except Exception as exc:
        raise ValueError(f"Failed to parse alpha list {spec!r}: {exc}") from exc
    if not xs:
        raise ValueError("Expected at least one alpha.")
    arr = np.asarray(xs, dtype=np.float64)
    if not np.all(arr > 0):
        raise ValueError(f"All alphas must be > 0; got {arr.tolist()}")
    return arr


def _fit_block_ridge(
    *,
    X_emb: np.ndarray,
    X_judge: np.ndarray,
    y: np.ndarray,
    alpha_emb: float,
    alpha_judge: float,
) -> dict:
    ae = float(alpha_emb)
    aj = float(alpha_judge)
    if not (ae > 0 and aj > 0):
        raise ValueError(f"alpha_emb and alpha_judge must be > 0; got {ae}, {aj}")

    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_emb.shape[0] != X_judge.shape[0] or X_emb.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X_emb={X_emb.shape} X_judge={X_judge.shape} y={y.shape}")

    emb_scaler = StandardScaler(with_mean=True, with_std=True)
    judge_scaler = StandardScaler(with_mean=True, with_std=True)
    X_emb_s = emb_scaler.fit_transform(X_emb)
    X_judge_s = judge_scaler.fit_transform(X_judge)

    X_t = np.concatenate([X_emb_s / math.sqrt(ae), X_judge_s / math.sqrt(aj)], axis=1)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_t, y)
    return {
        "ridge": model,
        "emb_scaler": emb_scaler,
        "judge_scaler": judge_scaler,
        "alpha_emb": ae,
        "alpha_judge": aj,
        "n_emb": int(X_emb.shape[1]),
        "n_judge": int(X_judge.shape[1]),
    }


def _predict_block_ridge(state: dict, *, X_emb: np.ndarray, X_judge: np.ndarray) -> np.ndarray:
    X_emb_s = state["emb_scaler"].transform(np.asarray(X_emb, dtype=np.float64))
    X_judge_s = state["judge_scaler"].transform(np.asarray(X_judge, dtype=np.float64))
    X_t = np.concatenate(
        [
            X_emb_s / math.sqrt(float(state["alpha_emb"])),
            X_judge_s / math.sqrt(float(state["alpha_judge"])),
        ],
        axis=1,
    )
    return np.asarray(state["ridge"].predict(X_t), dtype=np.float64).reshape(-1)


def _select_block_alphas_inner_cv(
    *,
    X_emb: np.ndarray,
    X_judge: np.ndarray,
    y: np.ndarray,
    alphas_emb: np.ndarray,
    alphas_judge: np.ndarray,
    inner_splits: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    k = int(min(int(inner_splits), max(2, int(y.shape[0]))))
    inner_cv = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    best_ae: Optional[float] = None
    best_aj: Optional[float] = None
    best_mse = float("inf")
    total = int(len(alphas_emb)) * int(len(alphas_judge))
    seen = 0
    for ae in alphas_emb:
        for aj in alphas_judge:
            seen += 1
            fold_mses: List[float] = []
            for train_idx, val_idx in inner_cv.split(y):
                state = _fit_block_ridge(
                    X_emb=X_emb[train_idx],
                    X_judge=X_judge[train_idx],
                    y=y[train_idx],
                    alpha_emb=float(ae),
                    alpha_judge=float(aj),
                )
                pred = _predict_block_ridge(state, X_emb=X_emb[val_idx], X_judge=X_judge[val_idx])
                err = y[val_idx] - pred
                fold_mses.append(float(np.mean(err * err)))
            mse = float(np.mean(fold_mses))
            if mse < best_mse:
                best_mse = mse
                best_ae = float(ae)
                best_aj = float(aj)
            if verbose and (seen == 1 or seen % 10 == 0 or seen == total):
                print(
                    f"Block-ridge inner CV: tried {seen}/{total} "
                    f"(alpha_emb={float(ae):g}, alpha_judge={float(aj):g}) "
                    f"mse={mse:.6g} best_mse={best_mse:.6g}"
                )
    if best_ae is None or best_aj is None:
        raise RuntimeError("Inner CV failed to select block alphas.")
    return float(best_ae), float(best_aj), float(best_mse)
