#!/usr/bin/env python3
"""
Joint (block) ridge difficulty prediction using:
  - [embedding vector, LLM-judge features] -> ridge -> difficulty_pred

The ridge objective uses two different L2 penalties:
  - embedding weights: alpha_emb * ||w_emb||^2
  - judge weights:     alpha_judge * ||w_judge||^2

For each of K CV folds over tasks/items:
  - Train IRT (1PL) on K-1 folds (no leakage; retrained per fold, no reuse).
  - Fit joint block-ridge on train items that have both embeddings and judge features.
  - Evaluate held-out AUROC on held-out items using:
        p(success) = sigmoid(theta_subject - z_item_pred_final)

LLM-judge features are read from:
  fulcrum/fellowship/llm_judge/features/verified/<task_id>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    import numpy as np


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Install the fellowship requirements (see "
            f"`fulcrum/fellowship/trajectory_embedding_requirements.txt`) in a CPython environment. "
            f"Original error: {e}"
        ) from e


DIFFICULTY_INSTRUCTION = (
    "How difficult is the above task for a coding agent? Please output one floating-point number from 0 (very easy) to 1 (very hard). Your difficulty score:\n"
)


VERIFIED_JUDGE_FEATURE_NAMES: List[str] = [
    "fix_in_description",
    "problem_clarity",
    "error_message_provided",
    "reproduction_steps",
    "fix_locality",
    "domain_knowledge_required",
    "fix_complexity",
    "logical_reasoning_required",
    "atypicality",
]

# SWE-bench Pro judge features (different schema).
PRO_JUDGE_FEATURE_NAMES: List[str] = [
    "fix_complexity",
    "verification_difficulty",
    "standard_pattern_available",
    "integration_complexity",
]

# Terminal-Bench judge features (different schema again).
TERMINAL_BENCH_JUDGE_FEATURE_NAMES: List[str] = [
    "solution_in_instruction",
    "task_clarity",
    "solution_size",
    "domain_knowledge_required",
    "task_complexity",
    "logical_reasoning_required",
    "atypicality",
    "tooling_complexity",
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


# Match ID normalization policy from the base script:
# - strip leading "instance_"
# - strip trailing "-v..." (including "-vc<hash>" and "-vnan")
_V_SUFFIX_RE = re.compile(r"-v.*$")


def normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


def set_torch_determinism(enabled: bool) -> None:
    """
    Toggle PyTorch deterministic algorithm behavior (best-effort).

    Mirrors `predict_question_difficulty.set_torch_determinism`.
    """
    try:
        import torch
    except Exception:
        return

    on = bool(enabled)
    try:
        torch.use_deterministic_algorithms(on, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(on)
        except Exception:
            pass
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = on
        torch.backends.cudnn.benchmark = (not on)
    except Exception:
        pass


def seed_everything(seed: int, *, deterministic: bool) -> None:
    """
    `seed_everything` with an explicit deterministic flag.

    Mirrors the base script's behavior:
    - seed python/numpy/torch/transformers
    - if deterministic: enable torch deterministic algorithms and disable TF32
    """
    import numpy as np

    s = int(seed)
    try:
        os.environ.setdefault("PYTHONHASHSEED", str(s))
    except Exception:
        pass

    random.seed(s)
    np.random.seed(s)

    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(s)
            except Exception:
                pass
    except Exception:
        torch = None  # type: ignore

    # Transformers helper (if available).
    try:
        from transformers import set_seed as _hf_set_seed  # type: ignore

        _hf_set_seed(s)
    except Exception:
        pass

    if deterministic:
        set_torch_determinism(True)
        # Reduce numeric drift from TF32 on Ampere+ GPUs.
        try:
            import torch  # type: ignore

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass


def iter_subject_responses_jsonl(path: str):
    """
    Yield (subject_id, responses) from a JSONL file with schema:
      {"subject_id": "...", "responses": {"task_id": 0/1, ...}}
    Normalizes item ids.
    """
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
            for raw_id, v in resp.items():
                tid = normalize_swebench_item_id(str(raw_id))
                if not tid:
                    continue
                try:
                    out[tid] = int(v)
                except Exception:
                    out[tid] = 1 if v else 0
            if out:
                yield sid, out


def load_all_responses(path: str) -> List[Tuple[str, Dict[str, int]]]:
    out: List[Tuple[str, Dict[str, int]]] = []
    for sid, resp in iter_subject_responses_jsonl(path):
        if resp:
            out.append((sid, resp))
    return out


def compute_zero_success_items(all_responses: List[Tuple[str, Dict[str, int]]]) -> List[str]:
    counts: Dict[str, int] = {}
    seen: Set[str] = set()
    for _, resp in all_responses:
        for tid, v in resp.items():
            seen.add(tid)
            counts[tid] = counts.get(tid, 0) + int(v)
    return sorted([tid for tid in seen if counts.get(tid, 0) == 0])


def _load_embeddings_from_npz(path: str):
    import numpy as np

    data = np.load(str(path), allow_pickle=True)
    task_ids = [str(x) for x in list(data["task_ids"].tolist())]
    X = data["X"].astype(np.float32)
    if X.shape[0] != len(task_ids):
        raise RuntimeError(f"Embeddings cache shape mismatch: X={X.shape} task_ids={len(task_ids)} path={path}")
    return task_ids, X


def _compute_embeddings_cache_path(
    *,
    out_dir: str,
    backbone: str,
    dataset_sources_str: str,
    split: str,
    instruction: str,
    embedding_layer: int,
    max_length: int,
    prompt_signature_fn,
) -> str:
    """
    Mirror the cache naming scheme used by `predict_question_difficulty.py` closely.
    """
    safe_backbone = str(backbone).replace("/", "__")
    ds_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dataset_sources_str))[:64]
    split_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(split))[:32]
    instr_sig = str(prompt_signature_fn(str(instruction)))
    layer_flag = "" if int(embedding_layer) == -1 else f"__layer{int(embedding_layer)}"
    idnorm_flag = "__idnorm_instance-v1"
    return os.path.join(
        str(out_dir),
        f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__qs-sol-instr__{instr_sig}{idnorm_flag}__{ds_flag}__{split_flag}__maxlen{int(max_length)}.npz",
    )


_JUDGE_INDEX_CACHE: Dict[str, Dict[str, str]] = {}


def _build_judge_index(features_dir: str) -> Dict[str, str]:
    """
    Build a mapping from normalized task id -> JSON file path.

    Why: Pro judge JSONs are stored with filenames like
      instance_<id>-v<hash>.json
    while the rest of this pipeline normalizes ids (drops `instance_` and `-v...`).
    """
    root = os.path.abspath(str(features_dir))
    if root in _JUDGE_INDEX_CACHE:
        return _JUDGE_INDEX_CACHE[root]

    idx: Dict[str, str] = {}
    try:
        names = [x for x in os.listdir(root) if x.endswith(".json")]
    except Exception:
        names = []
    for fn in names:
        stem = fn[:-5]
        norm = normalize_swebench_item_id(stem)
        if not norm:
            continue
        # Prefer the first seen file; collisions are rare and not critical here.
        idx.setdefault(norm, os.path.join(root, fn))

    _JUDGE_INDEX_CACHE[root] = idx
    return idx


def _infer_judge_schema(features_dir: str) -> str:
    """
    Heuristic schema inference:
    - Pro judge features live under `llm_judge/features/pro/`
    - Verified judge features live under `llm_judge/features/verified/`
    """
    p = os.path.abspath(str(features_dir)).replace("\\", "/").lower()
    # Match common layouts like ".../llm_judge/features/pro" or ".../features/pro/..."
    if "/features/pro" in p or p.endswith("/pro"):
        return "pro"
    if "/features/terminal_bench" in p or p.endswith("/terminal_bench") or p.endswith("/terminal-bench"):
        return "terminal_bench"
    return "verified"


def _load_judge_vector(
    task_id: str,
    *,
    features_dir: str,
    feature_names: Sequence[str],
    index: Dict[str, str],
) -> Optional[np.ndarray]:
    import numpy as np

    tid = str(task_id or "").strip()
    if not tid:
        return None

    # 1) exact match (for Verified, or if caller passes full Pro id)
    p = os.path.join(str(features_dir), f"{tid}.json")
    if not os.path.exists(p):
        # 2) normalized lookup (for Pro filenames with -v... suffix)
        norm = normalize_swebench_item_id(tid)
        p = index.get(norm, "")
        if not p or not os.path.exists(p):
            return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    xs: List[float] = []
    for k in feature_names:
        v = obj.get(k, None)
        if v is None:
            return None
        try:
            xs.append(float(v))
        except Exception:
            return None
    return np.asarray(xs, dtype=np.float32)


def _make_ridge_model(
    *,
    regressor: str,
    ridge_alpha: float,
    ridge_alphas: str,
    inner_splits: int,
    n_train: int,
    fold_seed: int,
):
    import numpy as np
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    name = str(regressor or "ridge_cv").strip()
    if name == "ridge":
        alpha = float(ridge_alpha)
        if not (alpha > 0):
            raise ValueError("--ridge_alpha must be > 0")
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    if name == "ridge_cv":
        try:
            alphas = np.array([float(x.strip()) for x in str(ridge_alphas).split(",") if x.strip()], dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to parse --ridge_alphas={ridge_alphas!r}: {e}") from e
        if alphas.size == 0:
            raise ValueError("Expected at least one alpha in --ridge_alphas")
        req_inner = int(inner_splits)
        if req_inner < 2:
            raise ValueError("--inner_splits must be >= 2")
        inner = int(min(req_inner, max(2, int(n_train))))
        inner_cv = KFold(n_splits=inner, shuffle=True, random_state=int(fold_seed))
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", RidgeCV(alphas=alphas, cv=inner_cv)),
            ]
        )
    raise ValueError(f"Unknown --regressor={regressor!r} (expected ridge or ridge_cv)")


def _parse_alpha_list(s: str):
    import numpy as np

    try:
        xs = [float(x.strip()) for x in str(s or "").split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse alpha list {s!r}: {e}") from e
    if not xs:
        raise ValueError("Expected at least one alpha.")
    arr = np.asarray(xs, dtype=np.float64)
    if not np.all(arr > 0):
        raise ValueError(f"All alphas must be > 0; got {arr.tolist()}")
    return arr


def _fit_block_ridge(
    *,
    X_emb,
    X_judge,
    y,
    alpha_emb: float,
    alpha_judge: float,
):
    """
    Fit a ridge model with different penalties per feature block:
      min ||y - X_emb w_emb - X_judge w_judge||^2 + alpha_emb||w_emb||^2 + alpha_judge||w_judge||^2

    We implement this by:
      - standardizing each block
      - scaling each block by 1/sqrt(alpha_block)
      - fitting sklearn Ridge(alpha=1.0) on the transformed design
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

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
    model = Ridge(alpha=1.0, fit_intercept=True, random_state=None)
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


def _predict_block_ridge(state, *, X_emb, X_judge):
    import numpy as np

    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    X_emb_s = state["emb_scaler"].transform(X_emb)
    X_judge_s = state["judge_scaler"].transform(X_judge)
    X_t = np.concatenate(
        [
            X_emb_s / math.sqrt(float(state["alpha_emb"])),
            X_judge_s / math.sqrt(float(state["alpha_judge"])),
        ],
        axis=1,
    )
    pred = state["ridge"].predict(X_t)
    return np.asarray(pred, dtype=np.float64).reshape(-1)


def _decompose_block_ridge_single(state, *, x_emb_raw, x_judge_raw):
    """
    Decompose prediction into embedding vs judge contributions.

    Returns a dict with two views:
      - **raw_dot**: dot products against raw feature vectors (what you asked for)
      - **std_contrib**: contributions in standardized space (exactly what the model computes pre-intercept)

    In raw space we can write:
      yhat = intercept_raw + dot(w_emb_raw, x_emb_raw) + dot(w_judge_raw, x_judge_raw)
    """
    import numpy as np

    ridge = state["ridge"]
    coef = np.asarray(getattr(ridge, "coef_", []), dtype=np.float64).reshape(-1)
    if coef.size == 0:
        raise RuntimeError("Model has no coef_; cannot decompose.")

    n_emb = int(state["n_emb"])
    w_emb_t = coef[:n_emb].reshape(-1)
    w_judge_t = coef[n_emb:].reshape(-1)

    alpha_emb = float(state["alpha_emb"])
    alpha_judge = float(state["alpha_judge"])
    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]

    x_emb_raw = np.asarray(x_emb_raw, dtype=np.float64).reshape(1, -1)
    x_judge_raw = np.asarray(x_judge_raw, dtype=np.float64).reshape(1, -1)

    # Standardized-space contributions (exactly matches feature construction used for training).
    x_emb_s = emb_scaler.transform(x_emb_raw)[0]
    x_judge_s = judge_scaler.transform(x_judge_raw)[0]
    w_emb_std = w_emb_t / math.sqrt(alpha_emb)
    w_judge_std = w_judge_t / math.sqrt(alpha_judge)
    emb_contrib_std = float(np.dot(x_emb_s, w_emb_std))
    judge_contrib_std = float(np.dot(x_judge_s, w_judge_std))
    intercept_model = float(getattr(ridge, "intercept_", 0.0))

    # Raw-space effective weights (so you can compute dot products with raw feature components).
    emb_scale = np.asarray(getattr(emb_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    judge_scale = np.asarray(getattr(judge_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    emb_mean = np.asarray(getattr(emb_scaler, "mean_", None), dtype=np.float64).reshape(-1)
    judge_mean = np.asarray(getattr(judge_scaler, "mean_", None), dtype=np.float64).reshape(-1)

    # guard: StandardScaler can produce zeros if a feature is constant; avoid infs.
    emb_scale = np.where(emb_scale == 0.0, 1.0, emb_scale)
    judge_scale = np.where(judge_scale == 0.0, 1.0, judge_scale)

    w_emb_raw = w_emb_std / emb_scale
    w_judge_raw = w_judge_std / judge_scale
    emb_dot_raw = float(np.dot(x_emb_raw.reshape(-1), w_emb_raw))
    judge_dot_raw = float(np.dot(x_judge_raw.reshape(-1), w_judge_raw))
    intercept_raw = intercept_model - float(np.dot(emb_mean, w_emb_raw)) - float(np.dot(judge_mean, w_judge_raw))

    pred_check = intercept_model + emb_contrib_std + judge_contrib_std
    pred_raw = intercept_raw + emb_dot_raw + judge_dot_raw

    return {
        "pred": float(pred_check),
        "pred_raw": float(pred_raw),
        "intercept_model": float(intercept_model),
        "intercept_raw": float(intercept_raw),
        "emb_contrib_std": float(emb_contrib_std),
        "judge_contrib_std": float(judge_contrib_std),
        "emb_dot_raw": float(emb_dot_raw),
        "judge_dot_raw": float(judge_dot_raw),
    }


def _extract_block_ridge_raw_weights(state):
    """
    Return (w_emb_raw, w_judge_raw, intercept_raw) so that:
      yhat = intercept_raw + dot(w_emb_raw, x_emb_raw) + dot(w_judge_raw, x_judge_raw)
    """
    import numpy as np

    ridge = state["ridge"]
    coef = np.asarray(getattr(ridge, "coef_", []), dtype=np.float64).reshape(-1)
    if coef.size == 0:
        raise RuntimeError("Model has no coef_.")

    n_emb = int(state["n_emb"])
    w_emb_t = coef[:n_emb].reshape(-1)
    w_judge_t = coef[n_emb:].reshape(-1)

    alpha_emb = float(state["alpha_emb"])
    alpha_judge = float(state["alpha_judge"])
    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]

    emb_scale = np.asarray(getattr(emb_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    judge_scale = np.asarray(getattr(judge_scaler, "scale_", None), dtype=np.float64).reshape(-1)
    emb_mean = np.asarray(getattr(emb_scaler, "mean_", None), dtype=np.float64).reshape(-1)
    judge_mean = np.asarray(getattr(judge_scaler, "mean_", None), dtype=np.float64).reshape(-1)
    emb_scale = np.where(emb_scale == 0.0, 1.0, emb_scale)
    judge_scale = np.where(judge_scale == 0.0, 1.0, judge_scale)

    w_emb_std = w_emb_t / math.sqrt(alpha_emb)
    w_judge_std = w_judge_t / math.sqrt(alpha_judge)
    w_emb_raw = w_emb_std / emb_scale
    w_judge_raw = w_judge_std / judge_scale

    intercept_model = float(getattr(ridge, "intercept_", 0.0))
    intercept_raw = intercept_model - float(np.dot(emb_mean, w_emb_raw)) - float(np.dot(judge_mean, w_judge_raw))
    return w_emb_raw.astype(np.float32, copy=False), w_judge_raw.astype(np.float32, copy=False), float(intercept_raw)


def save_regression_weights_block_ridge(
    *,
    out_dir: str,
    state,
    judge_feature_names: Sequence[str],
    metadata: dict,
) -> Tuple[str, str]:
    """
    Save a minimal representation of the *joint* block-ridge.

    Writes:
      - regression_weights.json (metadata)
      - regression_weights.npz  (arrays: coef_emb_raw, coef_judge_raw, intercept_raw, judge_feature_names, plus scaler stats)
    """
    import numpy as np

    os.makedirs(str(out_dir), exist_ok=True)
    w_emb_raw, w_judge_raw, intercept_raw = _extract_block_ridge_raw_weights(state)

    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]
    emb_mean = np.asarray(getattr(emb_scaler, "mean_", []), dtype=np.float32).reshape(-1)
    emb_scale = np.asarray(getattr(emb_scaler, "scale_", []), dtype=np.float32).reshape(-1)
    judge_mean = np.asarray(getattr(judge_scaler, "mean_", []), dtype=np.float32).reshape(-1)
    judge_scale = np.asarray(getattr(judge_scaler, "scale_", []), dtype=np.float32).reshape(-1)

    weights_npz = os.path.join(str(out_dir), "regression_weights.npz")
    np.savez_compressed(
        weights_npz,
        coef_emb_raw=np.asarray(w_emb_raw, dtype=np.float32).reshape(-1),
        coef_judge_raw=np.asarray(w_judge_raw, dtype=np.float32).reshape(-1),
        intercept_raw=np.asarray([float(intercept_raw)], dtype=np.float32),
        alpha_emb=np.asarray([float(state["alpha_emb"])], dtype=np.float32),
        alpha_judge=np.asarray([float(state["alpha_judge"])], dtype=np.float32),
        emb_scaler_mean=emb_mean,
        emb_scaler_scale=emb_scale,
        judge_scaler_mean=judge_mean,
        judge_scaler_scale=judge_scale,
        judge_feature_names=np.asarray(list(judge_feature_names), dtype=object),
        n_emb=np.asarray([int(state["n_emb"])], dtype=np.int64),
        n_judge=np.asarray([int(state["n_judge"])], dtype=np.int64),
    )

    weights_json = os.path.join(str(out_dir), "regression_weights.json")
    meta = dict(metadata or {})
    meta.update(
        {
            "regressor": "block_ridge",
            "n_emb": int(state["n_emb"]),
            "n_judge": int(state["n_judge"]),
            "alpha_emb": float(state["alpha_emb"]),
            "alpha_judge": float(state["alpha_judge"]),
            "judge_feature_names": list(judge_feature_names),
            "weights_npz": str(weights_npz),
        }
    )
    with open(weights_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return weights_json, weights_npz


def _select_block_alphas_inner_cv(
    *,
    X_emb,
    X_judge,
    y,
    alphas_emb,
    alphas_judge,
    inner_splits: int,
    seed: int,
):
    """
    Select (alpha_emb, alpha_judge) via inner KFold minimizing MSE.

    Note: We re-fit scalers per inner split to avoid leakage.
    """
    import numpy as np
    from sklearn.model_selection import KFold

    X_emb = np.asarray(X_emb, dtype=np.float64)
    X_judge = np.asarray(X_judge, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y.shape[0])
    k = int(min(int(inner_splits), max(2, n)))
    cv = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    best = (None, None, float("inf"))
    for ae in alphas_emb:
        for aj in alphas_judge:
            mse_sum = 0.0
            n_folds = 0
            for tr, va in cv.split(y):
                st = _fit_block_ridge(
                    X_emb=X_emb[tr],
                    X_judge=X_judge[tr],
                    y=y[tr],
                    alpha_emb=float(ae),
                    alpha_judge=float(aj),
                )
                p = _predict_block_ridge(st, X_emb=X_emb[va], X_judge=X_judge[va])
                err = y[va] - p
                mse_sum += float(np.mean(err * err))
                n_folds += 1
            mse = mse_sum / max(1, n_folds)
            if mse < best[2]:
                best = (float(ae), float(aj), float(mse))
    if best[0] is None or best[1] is None:
        raise RuntimeError("Inner CV failed to select alphas.")
    return float(best[0]), float(best[1]), float(best[2])


def write_filtered_responses_jsonl(
    *,
    all_responses: List[Tuple[str, Dict[str, int]]],
    item_ids: Sequence[str],
    out_path: str,
) -> Tuple[int, int]:
    """
    Write a py_irt-compatible JSONL with responses restricted to `item_ids`.

    Policy:
    - Include only subjects with at least one observed response among `item_ids`.
    - Write a complete response dict over `item_ids`, filling missing with 0.

    Returns: (n_subjects_written, n_items)
    """
    items = [normalize_swebench_item_id(x) for x in list(item_ids)]
    items = [x for x in items if x]
    item_set = set(items)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, resp in all_responses:
            present = {k: int(v) for k, v in resp.items() if k in item_set}
            if not present:
                continue
            complete = {tid: int(present.get(tid, 0)) for tid in items}
            f.write(json.dumps({"subject_id": sid, "responses": complete}) + "\n")
            n_written += 1
    return n_written, len(items)


def train_irt_1pl(
    *,
    responses_jsonl: str,
    epochs: int,
    device: str,
    seed: int,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train a 1PL IRT model via local `py_irt/` on the provided response JSONL.

    Returns:
      - theta_by_subject
      - diff_by_item (b)
    """
    _require("torch")
    _require("pyro")
    import torch
    import pyro  # type: ignore

    from py_irt.config import IrtConfig  # type: ignore
    from py_irt.training import IrtModelTrainer  # type: ignore

    os.makedirs(str(out_dir), exist_ok=True)
    pyro.clear_param_store()

    cfg = IrtConfig(
        model_type="1pl",
        epochs=int(epochs),
        priors="hierarchical",
        dims=1,
        seed=int(seed),
    )
    trainer = IrtModelTrainer(data_path=str(responses_jsonl), config=cfg, verbose=False)
    trainer.train(device=str(device))

    # Save the standard artifacts used elsewhere in this repo.
    trainer.save(os.path.join(out_dir, "parameters.json"))
    best = trainer.best_params or {}
    with open(os.path.join(out_dir, "best_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)

    ability = best.get("ability", [])
    diff = best.get("diff", [])
    subj_map = best.get("subject_ids", {})
    item_map = best.get("item_ids", {})

    theta_by_subject: Dict[str, float] = {}
    for i in range(len(ability)):
        sid = str(subj_map.get(i, "")).strip()
        if sid:
            theta_by_subject[sid] = float(ability[i])

    diff_by_item: Dict[str, float] = {}
    for i in range(len(diff)):
        tid = normalize_swebench_item_id(str(item_map.get(i, "")).strip())
        if tid:
            diff_by_item[tid] = float(diff[i])

    with open(os.path.join(out_dir, "abilities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "theta"])
        w.writeheader()
        for sid, theta in sorted(theta_by_subject.items()):
            w.writerow({"subject_id": sid, "theta": float(theta)})

    with open(os.path.join(out_dir, "items.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "b"])
        w.writeheader()
        for tid, b in sorted(diff_by_item.items()):
            w.writerow({"item_id": tid, "b": float(b)})

    return theta_by_subject, diff_by_item


def retrain_fold_irt(
    *,
    fold_root: str,
    train_items: List[str],
    all_responses: List[Tuple[str, Dict[str, int]]],
    irt_epochs: int,
    irt_device: str,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Always retrain IRT for this fold and overwrite fold_root/irt_1pl artifacts.
    """
    irt_dir = os.path.join(str(fold_root), "irt_1pl")
    if os.path.exists(irt_dir):
        shutil.rmtree(irt_dir, ignore_errors=True)
    os.makedirs(irt_dir, exist_ok=True)

    train_jsonl = os.path.join(str(fold_root), "train_responses.jsonl")
    n_subj_written, n_items_written = write_filtered_responses_jsonl(
        all_responses=all_responses, item_ids=train_items, out_path=train_jsonl
    )
    if n_subj_written == 0 or n_items_written == 0:
        raise RuntimeError(f"Fold IRT: wrote 0 subjects/items to {train_jsonl} (check filtering).")

    dev = str(irt_device or "cpu").strip() or "cpu"
    # Fall back to CPU if CUDA isn't available.
    try:
        import torch

        if dev.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: --irt_device=cuda requested but CUDA is unavailable; falling back to cpu for IRT.")
            dev = "cpu"
    except Exception:
        dev = "cpu"

    # Mirror base script policy: determinism OFF during IRT only.
    set_torch_determinism(False)
    seed_everything(int(seed), deterministic=False)

    theta_by_subject, diff_by_item = train_irt_1pl(
        responses_jsonl=train_jsonl,
        epochs=int(irt_epochs),
        device=str(dev),
        seed=int(seed),
        out_dir=str(irt_dir),
    )

    # Restore deterministic setting for downstream sklearn/etc.
    set_torch_determinism(True)
    seed_everything(int(seed), deterministic=True)
    if not theta_by_subject or not diff_by_item:
        raise RuntimeError(f"Fold IRT: produced empty outputs under {irt_dir}")
    return theta_by_subject, diff_by_item


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # Mirror base defaults for convenience.
    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/swebench_verified_combined_features")
    p.add_argument(
        "--embeddings_cache",
        type=str,
        default="",
        help="Path to embeddings cache (.npz). If empty, uses the path recorded in out_dir/metrics.json if present.",
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute and overwrite embeddings cache.")

    # If no embeddings cache is provided, we can compute embeddings like the base script.
    p.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional local JSON/JSONL dataset path. If set, overrides --dataset_name.",
    )
    p.add_argument("--backbone", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="auto")
    p.add_argument("--embedding_layer", type=int, default=-1)
    p.add_argument("--instruction", type=str, default=DIFFICULTY_INSTRUCTION)
    p.add_argument(
        "--agent_results",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl",
        help="Response-matrix JSONL: {'subject_id': ..., 'responses': {'task_id': 0/1, ...}}",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument(
        "--include_zero_success",
        action="store_true",
        help="Include items with 0 successes in CV/IRT (not recommended).",
    )
    p.add_argument("--irt_epochs", type=int, default=5000)
    p.add_argument("--irt_device", type=str, default="cuda", help="cpu or cuda (if available).")

    p.add_argument(
        "--judge_features_dir",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/llm_judge/features/verified",
        help="Directory with per-task LLM-judge feature JSONs (<task_id>.json).",
    )

    p.add_argument(
        "--regressor",
        type=str,
        default="ridge_cv",
        choices=["ridge", "ridge_cv"],
        help="Ridge variant for joint block-ridge (and the optional embedding-only baseline).",
    )
    p.add_argument("--ridge_alpha", type=float, default=10000.0)
    p.add_argument("--ridge_alphas", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    p.add_argument("--inner_splits", type=int, default=5)
    p.add_argument(
        "--ridge_alpha_emb",
        type=float,
        default=float("nan"),
        help="Embedding block ridge alpha (only used when --regressor=ridge). Defaults to --ridge_alpha when unset.",
    )
    p.add_argument(
        "--ridge_alpha_judge",
        type=float,
        default=float("nan"),
        help="Judge block ridge alpha (only used when --regressor=ridge). Defaults to --ridge_alpha when unset.",
    )
    p.add_argument(
        "--ridge_alphas_emb",
        type=str,
        default="",
        help="Embedding alpha grid for inner CV (only used when --regressor=ridge_cv). Defaults to --ridge_alphas when unset.",
    )
    p.add_argument(
        "--ridge_alphas_judge",
        type=str,
        default="",
        help="Judge alpha grid for inner CV (only used when --regressor=ridge_cv). Defaults to --ridge_alphas when unset.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print per-fold debug diagnostics (emb_auc/final_auc + selected block alphas + contribution summary).",
    )

    args = p.parse_args(argv)

    _require("numpy")
    _require("sklearn")
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    os.makedirs(str(args.out_dir), exist_ok=True)
    seed_everything(int(args.seed), deterministic=True)

    # Resolve embeddings cache.
    emb_cache = str(args.embeddings_cache or "").strip()
    if not emb_cache:
        metrics_path = os.path.join(str(args.out_dir), "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                emb_cache = str(m.get("embeddings_cache", "") or "").strip()
            except Exception:
                emb_cache = ""

    # If still missing: compute embeddings (mirrors base script).
    if not emb_cache:
        _require("torch")
        _require("transformers")
        _require("datasets")
        _require("tqdm")
        _require("huggingface_hub")

        import predict_question_difficulty as base

        dataset_name = str(args.dataset_name or "").strip()
        dataset_path = str(args.dataset_path or "").strip()
        if dataset_path:
            dataset_sources_str = f"json:{os.path.basename(dataset_path) or 'dataset.jsonl'}"
        else:
            dataset_sources_str = dataset_name or "princeton-nlp/SWE-bench_Verified"

        emb_cache = _compute_embeddings_cache_path(
            out_dir=str(args.out_dir),
            backbone=str(args.backbone),
            dataset_sources_str=str(dataset_sources_str),
            split=str(args.split),
            instruction=str(args.instruction),
            embedding_layer=int(args.embedding_layer),
            max_length=int(args.max_length),
            prompt_signature_fn=base.prompt_signature,
        )

        if os.path.exists(emb_cache) and not bool(args.overwrite):
            print(f"Loaded embeddings cache: {emb_cache}")
        else:
            # Mirror base script: deterministic embedding computation.
            base.seed_everything(int(args.seed), deterministic=True)
            items = list(
                base.iter_swebench_items(
                    dataset_name=str(dataset_name),
                    split=str(args.split),
                    dataset_path=str(dataset_path),
                )
            )
            if not items:
                raise RuntimeError(f"Loaded 0 items for embedding (dataset_name={dataset_name!r}, dataset_path={dataset_path!r})")

            ids_sorted, emb_by_id, counts_by_id, emb_dim = base.embed_items(
                items=items,
                backbone=str(args.backbone),
                trust_remote_code=bool(args.trust_remote_code),
                max_length=int(args.max_length),
                batch_size=int(args.batch_size),
                device_map=str(args.device_map),
                torch_dtype=str(args.torch_dtype),
                attn_implementation=str(args.attn_implementation),
                instruction=str(args.instruction),
                embedding_layer=int(args.embedding_layer),
            )
            if not ids_sorted:
                raise RuntimeError("No embeddings were produced (empty ids set).")

            X_full = np.stack([emb_by_id[r] for r in ids_sorted], axis=0).astype(np.float32)
            counts_arr = np.array([int(counts_by_id.get(r, 0)) for r in ids_sorted], dtype=np.int64)
            os.makedirs(os.path.dirname(emb_cache) or ".", exist_ok=True)
            np.savez_compressed(
                emb_cache,
                task_ids=np.array(ids_sorted, dtype=object),
                X=X_full,
                counts_kind=np.array(["text_len_chars"], dtype=object),
                counts=counts_arr,
                dataset_name=np.array([str(dataset_sources_str)], dtype=object),
                split=np.array([str(args.split)], dtype=object),
                dataset_path=np.array([str(dataset_path)], dtype=object),
                n_items=np.array([int(len(ids_sorted))], dtype=np.int64),
                instruction=np.array([str(args.instruction)], dtype=object),
                instruction_signature=np.array([str(base.prompt_signature(str(args.instruction)))], dtype=object),
                backbone=np.array([str(args.backbone)], dtype=object),
                max_length=np.array([int(args.max_length)], dtype=np.int64),
                embedding_dim=np.array([int(emb_dim)], dtype=np.int64),
                embedding_layer=np.array([int(args.embedding_layer)], dtype=np.int64),
            )
            print(f"Wrote embeddings cache: {emb_cache} (n={len(ids_sorted)}, dim={int(emb_dim)})")

    if not os.path.exists(emb_cache):
        raise FileNotFoundError(f"Embeddings cache not found: {emb_cache}")

    task_ids, X = _load_embeddings_from_npz(emb_cache)
    id_to_row = {tid: i for i, tid in enumerate(task_ids)}

    all_responses = load_all_responses(str(args.agent_results))
    if not all_responses:
        raise RuntimeError(f"Loaded 0 subject responses from --agent_results={args.agent_results!r}")

    response_items: Set[str] = set()
    for _, resp in all_responses:
        response_items.update(resp.keys())

    overlap_ids = [tid for tid in task_ids if tid in response_items]
    if not overlap_ids:
        raise RuntimeError("No overlap between embedded task_ids and item_ids found in --agent_results responses.")

    zero_success_ids = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)
    exclude_zero_success = not bool(args.include_zero_success)
    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} overlapped items "
            f"(agent_results={args.agent_results})"
        )
    else:
        eligible = list(overlap_ids)
    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV.")

    Xy = np.stack([X[id_to_row[tid]] for tid in eligible], axis=0).astype(np.float32)

    # Judge feature schema + index (supports Pro filenames with -v... suffix).
    feat_dir = str(args.judge_features_dir)
    idx = _build_judge_index(feat_dir)
    schema = _infer_judge_schema(feat_dir)
    if schema == "pro":
        judge_feature_names = PRO_JUDGE_FEATURE_NAMES
    elif schema == "terminal_bench":
        judge_feature_names = TERMINAL_BENCH_JUDGE_FEATURE_NAMES
    else:
        judge_feature_names = VERIFIED_JUDGE_FEATURE_NAMES

    outer_cv = KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
    fold_aucs: List[float] = []
    fold_n_obs: List[int] = []
    fold_n_items_scored: List[int] = []
    fold_aucs_embedding_only: List[float] = []
    fold_alpha_emb: List[float] = []
    fold_alpha_judge: List[float] = []

    # Out-of-fold per-item predictions for predictions.csv (may be NaN if judge features missing).
    eligible_index = {tid: i for i, tid in enumerate(eligible)}
    yhat_oof = np.full((int(len(eligible)),), np.nan, dtype=np.float64)
    fold_of_item = np.full((int(len(eligible)),), -1, dtype=np.int32)

    best_fold_auc = -float("inf")
    best_fold = -1
    best_joint_state = None
    best_fold_root = ""

    for fold, (tr, te) in enumerate(outer_cv.split(Xy), start=1):
        train_items = [eligible[int(i)] for i in tr.tolist()]
        test_items = [eligible[int(i)] for i in te.tolist()]

        fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
        # User-requested behavior: retrain IRT scores per fold.
        theta_by_subject, diff_by_item = retrain_fold_irt(
            fold_root=fold_root,
            train_items=train_items,
            all_responses=all_responses,
            irt_epochs=int(args.irt_epochs),
            irt_device=str(args.irt_device),
            seed=int(args.seed),
        )

        train_labeled = [tid for tid in train_items if tid in diff_by_item]
        if len(train_labeled) < 2:
            raise RuntimeError(f"Fold {fold}: only {len(train_labeled)} train items had IRT difficulties.")

        # -------------------------
        # Optional baseline: embeddings-only ridge (for debug / comparability)
        # -------------------------
        seed_everything(int(args.seed) + int(fold), deterministic=True)
        X_train = np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(np.float32)
        y_train = np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=np.float32)
        emb_model = _make_ridge_model(
            regressor=str(args.regressor),
            ridge_alpha=float(args.ridge_alpha),
            ridge_alphas=str(args.ridge_alphas),
            inner_splits=int(args.inner_splits),
            n_train=int(len(train_labeled)),
            fold_seed=int(args.seed) + int(fold),
        )
        emb_model.fit(X_train, y_train)

        X_test = np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(np.float32)
        emb_pred_test = emb_model.predict(X_test).astype(np.float32)
        emb_pred_by_item_test = {tid: float(z) for tid, z in zip(test_items, emb_pred_test.tolist())}

        # -------------------------
        # Joint block-ridge: [embeddings, judge_features] -> difficulty
        # -------------------------
        joint_emb_train_rows: List[np.ndarray] = []
        joint_judge_train_rows: List[np.ndarray] = []
        joint_y_train_rows: List[float] = []
        joint_train_items_used: List[str] = []

        for tid in train_labeled:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                continue
            joint_emb_train_rows.append(X[id_to_row[tid]].astype(np.float32))
            joint_judge_train_rows.append(j.astype(np.float32))
            joint_y_train_rows.append(float(diff_by_item[tid]))
            joint_train_items_used.append(tid)

        if len(joint_train_items_used) < 2:
            raise RuntimeError(
                f"Fold {fold}: only {len(joint_train_items_used)} train items had judge features; cannot fit joint block-ridge."
            )

        X_emb_joint_train = np.stack(joint_emb_train_rows, axis=0).astype(np.float32)
        X_judge_joint_train = np.stack(joint_judge_train_rows, axis=0).astype(np.float32)
        y_joint_train = np.asarray(joint_y_train_rows, dtype=np.float32)

        reg = str(args.regressor or "ridge_cv").strip()
        if reg == "ridge":
            alpha_emb = float(args.ridge_alpha_emb) if math.isfinite(float(args.ridge_alpha_emb)) else float(args.ridge_alpha)
            alpha_judge = (
                float(args.ridge_alpha_judge) if math.isfinite(float(args.ridge_alpha_judge)) else float(args.ridge_alpha)
            )
            joint_state = _fit_block_ridge(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alpha_emb=float(alpha_emb),
                alpha_judge=float(alpha_judge),
            )
            inner_best_mse = float("nan")
        else:
            ae_grid_s = str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas)
            aj_grid_s = str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas)
            ae_grid = _parse_alpha_list(ae_grid_s)
            aj_grid = _parse_alpha_list(aj_grid_s)
            alpha_emb, alpha_judge, inner_best_mse = _select_block_alphas_inner_cv(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alphas_emb=ae_grid,
                alphas_judge=aj_grid,
                inner_splits=int(args.inner_splits),
                seed=int(args.seed) + 2000 + int(fold),
            )
            joint_state = _fit_block_ridge(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alpha_emb=float(alpha_emb),
                alpha_judge=float(alpha_judge),
            )

        fold_alpha_emb.append(float(joint_state["alpha_emb"]))
        fold_alpha_judge.append(float(joint_state["alpha_judge"]))

        # -------------------------
        # (4) Evaluate on held-out fold
        # -------------------------
        final_pred_by_item: Dict[str, float] = {}
        contrib_by_item: Dict[str, Dict[str, float]] = {}
        n_missing_judge = 0
        for tid in test_items:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                n_missing_judge += 1
                continue
            x_emb = X[id_to_row[tid]].reshape(1, -1).astype(np.float32)
            x_j = j.reshape(1, -1).astype(np.float32)
            z_final = float(_predict_block_ridge(joint_state, X_emb=x_emb, X_judge=x_j)[0])
            final_pred_by_item[tid] = z_final
            try:
                contrib = _decompose_block_ridge_single(joint_state, x_emb_raw=x_emb, x_judge_raw=x_j)
                contrib_by_item[tid] = {
                    "pred": float(contrib["pred"]),
                    "pred_raw": float(contrib["pred_raw"]),
                    "intercept_model": float(contrib["intercept_model"]),
                    "intercept_raw": float(contrib["intercept_raw"]),
                    # What you asked for:
                    "emb_dot_raw": float(contrib["emb_dot_raw"]),
                    "judge_dot_raw": float(contrib["judge_dot_raw"]),
                    # Useful sanity view:
                    "emb_contrib_std": float(contrib["emb_contrib_std"]),
                    "judge_contrib_std": float(contrib["judge_contrib_std"]),
                }
            except Exception:
                # Decomposition is for analysis only; prediction should still work.
                pass

        # Fill OOF predictions for this fold's test items (NaN if missing judge).
        for tid in test_items:
            i = eligible_index.get(tid, None)
            if i is None:
                continue
            fold_of_item[int(i)] = int(fold)
            if tid in final_pred_by_item:
                yhat_oof[int(i)] = float(final_pred_by_item[tid])

        # Save per-item decomposition for this fold (test items only).
        try:
            os.makedirs(fold_root, exist_ok=True)
            with open(os.path.join(str(fold_root), "block_contributions_test_items.json"), "w", encoding="utf-8") as f:
                json.dump(contrib_by_item, f, indent=2, sort_keys=True)
        except Exception:
            pass

        # Score only items with final predictions (judge features present), so we can
        # compare embedding-only vs final AUC on the exact same observation set.
        scored_items = set(final_pred_by_item.keys())
        scores_final: List[float] = []
        scores_emb: List[float] = []
        labels: List[int] = []
        test_set = set(test_items)
        for sid, resp in all_responses:
            th = theta_by_subject.get(sid, None)
            if th is None:
                continue
            theta = float(th)
            for item_id, y_obs in resp.items():
                if item_id not in test_set:
                    continue
                if item_id not in scored_items:
                    continue
                z = final_pred_by_item.get(item_id, None)
                if z is None:
                    continue
                z_emb = emb_pred_by_item_test.get(item_id, None)
                if z_emb is None:
                    continue
                scores_final.append(_sigmoid(theta - float(z)))
                scores_emb.append(_sigmoid(theta - float(z_emb)))
                labels.append(int(y_obs))

        if len(labels) < 2 or len(set(int(x) for x in labels)) < 2:
            fold_auc = float("nan")
            fold_auc_emb = float("nan")
        else:
            fold_auc = float(roc_auc_score(labels, scores_final))
            fold_auc_emb = float(roc_auc_score(labels, scores_emb))
        fold_aucs.append(fold_auc)
        fold_aucs_embedding_only.append(fold_auc_emb)
        fold_n_obs.append(int(len(labels)))
        fold_n_items_scored.append(int(len(final_pred_by_item)))

        if fold_auc == fold_auc and float(fold_auc) > float(best_fold_auc):
            best_fold_auc = float(fold_auc)
            best_fold = int(fold)
            best_joint_state = joint_state
            best_fold_root = str(fold_root)

        if bool(args.debug):
            try:
                print(
                    f"Fold {fold:02d} debug: emb_auc={fold_auc_emb:.4f} final_auc={fold_auc:.4f} "
                    f"alpha_emb={float(joint_state['alpha_emb']):.3g} alpha_judge={float(joint_state['alpha_judge']):.3g}"
                )

                # Relative contribution summary on scored test items (raw dot-products).
                if contrib_by_item:
                    emb_abs = []
                    judge_abs = []
                    frac_emb = []
                    for tid in scored_items:
                        c = contrib_by_item.get(tid, None)
                        if not c:
                            continue
                        e = abs(float(c.get("emb_dot_raw", 0.0)))
                        jv = abs(float(c.get("judge_dot_raw", 0.0)))
                        if not (math.isfinite(e) and math.isfinite(jv)):
                            continue
                        emb_abs.append(e)
                        judge_abs.append(jv)
                        denom = e + jv
                        frac_emb.append(float(e / denom) if denom > 0 else float("nan"))
                    if emb_abs and judge_abs:
                        print(
                            f"Fold {fold:02d} contrib (test items): "
                            f"mean|emb_dot|={float(np.mean(emb_abs)):.3g} "
                            f"mean|judge_dot|={float(np.mean(judge_abs)):.3g} "
                            f"mean frac_emb={float(np.nanmean(np.asarray(frac_emb))):.3g}"
                        )
            except Exception as e:
                print(f"Fold {fold:02d} debug: failed to inspect judge weights ({e})")

        print(
            f"Fold {fold:02d}: auc={fold_auc} missing_judge={n_missing_judge}"
        )

    auc_arr = np.asarray(fold_aucs, dtype=np.float64)
    auc_mean = float(np.nanmean(auc_arr)) if auc_arr.size else float("nan")
    auc_std = float(np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
    print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")
    print("Per-fold ROC-AUC: " + ", ".join([str(x) for x in fold_aucs]))

    if best_joint_state is None or best_fold < 1:
        raise RuntimeError("Failed to select a best CV fold model by ROC-AUC (all folds NaN?).")

    # Save regression weights from the best fold (by AUC), mirroring predict_question_difficulty.py.
    weights_meta = {
        "script": os.path.abspath(__file__),
        "id_normalization": "strip instance_ prefix; strip -v.* suffix",
        "seed": int(args.seed),
        "deterministic": True,
        "cv_n_splits": int(args.cv_folds),
        "cv_best_auc_fold": int(best_fold),
        "cv_best_auc": float(best_fold_auc),
        "best_fold_root": str(best_fold_root),
        "embeddings_cache": str(emb_cache),
        "agent_results": str(args.agent_results),
        "judge_features_dir": str(args.judge_features_dir),
        "judge_feature_schema": str(schema),
        "regressor": str(args.regressor),
        "ridge_alpha": float(args.ridge_alpha),
        "ridge_alphas": str(args.ridge_alphas),
        "ridge_alphas_emb": str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas),
        "ridge_alphas_judge": str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas),
        "inner_splits": int(args.inner_splits),
    }
    weights_json, weights_npz = save_regression_weights_block_ridge(
        out_dir=str(args.out_dir),
        state=best_joint_state,
        judge_feature_names=judge_feature_names,
        metadata=weights_meta,
    )

    # Predict on zero-success items (excluded from CV/IRT, if requested).
    zero_success_ids = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)
    zero_embedded = [tid for tid in task_ids if tid in zero_success_set] if exclude_zero_success else []
    yhat_zero: Dict[str, float] = {}
    if zero_embedded:
        for tid in zero_embedded:
            j = _load_judge_vector(tid, features_dir=feat_dir, feature_names=judge_feature_names, index=idx)
            if j is None:
                continue
            x_emb = X[id_to_row[tid]].reshape(1, -1).astype(np.float32)
            x_j = j.reshape(1, -1).astype(np.float32)
            yhat_zero[tid] = float(_predict_block_ridge(best_joint_state, X_emb=x_emb, X_judge=x_j)[0])

    # Write per-item predictions CSV (OOF CV + optional zero_success rows), mirroring predict_question_difficulty.py.
    pred_path = os.path.join(str(args.out_dir), "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        w.writeheader()

        for i, tid in enumerate(eligible):
            v = float(yhat_oof[int(i)])
            fold_id = int(fold_of_item[int(i)]) if int(fold_of_item[int(i)]) > 0 else ""
            split = "cv_val" if (v == v) else "missing_judge"
            w.writerow({"item_id": tid, "diff_pred": (v if v == v else ""), "split": split, "fold": fold_id})

        if yhat_zero:
            for tid, v in sorted(yhat_zero.items()):
                w.writerow({"item_id": tid, "diff_pred": float(v), "split": "zero_success", "fold": ""})

    # Persist a small metrics file for convenience.
    metrics_out = os.path.join(str(args.out_dir), "metrics.json")
    try:
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "script": os.path.abspath(__file__),
                    "embeddings_cache": str(emb_cache),
                    "agent_results": str(args.agent_results),
                    "judge_features_dir": str(args.judge_features_dir),
                    "judge_feature_schema": str(schema),
                    "judge_feature_names": list(judge_feature_names),
                    "cv_folds": int(args.cv_folds),
                    "seed": int(args.seed),
                    "exclude_zero_success": bool(exclude_zero_success),
                    "n_items_total": int(len(task_ids)),
                    "n_items_with_responses": int(len(overlap_ids)),
                    "n_items_eligible_cv": int(len(eligible)),
                    "n_items_zero_success_in_responses": int(len(zero_success_ids)),
                    "regressor": str(args.regressor),
                    "ridge_alpha": float(args.ridge_alpha),
                    "ridge_alphas": str(args.ridge_alphas),
                    "inner_splits": int(args.inner_splits),
                    "ridge_alpha_emb": (
                        float(args.ridge_alpha_emb)
                        if math.isfinite(float(args.ridge_alpha_emb))
                        else float(args.ridge_alpha)
                    ),
                    "ridge_alpha_judge": (
                        float(args.ridge_alpha_judge)
                        if math.isfinite(float(args.ridge_alpha_judge))
                        else float(args.ridge_alpha)
                    ),
                    "ridge_alphas_emb": str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas),
                    "ridge_alphas_judge": str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas),
                    "cv_selected_alpha_emb_folds": [float(x) for x in fold_alpha_emb],
                    "cv_selected_alpha_judge_folds": [float(x) for x in fold_alpha_judge],
                    "cv_best_auc_fold": int(best_fold),
                    "cv_best_auc": float(best_fold_auc),
                    "regression_weights_json": str(weights_json),
                    "regression_weights_npz": str(weights_npz),
                    "predictions_csv": str(pred_path),
                    "n_items_zero_success_predicted": int(len(yhat_zero)),
                    "cv_test_auc_folds": [float(x) for x in fold_aucs],
                    "cv_test_auc_mean": float(auc_mean),
                    "cv_test_auc_std": float(auc_std),
                    "cv_test_auc_folds_embedding_only": [float(x) for x in fold_aucs_embedding_only],
                    "cv_test_n_obs_folds": [int(x) for x in fold_n_obs],
                    "cv_test_n_items_scored_folds": [int(x) for x in fold_n_items_scored],
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"Wrote metrics: {metrics_out}")
    except Exception as e:
        print(f"WARNING: failed to write metrics file: {metrics_out} ({e})")

    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote regression weights: {weights_json} (arrays in {weights_npz})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

