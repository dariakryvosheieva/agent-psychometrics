#!/usr/bin/env python3
"""
Multi-benchmark version of `predict_question_difficulty.py`.

This script is intentionally kept as close as possible to
`fulcrum/fellowship/predict_question_difficulty.py`, except:

- **IRT model**: uses shared (base LLM, scaffold) abilities rather than a single
  per-agent theta. We use:

    p(success | model, scaffold, item) = sigmoid( (theta_model + theta_scaffold) - b_item )

  Training is performed via `swebench_irt/train_model_scaffold_shared.py` and agent
  decomposition is performed via `swebench_irt/split_agents_model_scaffold.py`.

- **Data pool**: uses a *common pool of items* drawn from:
  - SWE-bench Verified
  - SWE-bench Pro
  - Terminal-Bench

The rest of the pipeline is identical:
  - Embed each task as (statement + solution + instruction).
  - Default: split items once into train vs zero-success, train IRT on train,
    fit a regressor from embeddings -> item difficulty, and **save learned weights**
    (no evaluation).
  - Optional (`--eval_mode=id`): K-fold CV over items. For each fold,
    train IRT on train-fold items only (no leakage), fit a regressor, predict held-out
    item difficulties, and evaluate held-out AUROC using the fold's IRT abilities.
  - Optional (`--eval_mode=ood`, default): do the default training-only flow, then
    additionally evaluate ROC-AUC on a 4th benchmark using the learned shared IRT
    (model+scaffold) abilities and the learned regressor weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# Reuse all the "shared" logic (embeddings, regression, caching, etc.) from the
# single-benchmark script to stay identical by construction.
import predict_question_difficulty as base


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
    """
    Yield (subject_id, responses) from a response-matrix JSONL:
      {"subject_id": "...", "responses": {"<item_id>": 0/1, ...}}

    If normalize_item_ids=True, normalizes ids with `base.normalize_swebench_item_id`.
    """
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
        for raw_id, v in resp.items():
            tid_raw = str(raw_id or "").strip()
            if not tid_raw:
                continue
            tid = base.normalize_swebench_item_id(tid_raw) if bool(normalize_item_ids) else tid_raw
            if not tid:
                continue
            try:
                out[tid] = int(v)
            except Exception:
                out[tid] = 1 if v else 0
        if out:
            yield sid, out


def iter_subject_responses_jsonl_terminal(path: str) -> Iterator[Tuple[str, Dict[str, int]]]:
    """
    Terminal-Bench response JSONL iterator.

    Schema: {"subject_id": "...", "responses": {"<task_id>": 0/1, ...}}
    IMPORTANT: does NOT normalize item ids (Terminal-Bench task ids must be preserved).
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
                tid = str(raw_id or "").strip()
                if not tid:
                    continue
                try:
                    out[tid] = int(v)
                except Exception:
                    out[tid] = 1 if v else 0
            if out:
                yield sid, out


def load_all_responses_terminal(path: str) -> List[Tuple[str, Dict[str, int]]]:
    out: List[Tuple[str, Dict[str, int]]] = []
    for sid, resp in iter_subject_responses_jsonl_terminal(path):
        if resp:
            out.append((sid, resp))
    return out


def _sigmoid(x: float) -> float:
    # Stable enough for typical theta/z ranges in this repo.
    return 1.0 / (1.0 + math.exp(-float(x)))


# -----------------------------
# Judge feature schema (per benchmark)
# -----------------------------
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


_JUDGE_INDEX_CACHE: Dict[Tuple[str, bool], Dict[str, str]] = {}


def _build_judge_index(features_dir: str, *, normalize_item_ids: bool) -> Dict[str, str]:
    """
    Build a mapping from (maybe-normalized) task id -> JSON file path.

    Why: Pro judge JSONs are often stored with filenames like:
      instance_<id>-v<hash>.json
    while this pipeline may normalize ids (drops `instance_` and `-v...`).
    """
    root = os.path.abspath(str(features_dir))
    key = (root, bool(normalize_item_ids))
    if key in _JUDGE_INDEX_CACHE:
        return _JUDGE_INDEX_CACHE[key]

    idx: Dict[str, str] = {}
    try:
        names = [x for x in os.listdir(root) if x.endswith(".json")]
    except Exception:
        names = []
    for fn in names:
        stem = fn[:-5]
        norm = base.normalize_swebench_item_id(stem) if bool(normalize_item_ids) else str(stem).strip()
        if not norm:
            continue
        idx.setdefault(norm, os.path.join(root, fn))

    _JUDGE_INDEX_CACHE[key] = idx
    return idx


def _load_judge_vector(
    item_id: str,
    *,
    features_dir: str,
    feature_names: Sequence[str],
    index: Dict[str, str],
    normalize_item_ids: bool,
):
    """
    Load judge feature vector for `item_id` from `<features_dir>/<item_id>.json`,
    with an index fallback (important for Pro `instance_...-v...` filenames).
    """
    tid = str(item_id or "").strip()
    if not tid:
        return None

    # 1) exact match
    p = os.path.join(str(features_dir), f"{tid}.json")
    if not os.path.exists(p):
        # 2) normalized / indexed lookup
        key = base.normalize_swebench_item_id(tid) if bool(normalize_item_ids) else tid
        p = index.get(key, "")
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
    return base.np.asarray(xs, dtype=base.np.float32)


def _parse_alpha_list(s: str):
    try:
        xs = [float(x.strip()) for x in str(s or "").split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse alpha list {s!r}: {e}") from e
    if not xs:
        raise ValueError("Expected at least one alpha.")
    arr = base.np.asarray(xs, dtype=base.np.float64)
    if not base.np.all(arr > 0):
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
    Fit ridge with different penalties per feature block:
      min ||y - X_emb w_emb - X_judge w_judge||^2 + alpha_emb||w_emb||^2 + alpha_judge||w_judge||^2

    Mirrors `predict_question_difficulty_combined_features.py`.
    """
    ae = float(alpha_emb)
    aj = float(alpha_judge)
    if not (ae > 0 and aj > 0):
        raise ValueError(f"alpha_emb and alpha_judge must be > 0; got {ae}, {aj}")

    X_emb = base.np.asarray(X_emb, dtype=base.np.float64)
    X_judge = base.np.asarray(X_judge, dtype=base.np.float64)
    y = base.np.asarray(y, dtype=base.np.float64).reshape(-1)
    if X_emb.shape[0] != X_judge.shape[0] or X_emb.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X_emb={X_emb.shape} X_judge={X_judge.shape} y={y.shape}")

    emb_scaler = base.StandardScaler(with_mean=True, with_std=True)
    judge_scaler = base.StandardScaler(with_mean=True, with_std=True)
    X_emb_s = emb_scaler.fit_transform(X_emb)
    X_judge_s = judge_scaler.fit_transform(X_judge)

    X_t = base.np.concatenate([X_emb_s / math.sqrt(ae), X_judge_s / math.sqrt(aj)], axis=1)
    model = base.Ridge(alpha=1.0, fit_intercept=True, random_state=None)
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
    X_emb = base.np.asarray(X_emb, dtype=base.np.float64)
    X_judge = base.np.asarray(X_judge, dtype=base.np.float64)
    X_emb_s = state["emb_scaler"].transform(X_emb)
    X_judge_s = state["judge_scaler"].transform(X_judge)
    X_t = base.np.concatenate(
        [
            X_emb_s / math.sqrt(float(state["alpha_emb"])),
            X_judge_s / math.sqrt(float(state["alpha_judge"])),
        ],
        axis=1,
    )
    pred = state["ridge"].predict(X_t)
    return base.np.asarray(pred, dtype=base.np.float64).reshape(-1)


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
    """
    X_emb = base.np.asarray(X_emb, dtype=base.np.float64)
    X_judge = base.np.asarray(X_judge, dtype=base.np.float64)
    y = base.np.asarray(y, dtype=base.np.float64).reshape(-1)
    n = int(y.shape[0])
    k = int(min(int(inner_splits), max(2, n)))
    cv = base.KFold(n_splits=k, shuffle=True, random_state=int(seed))

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
                mse_sum += float(base.np.mean(err * err))
                n_folds += 1
            mse = mse_sum / max(1, n_folds)
            if mse < best[2]:
                best = (float(ae), float(aj), float(mse))
    if best[0] is None or best[1] is None:
        raise RuntimeError("Inner CV failed to select block alphas.")
    return float(best[0]), float(best[1]), float(best[2])


def _extract_block_ridge_raw_weights(state):
    """
    Return (w_emb_raw, w_judge_raw, intercept_raw) so that:
      yhat = intercept_raw + dot(w_emb_raw, x_emb_raw) + dot(w_judge_raw, x_judge_raw)
    """
    ridge = state["ridge"]
    coef = base.np.asarray(getattr(ridge, "coef_", []), dtype=base.np.float64).reshape(-1)
    if coef.size == 0:
        raise RuntimeError("Model has no coef_.")

    n_emb = int(state["n_emb"])
    w_emb_t = coef[:n_emb].reshape(-1)
    w_judge_t = coef[n_emb:].reshape(-1)

    alpha_emb = float(state["alpha_emb"])
    alpha_judge = float(state["alpha_judge"])
    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]

    emb_scale = base.np.asarray(getattr(emb_scaler, "scale_", None), dtype=base.np.float64).reshape(-1)
    judge_scale = base.np.asarray(getattr(judge_scaler, "scale_", None), dtype=base.np.float64).reshape(-1)
    emb_mean = base.np.asarray(getattr(emb_scaler, "mean_", None), dtype=base.np.float64).reshape(-1)
    judge_mean = base.np.asarray(getattr(judge_scaler, "mean_", None), dtype=base.np.float64).reshape(-1)
    emb_scale = base.np.where(emb_scale == 0.0, 1.0, emb_scale)
    judge_scale = base.np.where(judge_scale == 0.0, 1.0, judge_scale)

    w_emb_std = w_emb_t / math.sqrt(alpha_emb)
    w_judge_std = w_judge_t / math.sqrt(alpha_judge)
    w_emb_raw = w_emb_std / emb_scale
    w_judge_raw = w_judge_std / judge_scale

    intercept_model = float(getattr(ridge, "intercept_", 0.0))
    intercept_raw = intercept_model - float(base.np.dot(emb_mean, w_emb_raw)) - float(base.np.dot(judge_mean, w_judge_raw))
    return w_emb_raw.astype(base.np.float32, copy=False), w_judge_raw.astype(base.np.float32, copy=False), float(intercept_raw)


def save_regression_weights_block_ridge(
    *,
    out_dir: str,
    state,
    judge_feature_names: Sequence[str],
    metadata: dict,
) -> Tuple[str, str]:
    """
    Save a minimal representation of the joint block-ridge.

    Writes:
      - regression_weights.json (metadata)
      - regression_weights.npz  (arrays: coef_emb_raw, coef_judge_raw, intercept_raw, judge_feature_names, plus scaler stats)
    """
    base.ensure_dir(str(out_dir))
    w_emb_raw, w_judge_raw, intercept_raw = _extract_block_ridge_raw_weights(state)

    emb_scaler = state["emb_scaler"]
    judge_scaler = state["judge_scaler"]
    emb_mean = base.np.asarray(getattr(emb_scaler, "mean_", []), dtype=base.np.float32).reshape(-1)
    emb_scale = base.np.asarray(getattr(emb_scaler, "scale_", []), dtype=base.np.float32).reshape(-1)
    judge_mean = base.np.asarray(getattr(judge_scaler, "mean_", []), dtype=base.np.float32).reshape(-1)
    judge_scale = base.np.asarray(getattr(judge_scaler, "scale_", []), dtype=base.np.float32).reshape(-1)

    weights_npz = os.path.join(str(out_dir), "regression_weights.npz")
    base.np.savez_compressed(
        weights_npz,
        coef_emb_raw=base.np.asarray(w_emb_raw, dtype=base.np.float32).reshape(-1),
        coef_judge_raw=base.np.asarray(w_judge_raw, dtype=base.np.float32).reshape(-1),
        intercept_raw=base.np.asarray([float(intercept_raw)], dtype=base.np.float32),
        alpha_emb=base.np.asarray([float(state["alpha_emb"])], dtype=base.np.float32),
        alpha_judge=base.np.asarray([float(state["alpha_judge"])], dtype=base.np.float32),
        emb_scaler_mean=emb_mean,
        emb_scaler_scale=emb_scale,
        judge_scaler_mean=judge_mean,
        judge_scaler_scale=judge_scale,
        judge_feature_names=base.np.asarray(list(judge_feature_names), dtype=object),
        n_emb=base.np.asarray([int(state["n_emb"])], dtype=base.np.int64),
        n_judge=base.np.asarray([int(state["n_judge"])], dtype=base.np.int64),
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


def _is_swebench_pro_benchmark(*, dataset_name: str, dataset_path: str) -> bool:
    """
    Only SWE-bench Pro should be "treated as Pro" for subject_id parsing.

    Heuristic is intentionally strict: it only returns True for obviously Pro identifiers.
    """
    s = " ".join([str(dataset_name or ""), str(dataset_path or "")]).lower()
    return ("swe-bench_pro" in s) or ("swebench_pro" in s) or ("swe_bench_pro" in s)


def evaluate_ood_auroc(
    *,
    ood_agent_results_jsonl: str,
    ood_normalize_item_ids: bool,
    ood_treat_as_pro: bool,
    ood_default_scaffold: Optional[str],
    z_by_item: Dict[str, float],
    theta_by_model: Dict[str, float],
    theta_by_scaffold: Dict[str, float],
) -> Tuple[float, dict]:
    """
    Evaluate ROC-AUC on an out-of-distribution response matrix using:
      p(success) = sigmoid((theta_model + theta_scaffold) - z_item_pred)
    """
    filt = _import_swebench_irt_module("filter_subjects_by_scaffold_count")
    split_mod = _import_swebench_irt_module("split_agents_model_scaffold")

    scores: List[float] = []
    labels: List[int] = []
    n_subjects_total = 0
    n_subjects_used = 0
    n_subjects_used_model_only = 0
    n_subjects_used_assumed_scaffold = 0
    n_obs_total = 0
    n_obs_scored = 0
    n_obs_scored_model_only = 0
    n_obs_scored_assumed_scaffold = 0
    n_obs_skipped_no_theta = 0
    n_obs_skipped_no_item = 0
    n_obs_skipped_bad_score = 0

    for sid, resp in iter_subject_responses_jsonl_generic(
        str(ood_agent_results_jsonl), normalize_item_ids=bool(ood_normalize_item_ids)
    ):
        n_subjects_total += 1

        assume_scaffold = str(ood_default_scaffold or "").strip()

        # Prefer using the canonical (model, scaffold) parsing from the IRT scripts,
        # but fall back to "model-only" for OOD benchmarks that store subject_id as
        # just the base model name (e.g. GSO), where scaffold is unavailable.
        used_model_only = False
        used_assumed_scaffold = False
        th: Optional[float] = None

        # If caller requests a fixed scaffold (e.g. GSO => OpenHands), treat the subject_id
        # as a model string and canonicalize it using shared parsing rules.
        if assume_scaffold:
            try:
                model_canon = str(split_mod._canonical_model(str(sid)))  # type: ignore[attr-defined]
                scaffold_canon = str(split_mod._canonical_scaffold(str(assume_scaffold)))  # type: ignore[attr-defined]
            except Exception:
                model_canon = str(sid)
                scaffold_canon = str(assume_scaffold)

            tm = theta_by_model.get(model_canon, None)
            ts = theta_by_scaffold.get(scaffold_canon, None)
            if ts is None:
                raise ValueError(
                    f"OOD assumed scaffold {scaffold_canon!r} not found in theta_by_scaffold "
                    f"(available={sorted(list(theta_by_scaffold.keys()))[:50]} ...)"
                )
            if tm is not None:
                th = float(tm) + float(ts)
                used_assumed_scaffold = True

        # Otherwise try to parse subject_id as (model, scaffold).
        m = filt._model_for_subject(sid, treat_as_pro=bool(ood_treat_as_pro))  # type: ignore[attr-defined]
        sc = filt._scaffold_for_subject(sid, treat_as_pro=bool(ood_treat_as_pro))  # type: ignore[attr-defined]
        if th is None and m is not None and sc is not None:
            tm = theta_by_model.get(str(m), None)
            ts = theta_by_scaffold.get(str(sc), None)
            if tm is not None and ts is not None:
                th = float(tm) + float(ts)

        # Final fallback: if we couldn't parse scaffold, treat subject_id as a model string.
        # Canonicalize the model and (if possible) use OpenHands scaffold (GSO convention),
        # otherwise fall back to model-only.
        if th is None:
            try:
                model_canon = str(split_mod._canonical_model(str(sid)))  # type: ignore[attr-defined]
            except Exception:
                model_canon = str(sid)

            tm = theta_by_model.get(model_canon, None)
            if tm is not None:
                # Prefer OpenHands when it exists in the trained scaffold list.
                ts = theta_by_scaffold.get("OpenHands", None)
                if ts is not None:
                    th = float(tm) + float(ts)
                    used_assumed_scaffold = True
                else:
                    th = float(tm)
                    used_model_only = True

        if th is None:
            n_obs_skipped_no_theta += int(len(resp))
            n_obs_total += int(len(resp))
            continue

        n_subjects_used += 1
        if used_model_only:
            n_subjects_used_model_only += 1
        if used_assumed_scaffold:
            n_subjects_used_assumed_scaffold += 1
        for item_id, y_obs in resp.items():
            n_obs_total += 1
            z = z_by_item.get(str(item_id), None)
            if z is None:
                n_obs_skipped_no_item += 1
                continue
            s = _sigmoid(th - float(z))
            if not math.isfinite(float(s)):
                n_obs_skipped_bad_score += 1
                continue
            scores.append(float(s))
            labels.append(int(y_obs))
            n_obs_scored += 1
            if used_model_only:
                n_obs_scored_model_only += 1
            if used_assumed_scaffold:
                n_obs_scored_assumed_scaffold += 1

    auc = float(base._compute_binary_auroc(scores, labels))
    n_pos = int(sum(int(x) for x in labels))
    n_neg = int(len(labels) - n_pos)
    meta = {
        "subjects_total": int(n_subjects_total),
        "subjects_used": int(n_subjects_used),
        "subjects_used_model_only": int(n_subjects_used_model_only),
        "subjects_used_assumed_scaffold": int(n_subjects_used_assumed_scaffold),
        "items_predicted": int(len(z_by_item)),
        "obs_total": int(n_obs_total),
        "obs_scored": int(n_obs_scored),
        "obs_scored_model_only": int(n_obs_scored_model_only),
        "obs_scored_assumed_scaffold": int(n_obs_scored_assumed_scaffold),
        "obs_skipped_no_theta": int(n_obs_skipped_no_theta),
        "obs_skipped_no_item": int(n_obs_skipped_no_item),
        "obs_skipped_bad_score": int(n_obs_skipped_bad_score),
        "labels_pos": int(n_pos),
        "labels_neg": int(n_neg),
    }
    return auc, meta


_OOD_PROMPT_TEMPLATE = """I’ve uploaded a python code repository in the directory workspace_dir_name. Consider the
following test script showing an example usage of the repository:
<test_script>
{SPEC_TEST}
</test_script>
Can you help me implement the necessary changes to the repository so that the runtime of
the <test_script> is optimized? Basic guidelines:
1. Your task is to make changes to non-test files in the /workspace directory to improve the
performance of the <test_script>.
2. Make changes while ensuring the repository is functionally equivalent to the original.
3. Do not overoptimize for just the specific inputs in <test_script>. Make general perfor-
mance improvements for the usage scenario shown.
4. You may need to rebuild the repo for your changes to take effect before testing. Some
rebuilds may take time to run, so be patient with running them.
Follow these steps to improve performance:
1. As a first step, explore the repository structure.
2. Create a script in the /workspace directory (e.g., /workspace/test_opt.py) to reproduce and
time the example, then execute it with python /workspace/<filename.py>.
3. Edit the source code of the repository to improve performance.
4. Rebuild and rerun your script to confirm that performance has improved.
"""


def _wrap_ood_problem_statement(prob_script: str) -> str:
    return _OOD_PROMPT_TEMPLATE.format(SPEC_TEST=str(prob_script or "").strip())


def load_ood_items_by_ids(
    *,
    dataset_name: str,
    split: str,
    dataset_path: str,
    item_ids: Sequence[str],
    normalize_item_ids: bool,
) -> Tuple[List[base.ItemRecord], List[str]]:
    """
    Load OOD tasks where:
      - problem statement is in column `prob_script`
      - gold solution is in column `gt_diff`

    The problem statement is wrapped into the fixed "spec test" prompt template.
    """
    want = [base.normalize_swebench_item_id(x) for x in list(item_ids)] if bool(normalize_item_ids) else [str(x) for x in list(item_ids)]
    want = [x for x in want if str(x).strip()]
    want_set = set(want)
    if not want:
        return [], []

    dataset_name = str(dataset_name or "").strip()
    dataset_path = str(dataset_path or "").strip()
    if bool(dataset_name) and bool(dataset_path):
        raise ValueError("Provide only one of dataset_name or dataset_path (OOD mode).")
    if not dataset_name and not dataset_path:
        raise ValueError("No dataset provided for OOD (set dataset_name or dataset_path).")

    if dataset_path:
        ds = base.load_dataset("json", data_files=str(dataset_path), split="train")
    else:
        ds = base.load_dataset(str(dataset_name), split=str(split))

    n_total = int(len(ds))
    if n_total == 0:
        raise RuntimeError(f"Loaded empty OOD dataset: {dataset_name or dataset_path} split={split}")

    id_keys = ["instance_id", "task_id", "id"]
    found: Dict[str, base.ItemRecord] = {}
    for i in range(n_total):
        row = ds[int(i)]
        raw_id = ""
        for k in id_keys:
            v = row.get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                raw_id = s
                break
        if not raw_id:
            continue
        item_id = base.normalize_swebench_item_id(raw_id) if bool(normalize_item_ids) else str(raw_id).strip()
        if not item_id or item_id not in want_set or item_id in found:
            continue

        qs = _wrap_ood_problem_statement(str(row.get("prob_script", "") or ""))
        sol = str(row.get("gt_diff", "") or "")
        found[item_id] = base.ItemRecord(item_id=item_id, question_statement=qs, solution=sol)
        if len(found) >= len(want_set):
            break

    items = [found[tid] for tid in want if tid in found]
    missing = [tid for tid in want if tid not in found]
    return items, missing


def _import_swebench_irt_module(module_name: str):
    """
    Import a module from `fulcrum/fellowship/swebench_irt/` while preserving sibling
    imports (these scripts often import each other via bare names).
    """
    here = Path(__file__).resolve().parent
    swe_irt_dir = str(here / "swebench_irt")
    if swe_irt_dir not in sys.path:
        sys.path.insert(0, swe_irt_dir)
    return __import__(str(module_name))


def filter_subjects_by_min_models_per_scaffold(
    *,
    input_jsonl: str,
    output_jsonl: str,
    min_models_per_scaffold: int,
    treat_as_pro: bool,
) -> None:
    """
    Filter a response-matrix JSONL to remove rare scaffolds before downstream steps.

    Uses the splitting/canonicalization logic from
    `swebench_irt/filter_subjects_by_scaffold_count.py` by importing that module and
    calling its internal helper functions.

    - If `min_models_per_scaffold <= 0`, no filtering is applied (records are copied through).
    - Filtering is within-file (matches the referenced script's behavior).
    """
    in_path = str(input_jsonl or "").strip()
    out_path = str(output_jsonl or "").strip()
    k = int(min_models_per_scaffold)
    if not in_path:
        raise ValueError("input_jsonl is empty")
    if not out_path:
        raise ValueError("output_jsonl is empty")
    if k < 0:
        raise ValueError("--min_models_per_scaffold must be >= 0")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    base.ensure_dir(os.path.dirname(out_path) or ".")

    filt = _import_swebench_irt_module("filter_subjects_by_scaffold_count")

    # No-op mode (but still drop invalid rows).
    if k <= 0:
        kept = 0
        with open(out_path, "w", encoding="utf-8") as f_out:
            for r in _iter_jsonl(in_path):
                subj = str(r.get("subject_id", "") or "").strip()
                resp = r.get("responses", {}) or {}
                if not subj or not isinstance(resp, dict) or not resp:
                    continue
                f_out.write(json.dumps({"subject_id": subj, "responses": resp}) + "\n")
                kept += 1
        if kept <= 0:
            raise RuntimeError(f"Filtering produced 0 records (no-op path) for: {in_path}")
        return

    from collections import defaultdict

    scaffolds_to_models: Dict[str, Set[str]] = defaultdict(set)
    for r in _iter_jsonl(in_path):
        subj = str(r.get("subject_id", "") or "").strip()
        sc = filt._scaffold_for_subject(subj, treat_as_pro=bool(treat_as_pro))  # type: ignore[attr-defined]
        m = filt._model_for_subject(subj, treat_as_pro=bool(treat_as_pro))  # type: ignore[attr-defined]
        if sc is None or m is None:
            continue
        scaffolds_to_models[str(sc)].add(str(m))

    keep_scaffolds: Set[str] = set([sc for sc, ms in scaffolds_to_models.items() if len(ms) >= int(k)])

    kept = 0
    dropped_unsplittable = 0
    dropped_rare = 0
    total = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for r in _iter_jsonl(in_path):
            total += 1
            subj = str(r.get("subject_id", "") or "").strip()
            sc = filt._scaffold_for_subject(subj, treat_as_pro=bool(treat_as_pro))  # type: ignore[attr-defined]
            m = filt._model_for_subject(subj, treat_as_pro=bool(treat_as_pro))  # type: ignore[attr-defined]
            if sc is None or m is None:
                dropped_unsplittable += 1
                continue
            if str(sc) not in keep_scaffolds:
                dropped_rare += 1
                continue
            f_out.write(json.dumps(r) + "\n")
            kept += 1

    print(f"Filtered subjects: {in_path} -> {out_path}")
    print(
        f"min_models_per_scaffold={int(k)} treat_as_pro={bool(treat_as_pro)} total={int(total)} kept={int(kept)} "
        f"dropped_unsplittable={int(dropped_unsplittable)} dropped_rare={int(dropped_rare)} "
        f"scaffolds_kept={len(keep_scaffolds)}"
    )
    if kept <= 0:
        raise RuntimeError(
            f"Filtering produced 0 records. Try lowering --min_models_per_scaffold. input={in_path} output={out_path}"
        )


def normalize_responses_jsonl(
    *,
    in_path: str,
    out_path: str,
    benchmark: str,
) -> None:
    """
    Write a normalized response-matrix JSONL for use by shared IRT training.

    - Verified/Pro: normalize ids with `normalize_swebench_item_id`.
    - Terminal-Bench: preserve ids.
    """
    base.ensure_dir(os.path.dirname(out_path) or ".")
    b = str(benchmark or "").strip().lower()
    if b not in {"verified", "pro", "terminal_bench"}:
        raise ValueError(f"Unknown benchmark for normalization: {benchmark}")

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in _iter_jsonl(in_path):
            sid = str(obj.get("subject_id", "") or "").strip()
            resp = obj.get("responses", {}) or {}
            if not sid or not isinstance(resp, dict):
                continue
            out_resp: Dict[str, int] = {}
            for raw_id, v in resp.items():
                if b in {"verified", "pro"}:
                    tid = base.normalize_swebench_item_id(str(raw_id))
                else:
                    tid = str(raw_id or "").strip()
                if not tid:
                    continue
                try:
                    out_resp[tid] = int(v)
                except Exception:
                    out_resp[tid] = 1 if v else 0
            if out_resp:
                f.write(json.dumps({"subject_id": sid, "responses": out_resp}) + "\n")


def iter_terminal_bench_items_from_jsonl(*, path: str) -> Iterator[base.ItemRecord]:
    """
    Yield Terminal-Bench tasks from a SWE-bench-like JSONL:
      {"task_id": "...", "problem_statement": "...", "patch": "..."}

    IMPORTANT: Terminal-Bench task IDs must be preserved exactly (no normalization).
    """
    p = str(path or "").strip()
    if not p:
        return
    if not os.path.exists(p):
        raise FileNotFoundError(f"Terminal-Bench tasks JSONL not found: {p}")

    for obj in _iter_jsonl(p):
        tid = str(obj.get("task_id", "") or "").strip()
        if not tid:
            continue
        qs = str(obj.get("problem_statement", "") or "")
        sol = str(obj.get("patch", "") or "")
        yield base.ItemRecord(item_id=tid, question_statement=qs, solution=sol)


def _import_shared_irt_module():
    """
    Import `swebench_irt/train_model_scaffold_shared.py` in a way that preserves its
    sibling import `from split_agents_model_scaffold import ...`.
    """
    return _import_swebench_irt_module("train_model_scaffold_shared")


def build_multibench_obs_for_items(
    *,
    obs_full,
    keep_item_ids: Sequence[str],
):
    """
    Filter a `train_model_scaffold_shared.MultiBenchObs` to observations on `keep_item_ids`,
    pruning unused models/scaffolds and remapping indices densely.
    """
    import torch

    keep: List[str] = [str(x) for x in keep_item_ids if str(x).strip()]
    keep_set = set(keep)
    if not keep:
        raise ValueError("keep_item_ids was empty")

    # Full item vocabulary -> indices.
    full_item_ids: List[str] = list(obs_full.item_ids)
    full_item_to_idx: Dict[str, int] = {iid: i for i, iid in enumerate(full_item_ids)}
    keep_idxs = [full_item_to_idx[iid] for iid in keep if iid in full_item_to_idx]
    keep_idx_set = set(keep_idxs)
    if not keep_idxs:
        raise RuntimeError("No keep_item_ids were present in obs_full.item_ids (unexpected).")

    mask_items = torch.zeros((len(full_item_ids),), dtype=torch.bool)
    mask_items[torch.tensor(sorted(keep_idx_set), dtype=torch.long)] = True
    mask_obs = mask_items[obs_full.item_idx]

    m_old = obs_full.model_idx[mask_obs]
    s_old = obs_full.scaffold_idx[mask_obs]
    i_old = obs_full.item_idx[mask_obs]
    y = obs_full.y[mask_obs]
    if int(y.numel()) == 0:
        raise RuntimeError("After filtering to keep_item_ids, there were 0 observations.")

    # Remap items.
    item_map = torch.full((len(full_item_ids),), -1, dtype=torch.long)
    keep_full_idxs_sorted = torch.tensor(sorted(keep_idx_set), dtype=torch.long)
    item_map[keep_full_idxs_sorted] = torch.arange(int(keep_full_idxs_sorted.numel()), dtype=torch.long)
    i_new = item_map[i_old]

    # Prune/remap models and scaffolds to only those observed.
    used_m = sorted(set(int(x) for x in m_old.detach().cpu().tolist()))
    used_s = sorted(set(int(x) for x in s_old.detach().cpu().tolist()))

    model_map = torch.full((len(obs_full.model_ids),), -1, dtype=torch.long)
    scaffold_map = torch.full((len(obs_full.scaffold_ids),), -1, dtype=torch.long)
    model_map[torch.tensor(used_m, dtype=torch.long)] = torch.arange(len(used_m), dtype=torch.long)
    scaffold_map[torch.tensor(used_s, dtype=torch.long)] = torch.arange(len(used_s), dtype=torch.long)
    m_new = model_map[m_old]
    s_new = scaffold_map[s_old]

    # Subset vocab lists.
    model_ids = [str(obs_full.model_ids[i]) for i in used_m]
    scaffold_ids = [str(obs_full.scaffold_ids[i]) for i in used_s]
    item_ids = [full_item_ids[i] for i in sorted(keep_idx_set)]

    ms = _import_shared_irt_module()
    return ms.MultiBenchObs(
        model_idx=m_new,
        scaffold_idx=s_new,
        item_idx=i_new,
        y=y,
        model_ids=model_ids,
        scaffold_ids=scaffold_ids,
        item_ids=item_ids,
        verified_item_ids=set([iid for iid in item_ids if iid in set(obs_full.verified_item_ids)]),
        pro_item_ids=set([iid for iid in item_ids if iid in set(obs_full.pro_item_ids)]),
        terminal_bench_item_ids=set([iid for iid in item_ids if iid in set(obs_full.terminal_bench_item_ids)]),
        agent_split_df=obs_full.agent_split_df,
    )


def train_irt_model_scaffold_1pl(
    *,
    obs_train,
    irt_model: str,
    epochs: int,
    device: str,
    seed: int,
    lr: float,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Train shared model+scaffold 1PL (Rasch) on `obs_train`.

    Supports:
      - 1D 1PL: p = sigmoid((theta_model + theta_scaffold) - b_item)
      - 2D 1PL: p = sigmoid(sum_d (theta_model[d] + theta_scaffold[d] - b_item[d]))

    Returns:
      - theta_by_model
      - theta_by_scaffold
      - diff_by_item (b)
    """
    base._require("pyro")
    import torch
    import pyro  # type: ignore

    ms = _import_shared_irt_module()

    dev = str(device or "cpu").strip() or "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: --irt_device=cuda requested but CUDA is unavailable; falling back to cpu for IRT.")
        dev = "cpu"
    torch_device = torch.device(dev)

    # Seed + deterministic policy mirrors the single-benchmark script:
    # seeded RNG, but torch determinism disabled during IRT for stability.
    try:
        os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    except Exception:
        pass
    ms.set_seed(int(seed))
    pyro.clear_param_store()

    obs = obs_train
    # Move tensors to target device for SVI.
    obs_dev = ms.MultiBenchObs(
        model_idx=obs.model_idx.to(torch_device),
        scaffold_idx=obs.scaffold_idx.to(torch_device),
        item_idx=obs.item_idx.to(torch_device),
        y=obs.y.to(torch_device),
        model_ids=list(obs.model_ids),
        scaffold_ids=list(obs.scaffold_ids),
        item_ids=list(obs.item_ids),
        verified_item_ids=set(obs.verified_item_ids),
        pro_item_ids=set(obs.pro_item_ids),
        terminal_bench_item_ids=set(obs.terminal_bench_item_ids),
        agent_split_df=obs.agent_split_df,
    )

    irt_model_norm = str(irt_model or "1d_1pl").strip().lower()
    if irt_model_norm == "1d_1pl":
        model_obj = ms.ModelScaffold1PL(len(obs_dev.model_ids), len(obs_dev.scaffold_ids), len(obs_dev.item_ids))
        model_type = "1pl"
    elif irt_model_norm == "2d_1pl":
        model_obj = ms.ModelScaffold2D1PL(len(obs_dev.model_ids), len(obs_dev.scaffold_ids), len(obs_dev.item_ids), dims=2)
        model_type = "2d_1pl"
    else:
        raise ValueError(f"Unknown IRT model: {irt_model!r} (expected '1d_1pl' or '2d_1pl').")

    _ = ms.train_svi(model_obj.model, model_obj.guide, obs_dev, epochs=int(epochs), lr=float(lr))

    # Save fold artifacts in the same spirit as the single-benchmark script.
    outp = Path(str(out_dir))
    outp.mkdir(parents=True, exist_ok=True)
    ms.save_outputs(out_dir=outp, obs=obs_dev, model_type=model_type)
    try:
        obs_dev.agent_split_df.to_csv(outp / "agent_splits.csv", index=False)
    except Exception:
        pass

    # Extract centered abilities + item difficulties.
    #
    # Note: downstream scoring code expects *scalar* thetas and item difficulties.
    # For 2D 1PL we use the equal-weight sum over dimensions, matching the model
    # definition in `swebench_irt/train_model_scaffold_shared.py`.
    theta_m_raw = pyro.param("loc_theta_model_raw").detach().cpu()
    theta_s_raw = pyro.param("loc_theta_scaffold_raw").detach().cpu()
    if theta_m_raw.ndim == 1:
        theta_m_vec = theta_m_raw - theta_m_raw.mean()
    elif theta_m_raw.ndim == 2:
        theta_m_vec = (theta_m_raw - theta_m_raw.mean(dim=0, keepdim=True)).sum(dim=1)
    else:
        raise ValueError(f"Unexpected loc_theta_model_raw ndim={int(theta_m_raw.ndim)} for IRT model {irt_model_norm!r}")

    if theta_s_raw.ndim == 1:
        theta_s_vec = theta_s_raw - theta_s_raw.mean()
    elif theta_s_raw.ndim == 2:
        theta_s_vec = (theta_s_raw - theta_s_raw.mean(dim=0, keepdim=True)).sum(dim=1)
    else:
        raise ValueError(f"Unexpected loc_theta_scaffold_raw ndim={int(theta_s_raw.ndim)} for IRT model {irt_model_norm!r}")

    b_loc = pyro.param("loc_b").detach().cpu()
    if b_loc.ndim == 1:
        b_vec = b_loc
    elif b_loc.ndim == 2:
        b_vec = b_loc.sum(dim=1)
    else:
        raise ValueError(f"Unexpected loc_b ndim={int(b_loc.ndim)} for IRT model {irt_model_norm!r}")

    theta_m = theta_m_vec.numpy().tolist()
    theta_s = theta_s_vec.numpy().tolist()
    b_out = b_vec.numpy().tolist()

    theta_by_model: Dict[str, float] = {str(mid): float(theta_m[i]) for i, mid in enumerate(obs_dev.model_ids)}
    theta_by_scaffold: Dict[str, float] = {str(sid): float(theta_s[i]) for i, sid in enumerate(obs_dev.scaffold_ids)}
    diff_by_item: Dict[str, float] = {str(iid): float(b_out[i]) for i, iid in enumerate(obs_dev.item_ids)}
    return theta_by_model, theta_by_scaffold, diff_by_item


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # -----------------------------
    # Global settings
    # -----------------------------
    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/multi_benchmark_ood")
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--min_models_per_scaffold",
        type=int,
        default=2,
        help=(
            "Filter subjects to scaffolds with >= this many distinct base models (within each benchmark JSONL) "
            "before embeddings + ridge + IRT. Set to 0 to disable."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include_zero_success",
        action="store_true",
        help="Include items with 0 successes in CV/IRT (not recommended; can destabilize IRT).",
    )

    # -----------------------------
    # Multi-benchmark item sources
    # -----------------------------
    p.add_argument("--verified_dataset_name", type=str, default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--verified_dataset_path", type=str, default="")
    p.add_argument("--verified_split", type=str, default="test")
    p.add_argument("--pro_dataset_name", type=str, default="ScaleAI/SWE-bench_Pro")
    p.add_argument("--pro_dataset_path", type=str, default="")
    p.add_argument("--pro_split", type=str, default="test")
    p.add_argument(
        "--terminal_bench_tasks_jsonl",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/terminal_bench_tasks.jsonl",
        help="Terminal-Bench tasks JSONL with fields: task_id, problem_statement, patch.",
    )

    # -----------------------------
    # IRT model settings
    # -----------------------------
    p.add_argument("--irt_epochs", type=int, default=5000)
    p.add_argument("--irt_device", type=str, default="cuda", help="Device for IRT training (cuda or cpu).")
    p.add_argument("--irt_lr", type=float, default=0.01, help="Learning rate for Pyro SVI (shared model+scaffold IRT).")
    p.add_argument(
        "--irt_model",
        type=str,
        default="1d_1pl",
        choices=["1d_1pl", "2d_1pl"],
        help="IRT model family for training. Default is 1D 1PL (Rasch).",
    )

    # -----------------------------
    # Embedding model settings
    # -----------------------------
    p.add_argument("--backbone", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto", help="HF device_map (e.g. auto). Use 'none' to force single-device .to(device).")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="auto", help="e.g. auto, flash_attention_2")
    p.add_argument(
        "--embedding_layer",
        type=int,
        default=-1,
        help="Which hidden layer to pool embeddings from (0-based over returned hidden_states; negatives allowed). -1 means last.",
    )
    p.add_argument("--instruction", type=str, default=base.DIFFICULTY_INSTRUCTION, help="Instruction text appended last in the embedding input.")

    # -----------------------------
    # Regressor settings
    # -----------------------------
    p.add_argument(
        "--regressor",
        type=str,
        default="ridge_cv",
        choices=["linear", "ridge", "ridge_cv"],
        help="Regression model (same options as single-benchmark script).",
    )
    p.add_argument("--ridge_alpha", type=float, default=10000.0)
    p.add_argument("--ridge_alphas", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument(
        "--inner_splits",
        type=int,
        default=5,
        help="Inner CV splits for RidgeCV (used when --regressor=ridge_cv). Will be capped by train size; must be >=2.",
    )

    # -----------------------------
    # Optional joint ridge training (embeddings + judge features)
    # -----------------------------
    p.add_argument(
        "--include_judge",
        action="store_true",
        help=(
            "Train a joint (block) ridge model over [embeddings, LLM-judge features] with separate ridge penalties "
            "for the embedding vs judge blocks. If unset, preserves historical embeddings-only behavior."
        ),
    )
    p.add_argument(
        "--verified_judge_features_dir",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/llm_judge/features/verified",
        help="Directory with per-item Verified judge feature JSONs (<item_id>.json).",
    )
    p.add_argument(
        "--pro_judge_features_dir",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/llm_judge/features/pro",
        help="Directory with per-item Pro judge feature JSONs (often instance_<id>-v... .json).",
    )
    p.add_argument(
        "--terminal_bench_judge_features_dir",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/llm_judge/features/terminal_bench",
        help="Directory with per-item Terminal-Bench judge feature JSONs (<task_id>.json).",
    )
    p.add_argument(
        "--ridge_alpha_emb",
        type=float,
        default=float("nan"),
        help=(
            "Embedding block ridge alpha (only used when --include_judge and --regressor=ridge). "
            "Defaults to --ridge_alpha when unset."
        ),
    )
    p.add_argument(
        "--ridge_alpha_judge",
        type=float,
        default=float("nan"),
        help=(
            "Judge block ridge alpha (only used when --include_judge and --regressor=ridge). "
            "Defaults to --ridge_alpha when unset."
        ),
    )
    p.add_argument(
        "--ridge_alphas_emb",
        type=str,
        default="",
        help=(
            "Embedding alpha grid for inner CV (only used when --include_judge and --regressor=ridge_cv). "
            "Defaults to --ridge_alphas when unset."
        ),
    )
    p.add_argument(
        "--ridge_alphas_judge",
        type=str,
        default="",
        help=(
            "Judge alpha grid for inner CV (only used when --include_judge and --regressor=ridge_cv). "
            "Defaults to --ridge_alphas when unset."
        ),
    )

    p.add_argument(
        "--eval_mode",
        type=str,
        default="ood",
        choices=["id", "ood"],
        help=(
            "Evaluation mode. ID runs the existing in-distribution K-fold CV + AUROC pipeline (previously "
            "--evaluate_in_distribution). OOD (default) runs the default training-only flow, then additionally "
            "evaluates AUROC on a 4th benchmark specified by --ood_* flags using the learned shared IRT abilities "
            "and regressor weights."
        ),
    )

    # -----------------------------
    # ID eval
    # -----------------------------
    p.add_argument(
        "--verified_agent_results",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl",
        help="Verified response-matrix JSONL: {'subject_id': ..., 'responses': {'item_id': 0/1, ...}}",
    )
    p.add_argument(
        "--pro_agent_results",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_pro.jsonl",
        help="Pro response-matrix JSONL: {'subject_id': ..., 'responses': {'item_id': 0/1, ...}}",
    )
    p.add_argument(
        "--terminal_bench_agent_results",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/terminal_bench_2.0.jsonl",
        help="Terminal-Bench response-matrix JSONL: {'subject_id': ..., 'responses': {'task_id': 0/1, ...}}",
    )

    # -----------------------------
    # OOD eval
    # -----------------------------
    p.add_argument(
        "--ood_dataset_name",
        type=str,
        default="gso-bench/gso",
        help=(
            "HF dataset repo for the 4th (OOD) benchmark tasks. Ignored if --ood_dataset_path is set. "
            "Used only when --eval_mode=ood."
        ),
    )
    p.add_argument(
        "--ood_dataset_path",
        type=str,
        default="",
        help=(
            "Optional local JSON/JSONL dataset path for the 4th (OOD) benchmark tasks. If set, loads via "
            "datasets('json', data_files=..., split='train'). Used only when --eval_mode=ood."
        ),
    )
    p.add_argument("--ood_split", type=str, default="test", help="Split name for --ood_dataset_name (OOD mode only).")
    p.add_argument(
        "--ood_agent_results",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/gso.jsonl",
        help="OOD response-matrix JSONL: {'subject_id': ..., 'responses': {'item_id': 0/1, ...}} (OOD mode only).",
    )
    p.add_argument(
        "--ood_judge_features_dir",
        type=str,
        default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/llm_judge/features/gso",
        help=(
            "Optional directory with per-item OOD judge feature JSONs (<item_id>.json). "
            "Assumed to use one of the existing judge schemas (Verified/Pro/Terminal-Bench). "
            "Used only when --eval_mode=ood and --include_judge."
        ),
    )
    p.add_argument(
        "--ood_no_normalize_item_ids",
        dest="ood_normalize_item_ids",
        action="store_false",
        default=True,
        help="Disable normalization of OOD item ids (default is to normalize). Useful for Terminal-Bench-style ids.",
    )
    
    
    args = p.parse_args(argv)
    base.ensure_dir(args.out_dir)
    base.seed_everything(int(args.seed), deterministic=True)

    irt_model = str(args.irt_model or "1d_1pl").strip().lower()
    if irt_model not in {"1d_1pl", "2d_1pl"}:
        raise ValueError(f"Unknown --irt_model: {args.irt_model!r}")

    def _irt_out_dir_name(model_name: str) -> str:
        m = str(model_name or "").strip().lower()
        if m == "1d_1pl":
            # Preserve historical output directory for default behavior.
            return "irt_model_scaffold_1pl"
        if m == "2d_1pl":
            return "irt_model_scaffold_2d_1pl"
        return f"irt_model_scaffold_{m}"

    irt_model_label = f"model+scaffold {irt_model.replace('_', ' ')} (shared)"

    verified_agent_results_raw = str(args.verified_agent_results)
    pro_agent_results_raw = str(args.pro_agent_results)
    terminal_agent_results_raw = str(args.terminal_bench_agent_results)

    # -----------------------------
    # Filter subjects + normalize responses (no persistent dirs)
    # -----------------------------
    #
    # User request: do NOT write `filtered_subjects/` or `normalized_inputs/`.
    # We therefore write intermediate JSONLs into a temporary directory that is
    # cleaned up automatically after the run.
    tmp = tempfile.TemporaryDirectory(prefix="multibench_tmp_")
    tmp_dir = tmp.name

    # Filtered response matrices (used for overlap/eligible items + evaluation).
    verified_agent_results = os.path.join(tmp_dir, "verified.filtered.jsonl")
    pro_agent_results = os.path.join(tmp_dir, "pro.filtered.jsonl")
    terminal_agent_results = os.path.join(tmp_dir, "terminal_bench.filtered.jsonl")

    filter_subjects_by_min_models_per_scaffold(
        input_jsonl=verified_agent_results_raw,
        output_jsonl=verified_agent_results,
        min_models_per_scaffold=int(args.min_models_per_scaffold),
        treat_as_pro=False,
    )
    filter_subjects_by_min_models_per_scaffold(
        input_jsonl=pro_agent_results_raw,
        output_jsonl=pro_agent_results,
        min_models_per_scaffold=int(args.min_models_per_scaffold),
        treat_as_pro=True,
    )
    if terminal_agent_results_raw.strip() and os.path.exists(terminal_agent_results_raw):
        filter_subjects_by_min_models_per_scaffold(
            input_jsonl=terminal_agent_results_raw,
            output_jsonl=terminal_agent_results,
            min_models_per_scaffold=int(args.min_models_per_scaffold),
            treat_as_pro=False,
        )
    else:
        terminal_agent_results = ""

    # -----------------------------
    # Embeddings cache key (multi-source)
    # -----------------------------
    verified_src = f"verified:{('json:' + os.path.basename(str(args.verified_dataset_path))) if str(args.verified_dataset_path).strip() else str(args.verified_dataset_name)}:{str(args.verified_split)}"
    pro_src = f"pro:{('json:' + os.path.basename(str(args.pro_dataset_path))) if str(args.pro_dataset_path).strip() else str(args.pro_dataset_name)}:{str(args.pro_split)}"
    terminal_src = f"terminal_jsonl:{os.path.basename(str(args.terminal_bench_tasks_jsonl)) or 'terminal_bench_tasks.jsonl'}"
    dataset_sources_str = " | ".join([verified_src, pro_src, terminal_src]) + f" | min_models_per_scaffold={int(args.min_models_per_scaffold)}"

    safe_backbone = str(args.backbone).replace("/", "__")
    ds_flag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dataset_sources_str))[:96]
    instr_sig = base.prompt_signature(str(args.instruction))
    layer_flag = "" if int(args.embedding_layer) == -1 else f"__layer{int(args.embedding_layer)}"
    idnorm_flag = "__idnorm_multibench"
    emb_cache = str(args.embeddings_cache or "").strip()
    if not emb_cache:
        emb_cache = os.path.join(
            args.out_dir,
            f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__qs-sol-instr__{instr_sig}{idnorm_flag}__{ds_flag}__maxlen{int(args.max_length)}.npz",
        )

    # -----------------------------
    # Load or compute embeddings (pool items from 3 benchmarks)
    # -----------------------------
    cache_exists = bool(os.path.exists(emb_cache))
    if str(args.embeddings_cache or "").strip() and not cache_exists and not bool(args.overwrite):
        print(
            f"WARNING: --embeddings_cache was provided but file does not exist: {emb_cache} "
            f"(cwd={os.getcwd()}). Will recompute embeddings."
        )

    if os.path.exists(emb_cache) and not args.overwrite:
        data = base.np.load(emb_cache, allow_pickle=True)
        task_ids = [str(x) for x in list(data["task_ids"].tolist())]
        X = data["X"].astype(base.np.float32)
        counts_kind = str(base._npz_scalar(data.get("counts_kind", None), "")) if "counts_kind" in data else ""
        cached_layer = int(base._npz_scalar(data.get("embedding_layer", None), -1)) if "embedding_layer" in data else -1
        if int(args.embedding_layer) != int(cached_layer):
            raise RuntimeError(
                f"Embeddings cache was created with embedding_layer={cached_layer}, but you requested "
                f"--embedding_layer={int(args.embedding_layer)}. Use --overwrite, or pick a different cache file."
            )
        print(
            f"Loaded embeddings cache: {emb_cache} (n={len(task_ids)}, dim={X.shape[1]}, counts_kind={counts_kind or 'unknown'}, embedding_layer={cached_layer})"
        )
    else:
        items: List[base.ItemRecord] = []

        # SWE-bench Verified items (normalize ids via base.iter_swebench_items).
        items.extend(
            list(
                base.iter_swebench_items(
                    dataset_name=str(args.verified_dataset_name),
                    split=str(args.verified_split),
                    dataset_path=str(args.verified_dataset_path),
                )
            )
        )

        # SWE-bench Pro items (normalize ids via base.iter_swebench_items).
        items.extend(
            list(
                base.iter_swebench_items(
                    dataset_name=str(args.pro_dataset_name),
                    split=str(args.pro_split),
                    dataset_path=str(args.pro_dataset_path),
                )
            )
        )

        # Terminal-Bench items (do NOT normalize ids).
        items.extend(list(iter_terminal_bench_items_from_jsonl(path=str(args.terminal_bench_tasks_jsonl))))

        # Deduplicate by id (keep first; warn on collisions).
        by_id: Dict[str, base.ItemRecord] = {}
        collisions: List[str] = []
        for it in items:
            iid = str(it.item_id)
            if iid in by_id:
                collisions.append(iid)
                continue
            by_id[iid] = it
        if collisions:
            print(f"WARNING: {len(collisions)} duplicate item_ids across benchmarks; keeping first occurrence. Example: {collisions[:10]}")
        items = list(by_id.values())

        print(f"Loaded dataset items: {len(items)} (sources={dataset_sources_str})")

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

        X = base.np.stack([emb_by_id[r] for r in ids_sorted], axis=0).astype(base.np.float32)
        counts_arr = base.np.array([int(counts_by_id.get(r, 0)) for r in ids_sorted], dtype=base.np.int64)

        base.np.savez_compressed(
            emb_cache,
            task_ids=base.np.array(ids_sorted, dtype=object),
            X=X,
            counts_kind=base.np.array(["text_len_chars"], dtype=object),
            counts=counts_arr,
            dataset_name=base.np.array([str(dataset_sources_str)], dtype=object),
            embedding_layer=base.np.array([int(args.embedding_layer)], dtype=base.np.int64),
        )
        print(f"Saved embeddings cache: {emb_cache} (n={len(ids_sorted)}, dim={emb_dim})")
        task_ids = list(ids_sorted)

    id_to_row: Dict[str, int] = {tid: int(i) for i, tid in enumerate(task_ids)}

    # -----------------------------
    # Load responses (multi-benchmark)
    # -----------------------------
    verified_responses = base.load_all_responses(str(verified_agent_results))
    pro_responses = base.load_all_responses(str(pro_agent_results))
    terminal_responses: List[Tuple[str, Dict[str, int]]] = []
    if str(terminal_agent_results).strip() and os.path.exists(str(terminal_agent_results)):
        terminal_responses = load_all_responses_terminal(str(terminal_agent_results))
    else:
        if str(args.terminal_bench_agent_results).strip():
            print(f"WARNING: terminal_bench_agent_results not found, skipping: {args.terminal_bench_agent_results}")

    # Combined for zero-success checks and overlap.
    all_responses_tagged: List[Tuple[str, str, Dict[str, int]]] = []
    all_responses_tagged.extend([("verified", sid, resp) for sid, resp in verified_responses])
    all_responses_tagged.extend([("pro", sid, resp) for sid, resp in pro_responses])
    all_responses_tagged.extend([("terminal_bench", sid, resp) for sid, resp in terminal_responses])

    response_items: Set[str] = set()
    for _, _, resp in all_responses_tagged:
        response_items.update(resp.keys())

    overlap_ids = [tid for tid in task_ids if tid in response_items]
    if not overlap_ids:
        raise RuntimeError("No overlap between embedded task_ids and item_ids found in the provided responses.")

    # Zero-success items across the full multi-benchmark response pool.
    # (We reuse the same helper, with a lightly-adapted input.)
    flat_for_zero_success: List[Tuple[str, Dict[str, int]]] = [(f"{b}::{sid}", resp) for b, sid, resp in all_responses_tagged]
    zero_success_ids = base.compute_zero_success_items(flat_for_zero_success)
    zero_success_set = set(zero_success_ids)
    exclude_zero_success = not bool(args.include_zero_success)
    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items from CV/IRT: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} overlapped items "
            f"(verified_agent_results={verified_agent_results}, pro_agent_results={pro_agent_results}, terminal_agent_results={terminal_agent_results})"
        )
    else:
        eligible = list(overlap_ids)
    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV/IRT.")

    Xy = base.np.stack([X[id_to_row[tid]] for tid in eligible], axis=0).astype(base.np.float32)

    # -----------------------------
    # Prepare shared IRT obs + agent decomposition
    # -----------------------------
    verified_norm = os.path.join(tmp_dir, "verified.normalized.jsonl")
    pro_norm = os.path.join(tmp_dir, "pro.normalized.jsonl")
    terminal_norm = os.path.join(tmp_dir, "terminal_bench.normalized.jsonl")

    normalize_responses_jsonl(in_path=str(verified_agent_results), out_path=verified_norm, benchmark="verified")
    normalize_responses_jsonl(in_path=str(pro_agent_results), out_path=pro_norm, benchmark="pro")
    term_path_for_irt: Optional[str] = None
    if terminal_responses and str(terminal_agent_results).strip():
        normalize_responses_jsonl(in_path=str(terminal_agent_results), out_path=terminal_norm, benchmark="terminal_bench")
        term_path_for_irt = terminal_norm

    ms = _import_shared_irt_module()
    obs_full = ms.load_multibench_split_irt_data(
        verified_path=ms.resolve_path(verified_norm),
        pro_path=ms.resolve_path(pro_norm),
        terminal_bench_path=ms.resolve_path(term_path_for_irt) if term_path_for_irt else None,
    )

    agent_to_ms_pair: Dict[str, Tuple[str, str]] = {}
    try:
        # agent_split_df includes rows for all included agents (after split filtering).
        for row in obs_full.agent_split_df.to_dict(orient="records"):
            bench = str(row.get("benchmark", "") or "").strip()
            agent = str(row.get("agent", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            scaffold = str(row.get("scaffold", "") or "").strip()
            if bench and agent and model and scaffold:
                agent_to_ms_pair[f"{bench}::{agent}"] = (model, scaffold)
    except Exception:
        agent_to_ms_pair = {}

    # -----------------------------
    # Regression model factory (identical)
    # -----------------------------
    regressor_name = str(args.regressor)
    alphas: base.np.ndarray = base.np.array([], dtype=base.np.float64)

    def _make_model(*, n_train: int, fold_seed: int):
        nonlocal alphas
        if regressor_name == "linear":
            return base.LinearRegression()
        if regressor_name == "ridge":
            alpha = float(args.ridge_alpha)
            if not (alpha > 0):
                raise ValueError("--ridge_alpha must be > 0")
            return base.Pipeline(
                steps=[("scaler", base.StandardScaler(with_mean=True, with_std=True)), ("ridge", base.Ridge(alpha=alpha))]
            )
        if regressor_name == "ridge_cv":
            try:
                alphas = base.np.array([float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()], dtype=base.np.float64)
            except Exception as e:
                raise ValueError(f"Failed to parse --ridge_alphas={args.ridge_alphas!r}: {e}") from e
            if alphas.size == 0:
                raise ValueError("Expected at least one alpha in --ridge_alphas")
            req_inner = int(args.inner_splits)
            if req_inner < 2:
                raise ValueError("--inner_splits must be >= 2")
            inner_splits = int(min(req_inner, max(2, int(n_train))))
            inner_cv = base.KFold(n_splits=int(inner_splits), shuffle=True, random_state=int(fold_seed))
            return base.Pipeline(
                steps=[
                    ("scaler", base.StandardScaler(with_mean=True, with_std=True)),
                    ("ridge", base.RidgeCV(alphas=alphas, cv=inner_cv)),
                ]
            )
        raise AssertionError(f"Unhandled regressor: {regressor_name}")

    use_joint = bool(getattr(args, "include_judge", False))
    if use_joint and str(regressor_name) not in {"ridge", "ridge_cv"}:
        raise ValueError("--include_judge requires --regressor to be ridge or ridge_cv (linear is not supported).")

    # Normalize early; argparse choices are lowercase.
    eval_mode = str(args.eval_mode or "ood").strip().lower()
    if eval_mode not in {"id", "ood"}:
        raise ValueError(f"Unknown --eval_mode: {args.eval_mode!r}")

    if eval_mode == "id" and use_joint:
        # -----------------------------
        # K-fold CV over items (joint block-ridge: embeddings + judge features)
        # -----------------------------
        outer_cv = base.KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        cv_test_auc_folds: List[float] = []
        cv_test_auc_folds_embedding_only: List[float] = []
        cv_test_n_obs_folds: List[int] = []
        cv_test_n_items_scored_folds: List[int] = []
        yhat_oof = base.np.full((int(len(eligible)),), base.np.nan, dtype=base.np.float64)
        fold_of_item = base.np.full((int(len(eligible)),), -1, dtype=base.np.int32)

        eligible_index = {tid: i for i, tid in enumerate(eligible)}

        verified_item_set = set(obs_full.verified_item_ids)
        pro_item_set = set(obs_full.pro_item_ids)
        terminal_item_set = set(obs_full.terminal_bench_item_ids)

        # Judge feature vector is a fixed concat of per-benchmark schemas.
        verified_off = 0
        pro_off = verified_off + int(len(VERIFIED_JUDGE_FEATURE_NAMES))
        terminal_off = pro_off + int(len(PRO_JUDGE_FEATURE_NAMES))
        judge_dim = terminal_off + int(len(TERMINAL_BENCH_JUDGE_FEATURE_NAMES))
        judge_feature_names_full: List[str] = (
            [f"verified::{k}" for k in VERIFIED_JUDGE_FEATURE_NAMES]
            + [f"pro::{k}" for k in PRO_JUDGE_FEATURE_NAMES]
            + [f"terminal_bench::{k}" for k in TERMINAL_BENCH_JUDGE_FEATURE_NAMES]
        )

        verified_feat_dir = str(args.verified_judge_features_dir)
        pro_feat_dir = str(args.pro_judge_features_dir)
        terminal_feat_dir = str(args.terminal_bench_judge_features_dir)
        verified_idx = _build_judge_index(verified_feat_dir, normalize_item_ids=True)
        pro_idx = _build_judge_index(pro_feat_dir, normalize_item_ids=True)
        terminal_idx = _build_judge_index(terminal_feat_dir, normalize_item_ids=False)

        def _judge_full_vec_for_item(item_id: str):
            tid = str(item_id)
            x = base.np.zeros((int(judge_dim),), dtype=base.np.float32)
            if tid in verified_item_set:
                v = _load_judge_vector(
                    tid,
                    features_dir=verified_feat_dir,
                    feature_names=VERIFIED_JUDGE_FEATURE_NAMES,
                    index=verified_idx,
                    normalize_item_ids=True,
                )
                if v is None:
                    return None
                x[verified_off:pro_off] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                return x
            if tid in pro_item_set:
                v = _load_judge_vector(
                    tid,
                    features_dir=pro_feat_dir,
                    feature_names=PRO_JUDGE_FEATURE_NAMES,
                    index=pro_idx,
                    normalize_item_ids=True,
                )
                if v is None:
                    return None
                x[pro_off:terminal_off] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                return x
            if tid in terminal_item_set:
                v = _load_judge_vector(
                    tid,
                    features_dir=terminal_feat_dir,
                    feature_names=TERMINAL_BENCH_JUDGE_FEATURE_NAMES,
                    index=terminal_idx,
                    normalize_item_ids=False,
                )
                if v is None:
                    return None
                x[terminal_off:] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                return x
            return None

        best_fold_auc = -float("inf")
        best_fold = -1
        best_joint_state = None
        fold_alpha_emb: List[float] = []
        fold_alpha_judge: List[float] = []

        for fold, (tr, te) in enumerate(outer_cv.split(Xy), start=1):
            train_items = [eligible[int(i)] for i in tr.tolist()]
            test_items = [eligible[int(i)] for i in te.tolist()]

            fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
            base.ensure_dir(fold_root)

            # Save train/test item lists (debugging / provenance).
            base.save_json(os.path.join(fold_root, "train_items.json"), {"items": list(train_items)})
            base.save_json(os.path.join(fold_root, "test_items.json"), {"items": list(test_items)})

            # IRT on train items only (no leakage).
            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)

            obs_train = build_multibench_obs_for_items(obs_full=obs_full, keep_item_ids=train_items)
            theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
                obs_train=obs_train,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, _irt_out_dir_name(irt_model)),
            )

            # Restore determinism for downstream sklearn/regression steps.
            base.set_torch_determinism(True)

            if not theta_by_model or not theta_by_scaffold:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 model/scaffold thetas (unexpected).")
            if not diff_by_item:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 item difficulties (unexpected).")

            train_labeled = [tid for tid in train_items if tid in diff_by_item]
            if len(train_labeled) < 2:
                raise RuntimeError(
                    f"Fold {fold}: only {len(train_labeled)} train items had IRT difficulties; cannot fit regressor."
                )

            # Embeddings-only baseline (for comparability on the scored subset).
            base.seed_everything(int(args.seed) + int(fold), deterministic=True)
            X_train = base.np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(base.np.float32)
            y_train = base.np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=base.np.float32)
            emb_model = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed) + int(fold))
            emb_model.fit(X_train, y_train)

            X_test = base.np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(base.np.float32)
            emb_pred_test = emb_model.predict(X_test).astype(base.np.float64)
            emb_pred_by_item_test = {tid: float(z) for tid, z in zip(test_items, emb_pred_test.tolist())}

            # Joint block-ridge training uses only train items with judge features available.
            joint_emb_train_rows = []
            joint_judge_train_rows = []
            joint_y_train_rows = []
            joint_train_items_used: List[str] = []
            for tid in train_labeled:
                jv = _judge_full_vec_for_item(tid)
                if jv is None:
                    continue
                joint_emb_train_rows.append(X[id_to_row[tid]].astype(base.np.float32))
                joint_judge_train_rows.append(base.np.asarray(jv, dtype=base.np.float32))
                joint_y_train_rows.append(float(diff_by_item[tid]))
                joint_train_items_used.append(tid)
            if len(joint_train_items_used) < 2:
                raise RuntimeError(
                    f"Fold {fold}: only {len(joint_train_items_used)} train items had judge features; cannot fit joint ridge."
                )

            X_emb_joint_train = base.np.stack(joint_emb_train_rows, axis=0).astype(base.np.float32)
            X_judge_joint_train = base.np.stack(joint_judge_train_rows, axis=0).astype(base.np.float32)
            y_joint_train = base.np.asarray(joint_y_train_rows, dtype=base.np.float32)

            reg = str(regressor_name or "ridge_cv").strip()
            if reg == "ridge":
                alpha_emb = (
                    float(args.ridge_alpha_emb) if math.isfinite(float(args.ridge_alpha_emb)) else float(args.ridge_alpha)
                )
                alpha_judge = (
                    float(args.ridge_alpha_judge)
                    if math.isfinite(float(args.ridge_alpha_judge))
                    else float(args.ridge_alpha)
                )
                joint_state = _fit_block_ridge(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alpha_emb=float(alpha_emb),
                    alpha_judge=float(alpha_judge),
                )
            else:
                ae_grid_s = str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas)
                aj_grid_s = str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas)
                ae_grid = _parse_alpha_list(ae_grid_s)
                aj_grid = _parse_alpha_list(aj_grid_s)
                alpha_emb, alpha_judge, _ = _select_block_alphas_inner_cv(
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

            # Predict held-out items where judge features exist.
            final_pred_by_item: Dict[str, float] = {}
            n_missing_judge = 0
            for tid in test_items:
                jv = _judge_full_vec_for_item(tid)
                if jv is None:
                    n_missing_judge += 1
                    continue
                x_emb = X[id_to_row[tid]].reshape(1, -1).astype(base.np.float32)
                x_j = base.np.asarray(jv, dtype=base.np.float32).reshape(1, -1)
                final_pred_by_item[tid] = float(_predict_block_ridge(joint_state, X_emb=x_emb, X_judge=x_j)[0])

            # Fill OOF predictions (NaN if missing judge).
            for tid in test_items:
                i = eligible_index.get(tid, None)
                if i is None:
                    continue
                fold_of_item[int(i)] = int(fold)
                if tid in final_pred_by_item:
                    yhat_oof[int(i)] = float(final_pred_by_item[tid])

            # Held-out AUROC: score only items with final predictions (judge present),
            # and compare embedding-only vs final on the exact same obs set.
            scored_items = set(final_pred_by_item.keys())
            scores_final: List[float] = []
            scores_emb: List[float] = []
            labels: List[int] = []
            test_set = set(test_items)

            for bench, sid, resp in all_responses_tagged:
                key = f"{bench}::{sid}"
                pair = agent_to_ms_pair.get(key, None)
                if pair is None:
                    continue
                model_name, scaffold = pair
                tm = theta_by_model.get(model_name, None)
                ts = theta_by_scaffold.get(scaffold, None)
                if tm is None or ts is None:
                    continue
                th = float(tm) + float(ts)
                for item_id, y_obs in resp.items():
                    if item_id not in test_set:
                        continue
                    if item_id not in scored_items:
                        continue
                    z = final_pred_by_item.get(item_id, None)
                    z_emb = emb_pred_by_item_test.get(item_id, None)
                    if z is None or z_emb is None:
                        continue
                    scores_final.append(_sigmoid(th - float(z)))
                    scores_emb.append(_sigmoid(th - float(z_emb)))
                    labels.append(int(y_obs))

            fold_auc = float(base._compute_binary_auroc(scores_final, labels))
            fold_auc_emb = float(base._compute_binary_auroc(scores_emb, labels))
            cv_test_auc_folds.append(float(fold_auc))
            cv_test_auc_folds_embedding_only.append(float(fold_auc_emb))
            cv_test_n_obs_folds.append(int(len(labels)))
            cv_test_n_items_scored_folds.append(int(len(scored_items)))

            print(f"Fold {fold:02d}: auc={fold_auc} missing_judge={n_missing_judge}")
            if fold_auc == fold_auc and fold_auc > best_fold_auc:
                best_fold_auc = float(fold_auc)
                best_fold = int(fold)
                best_joint_state = joint_state

        if best_joint_state is None or best_fold < 1:
            raise RuntimeError("Failed to select a best CV fold joint model by ROC-AUC (all folds NaN?).")

        auc_arr = base.np.asarray(cv_test_auc_folds, dtype=base.np.float64)
        auc_mean = float(base.np.nanmean(auc_arr)) if auc_arr.size else float("nan")
        auc_std = float(base.np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")
        print("Per-fold ROC-AUC: " + ", ".join([str(x) for x in cv_test_auc_folds]))

        # Save regression weights from the best fold.
        weights_meta = {
            "eval_mode": "id",
            "script": os.path.abspath(__file__),
            "include_judge": True,
            "id_normalization": "Verified/Pro: strip instance_ prefix; strip -v.* suffix. Terminal-Bench: identity.",
            "min_models_per_scaffold": int(args.min_models_per_scaffold),
            "seed": int(args.seed),
            "deterministic": True,
            "irt_seeded": True,
            "irt_deterministic": False,
            "cv_n_splits": int(args.cv_folds),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "irt_model": str(irt_model_label),
            "regressor": str(regressor_name),
            "ridge_alpha": float(args.ridge_alpha),
            "ridge_alphas": str(args.ridge_alphas),
            "ridge_alphas_emb": str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas),
            "ridge_alphas_judge": str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas),
            "inner_splits": int(args.inner_splits),
            "verified_judge_features_dir": str(args.verified_judge_features_dir),
            "pro_judge_features_dir": str(args.pro_judge_features_dir),
            "terminal_bench_judge_features_dir": str(args.terminal_bench_judge_features_dir),
            "judge_feature_names": list(judge_feature_names_full),
        }
        weights_json, weights_npz = save_regression_weights_block_ridge(
            out_dir=str(args.out_dir),
            state=best_joint_state,
            judge_feature_names=judge_feature_names_full,
            metadata=weights_meta,
        )

        metrics = {
            "eval_mode": "id",
            "include_judge": True,
            "n_items_total": int(len(task_ids)),
            "n_items_with_responses": int(len(overlap_ids)),
            "n_items_eligible_cv_irt": int(len(eligible)),
            "exclude_zero_success": bool(exclude_zero_success),
            "n_items_zero_success_in_responses": int(len(zero_success_ids)),
            "embedding_dim": int(Xy.shape[1]),
            "judge_feature_dim": int(judge_dim),
            "seed": int(args.seed),
            "deterministic": True,
            "irt_seeded": True,
            "irt_deterministic": False,
            "cv_n_splits": int(args.cv_folds),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_auc_folds_embedding_only": [float(x) for x in cv_test_auc_folds_embedding_only],
            "cv_test_n_obs_folds": [int(x) for x in cv_test_n_obs_folds],
            "cv_test_n_items_scored_folds": [int(x) for x in cv_test_n_items_scored_folds],
            "cv_selected_alpha_emb_folds": [float(x) for x in fold_alpha_emb],
            "cv_selected_alpha_judge_folds": [float(x) for x in fold_alpha_judge],
            "irt_epochs": int(args.irt_epochs),
            "irt_device": str(args.irt_device),
            "irt_lr": float(args.irt_lr),
            "irt_model": str(irt_model_label),
            "regressor": str(regressor_name),
            "ridge_alpha": float(args.ridge_alpha),
            "ridge_alphas": str(args.ridge_alphas),
            "inner_splits": int(args.inner_splits),
            "dataset_sources": str(dataset_sources_str),
            "verified_agent_results_raw": str(verified_agent_results_raw),
            "pro_agent_results_raw": str(pro_agent_results_raw),
            "terminal_bench_agent_results_raw": str(terminal_agent_results_raw),
            "instruction": str(args.instruction),
            "instruction_signature": instr_sig,
            "batch_size": int(args.batch_size),
            "device_map": str(args.device_map),
            "torch_dtype": str(args.torch_dtype),
            "attn_implementation": str(args.attn_implementation),
            "embeddings_cache": emb_cache,
            "regression_weights_json": str(weights_json),
            "regression_weights_npz": str(weights_npz),
        }
        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

        # Write per-item predictions (OOF CV; NaNs are written as missing_judge).
        pred_path = os.path.join(args.out_dir, "predictions.csv")
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
            w.writeheader()
            for i, tid in enumerate(eligible):
                v = float(yhat_oof[int(i)])
                fold_id = int(fold_of_item[int(i)]) if int(fold_of_item[int(i)]) > 0 else ""
                split = "cv_val" if (v == v) else "missing_judge"
                w.writerow({"item_id": tid, "diff_pred": (v if v == v else ""), "split": split, "fold": fold_id})

        print(f"Wrote predictions: {pred_path}")
        print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
        print(f"Wrote regression weights: {weights_json} (arrays in {weights_npz})")
        return 0

    if eval_mode == "id":
        # -----------------------------
        # K-fold CV over items (existing in-distribution evaluation)
        # -----------------------------
        outer_cv = base.KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        cv_test_auc_folds: List[float] = []
        cv_test_n_obs_folds: List[int] = []
        yhat_oof = base.np.full((int(len(eligible)),), base.np.nan, dtype=base.np.float64)
        fold_of_item = base.np.full((int(len(eligible)),), -1, dtype=base.np.int32)

        best_fold_auc = -float("inf")
        best_fold = -1
        best_model = None

        for fold, (tr, te) in enumerate(outer_cv.split(Xy), start=1):
            train_items = [eligible[int(i)] for i in tr.tolist()]
            test_items = [eligible[int(i)] for i in te.tolist()]

            fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
            base.ensure_dir(fold_root)

            # Save train/test item lists (debugging / provenance).
            base.save_json(os.path.join(fold_root, "train_items.json"), {"items": list(train_items)})
            base.save_json(os.path.join(fold_root, "test_items.json"), {"items": list(test_items)})

            # IRT on train items only (no leakage).
            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)

            obs_train = build_multibench_obs_for_items(obs_full=obs_full, keep_item_ids=train_items)
            theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
                obs_train=obs_train,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, _irt_out_dir_name(irt_model)),
            )

            # Restore determinism for downstream sklearn/regression steps.
            base.set_torch_determinism(True)

            if not theta_by_model or not theta_by_scaffold:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 model/scaffold thetas (unexpected).")
            if not diff_by_item:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 item difficulties (unexpected).")

            train_labeled = [tid for tid in train_items if tid in diff_by_item]
            if len(train_labeled) < 2:
                raise RuntimeError(
                    f"Fold {fold}: only {len(train_labeled)} train items had IRT difficulties; cannot fit regressor."
                )

            base.seed_everything(int(args.seed) + int(fold), deterministic=True)
            X_train = base.np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(base.np.float32)
            y_train = base.np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=base.np.float32)

            m = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed) + int(fold))
            m.fit(X_train, y_train)

            X_test = base.np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(base.np.float32)
            pred = m.predict(X_test).astype(base.np.float64)
            yhat_oof[te] = pred
            fold_of_item[te] = int(fold)

            # Held-out AUROC using held-out items only, with theta_model+theta_scaffold from fold's IRT.
            z_by_item = {tid: float(z) for tid, z in zip(test_items, pred.tolist())}
            scores: List[float] = []
            labels: List[int] = []
            test_set = set(test_items)

            for bench, sid, resp in all_responses_tagged:
                key = f"{bench}::{sid}"
                pair = agent_to_ms_pair.get(key, None)
                if pair is None:
                    continue
                model, scaffold = pair
                tm = theta_by_model.get(model, None)
                ts = theta_by_scaffold.get(scaffold, None)
                if tm is None or ts is None:
                    continue
                th = float(tm) + float(ts)
                for item_id, y_obs in resp.items():
                    if item_id not in test_set:
                        continue
                    z = z_by_item.get(item_id, None)
                    if z is None:
                        continue
                    scores.append(1.0 / (1.0 + math.exp(-(th - float(z)))))
                    labels.append(int(y_obs))

            fold_auc = float(base._compute_binary_auroc(scores, labels))
            cv_test_auc_folds.append(float(fold_auc))
            cv_test_n_obs_folds.append(int(len(labels)))
            if fold_auc == fold_auc and fold_auc > best_fold_auc:
                best_fold_auc = float(fold_auc)
                best_fold = int(fold)
                best_model = m

        if base.np.isnan(yhat_oof).any() or (fold_of_item < 0).any():
            raise RuntimeError("KFold CV produced incomplete out-of-fold predictions (unexpected).")
        if best_model is None or best_fold < 1:
            raise RuntimeError("Failed to select a best CV fold model by ROC-AUC (all folds NaN?).")

        auc_arr = base.np.asarray(cv_test_auc_folds, dtype=base.np.float64)
        auc_mean = float(base.np.nanmean(auc_arr)) if auc_arr.size else float("nan")
        auc_std = float(base.np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")
        print("Per-fold ROC-AUC: " + ", ".join([str(x) for x in cv_test_auc_folds]))

        # Use the best-fold model for saving weights + predicting excluded items.
        model = best_model

        ridge_alpha = None
        if regressor_name in ("ridge", "ridge_cv"):
            try:
                ridge_alpha = float(model.named_steps["ridge"].alpha_)
            except Exception:
                ridge_alpha = None

        metrics = {
            "eval_mode": "id",
            "n_items_total": int(len(task_ids)),
            "n_items_with_responses": int(len(overlap_ids)),
            "n_items_eligible_cv_irt": int(len(eligible)),
            "exclude_zero_success": bool(exclude_zero_success),
            "n_items_zero_success_in_responses": int(len(zero_success_ids)),
            "embedding_dim": int(Xy.shape[1]),
            "seed": int(args.seed),
            "deterministic": True,
            "irt_seeded": True,
            "irt_deterministic": False,
            "cv_n_splits": int(args.cv_folds),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_n_obs_folds": [int(x) for x in cv_test_n_obs_folds],
            "irt_epochs": int(args.irt_epochs),
            "irt_device": str(args.irt_device),
            "irt_lr": float(args.irt_lr),
            "irt_model": str(irt_model_label),
            "regressor": regressor_name,
            "ridge_alpha": ridge_alpha,
            "ridge_alphas_searched": [float(x) for x in base.np.asarray(alphas).tolist()],
            "inner_splits": int(args.inner_splits),
            "backbone": str(args.backbone),
            "pooling": "last_token_of_hidden_state",
            "embedding_layer": int(args.embedding_layer),
            "max_length": int(args.max_length),
            "dataset_sources": str(dataset_sources_str),
            "verified_dataset_name": str(args.verified_dataset_name),
            "verified_dataset_path": str(args.verified_dataset_path),
            "verified_split": str(args.verified_split),
            "pro_dataset_name": str(args.pro_dataset_name),
            "pro_dataset_path": str(args.pro_dataset_path),
            "pro_split": str(args.pro_split),
            "terminal_bench_tasks_jsonl": str(args.terminal_bench_tasks_jsonl),
            "min_models_per_scaffold": int(args.min_models_per_scaffold),
            "verified_agent_results_raw": str(verified_agent_results_raw),
            "pro_agent_results_raw": str(pro_agent_results_raw),
            "terminal_bench_agent_results_raw": str(terminal_agent_results_raw),
            "instruction": str(args.instruction),
            "instruction_signature": instr_sig,
            "batch_size": int(args.batch_size),
            "device_map": str(args.device_map),
            "torch_dtype": str(args.torch_dtype),
            "attn_implementation": str(args.attn_implementation),
            "embeddings_cache": emb_cache,
        }

        weights_meta = {
            "eval_mode": "id",
            "script": os.path.abspath(__file__),
            "backbone": str(args.backbone),
            "trust_remote_code": bool(args.trust_remote_code),
            "pooling": "last_token_of_hidden_state",
            "embedding_layer": int(args.embedding_layer),
            "max_length": int(args.max_length),
            "instruction": str(args.instruction),
            "instruction_signature": str(instr_sig),
            "device_map": str(args.device_map),
            "torch_dtype": str(args.torch_dtype),
            "attn_implementation": str(args.attn_implementation),
            "dataset_sources": str(dataset_sources_str),
            "id_normalization": "Verified/Pro: strip instance_ prefix; strip -v.* suffix. Terminal-Bench: identity.",
            "min_models_per_scaffold": int(args.min_models_per_scaffold),
            "seed": int(args.seed),
            "deterministic": True,
            "irt_seeded": True,
            "irt_deterministic": False,
            "cv_n_splits": int(args.cv_folds),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "ridge_alpha": ridge_alpha,
            "ridge_alphas_searched": [float(x) for x in base.np.asarray(alphas).tolist()],
            "inner_splits": int(args.inner_splits),
            "irt_model": str(irt_model_label),
        }
        weights_json, weights_npz = base.save_regression_weights(
            out_dir=str(args.out_dir),
            model=model,
            regressor_name=str(regressor_name),
            feature_dim=int(Xy.shape[1]),
            metadata=weights_meta,
        )
        metrics.update({"regression_weights_json": weights_json, "regression_weights_npz": weights_npz})

        # Predict on zero-success items (excluded from CV/IRT, if requested).
        zero_embedded: List[str] = []
        yhat_zero: Optional[base.np.ndarray] = None
        if bool(exclude_zero_success) and zero_success_set:
            zero_embedded = [tid for tid in task_ids if tid in zero_success_set]
            if zero_embedded:
                X_zero = base.np.stack([X[id_to_row[tid]] for tid in zero_embedded], axis=0).astype(base.np.float32)
                yhat_zero = model.predict(X_zero).astype(base.np.float64)
            else:
                print("NOTE: zero-success ids provided, but none were present in embedded task_ids; nothing to predict.")

        metrics.update(
            {
                "n_items_zero_success_embedded": int(len(zero_embedded)),
                "n_items_zero_success_predicted": int(0 if yhat_zero is None else int(base.np.asarray(yhat_zero).size)),
            }
        )
        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

        # Write per-item predictions (OOF CV + optional zero_success rows).
        pred_path = os.path.join(args.out_dir, "predictions.csv")
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
            w.writeheader()

            for i, tid in enumerate(eligible):
                w.writerow(
                    {
                        "item_id": tid,
                        "diff_pred": float(yhat_oof[i]),
                        "split": "cv_val",
                        "fold": int(fold_of_item[i]),
                    }
                )

            if yhat_zero is not None and zero_embedded:
                for tid, score in zip(zero_embedded, yhat_zero.tolist()):
                    w.writerow(
                        {
                            "item_id": tid,
                            "diff_pred": float(score),
                            "split": "zero_success",
                            "fold": "",
                        }
                    )

        print(f"Wrote predictions: {pred_path}")
        print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
        return 0

    # -----------------------------
    # Training-only mode (default): single train vs zero-success split
    # -----------------------------
    if bool(args.include_zero_success):
        print(
            "NOTE: --include_zero_success is ignored when --eval_mode is not ID. "
            "Training-only mode always splits into train vs zero-success and trains only on the train split."
        )

    train_items = [tid for tid in overlap_ids if tid not in zero_success_set]
    if not train_items:
        raise RuntimeError("Training-only mode: after excluding zero-success items, no train items remain.")

    # IRT on train items only.
    base.set_torch_determinism(False)
    base.seed_everything(int(args.seed), deterministic=False)
    obs_train = build_multibench_obs_for_items(obs_full=obs_full, keep_item_ids=train_items)
    theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
        obs_train=obs_train,
        irt_model=str(irt_model),
        epochs=int(args.irt_epochs),
        device=str(args.irt_device),
        seed=int(args.seed),
        lr=float(args.irt_lr),
        out_dir=os.path.join(str(args.out_dir), _irt_out_dir_name(irt_model)),
    )
    base.set_torch_determinism(True)

    train_labeled = [tid for tid in train_items if tid in diff_by_item]
    if len(train_labeled) < 2:
        raise RuntimeError(
            f"Training-only mode: only {len(train_labeled)} train items had IRT difficulties; cannot fit regressor."
        )

    # Fit regressor on train split only (no evaluation).
    base.seed_everything(int(args.seed), deterministic=True)
    X_train = base.np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(base.np.float32)
    y_train = base.np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=base.np.float32)

    model = None
    joint_state = None
    judge_feature_names_full: List[str] = []
    judge_dim = 0

    if not use_joint:
        model = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed))
        model.fit(X_train, y_train)
    else:
        verified_item_set = set(obs_full.verified_item_ids)
        pro_item_set = set(obs_full.pro_item_ids)
        terminal_item_set = set(obs_full.terminal_bench_item_ids)

        verified_off = 0
        pro_off = verified_off + int(len(VERIFIED_JUDGE_FEATURE_NAMES))
        terminal_off = pro_off + int(len(PRO_JUDGE_FEATURE_NAMES))
        judge_dim = terminal_off + int(len(TERMINAL_BENCH_JUDGE_FEATURE_NAMES))
        judge_feature_names_full = (
            [f"verified::{k}" for k in VERIFIED_JUDGE_FEATURE_NAMES]
            + [f"pro::{k}" for k in PRO_JUDGE_FEATURE_NAMES]
            + [f"terminal_bench::{k}" for k in TERMINAL_BENCH_JUDGE_FEATURE_NAMES]
        )

        verified_feat_dir = str(args.verified_judge_features_dir)
        pro_feat_dir = str(args.pro_judge_features_dir)
        terminal_feat_dir = str(args.terminal_bench_judge_features_dir)
        verified_idx = _build_judge_index(verified_feat_dir, normalize_item_ids=True)
        pro_idx = _build_judge_index(pro_feat_dir, normalize_item_ids=True)
        terminal_idx = _build_judge_index(terminal_feat_dir, normalize_item_ids=False)

        def _judge_full_vec_for_item(item_id: str):
            tid = str(item_id)
            x = base.np.zeros((int(judge_dim),), dtype=base.np.float32)
            if tid in verified_item_set:
                v = _load_judge_vector(
                    tid,
                    features_dir=verified_feat_dir,
                    feature_names=VERIFIED_JUDGE_FEATURE_NAMES,
                    index=verified_idx,
                    normalize_item_ids=True,
                )
                if v is None:
                    return None
                x[verified_off:pro_off] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                return x
            if tid in pro_item_set:
                v = _load_judge_vector(
                    tid,
                    features_dir=pro_feat_dir,
                    feature_names=PRO_JUDGE_FEATURE_NAMES,
                    index=pro_idx,
                    normalize_item_ids=True,
                )
                if v is None:
                    return None
                x[pro_off:terminal_off] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                return x
            if tid in terminal_item_set:
                v = _load_judge_vector(
                    tid,
                    features_dir=terminal_feat_dir,
                    feature_names=TERMINAL_BENCH_JUDGE_FEATURE_NAMES,
                    index=terminal_idx,
                    normalize_item_ids=False,
                )
                if v is None:
                    return None
                x[terminal_off:] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                return x
            return None

        joint_emb_train_rows = []
        joint_judge_train_rows = []
        joint_y_train_rows = []
        joint_train_items_used: List[str] = []
        for tid in train_labeled:
            jv = _judge_full_vec_for_item(tid)
            if jv is None:
                continue
            joint_emb_train_rows.append(X[id_to_row[tid]].astype(base.np.float32))
            joint_judge_train_rows.append(base.np.asarray(jv, dtype=base.np.float32))
            joint_y_train_rows.append(float(diff_by_item[tid]))
            joint_train_items_used.append(tid)
        if len(joint_train_items_used) < 2:
            raise RuntimeError(
                f"Training-only mode: only {len(joint_train_items_used)} train items had judge features; cannot fit joint ridge."
            )

        X_emb_joint_train = base.np.stack(joint_emb_train_rows, axis=0).astype(base.np.float32)
        X_judge_joint_train = base.np.stack(joint_judge_train_rows, axis=0).astype(base.np.float32)
        y_joint_train = base.np.asarray(joint_y_train_rows, dtype=base.np.float32)

        reg = str(regressor_name or "ridge_cv").strip()
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
        else:
            ae_grid_s = str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas)
            aj_grid_s = str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas)
            ae_grid = _parse_alpha_list(ae_grid_s)
            aj_grid = _parse_alpha_list(aj_grid_s)
            alpha_emb, alpha_judge, _ = _select_block_alphas_inner_cv(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alphas_emb=ae_grid,
                alphas_judge=aj_grid,
                inner_splits=int(args.inner_splits),
                seed=int(args.seed) + 2000,
            )
            joint_state = _fit_block_ridge(
                X_emb=X_emb_joint_train,
                X_judge=X_judge_joint_train,
                y=y_joint_train,
                alpha_emb=float(alpha_emb),
                alpha_judge=float(alpha_judge),
            )

    ridge_alpha = None
    if regressor_name in ("ridge", "ridge_cv"):
        if not use_joint:
            try:
                ridge_alpha = float(model.named_steps["ridge"].alpha_)
            except Exception:
                ridge_alpha = None

    weights_meta = {
        "eval_mode": str(eval_mode),
        "script": os.path.abspath(__file__),
        "backbone": str(args.backbone),
        "trust_remote_code": bool(args.trust_remote_code),
        "pooling": "last_token_of_hidden_state",
        "embedding_layer": int(args.embedding_layer),
        "max_length": int(args.max_length),
        "instruction": str(args.instruction),
        "instruction_signature": str(instr_sig),
        "device_map": str(args.device_map),
        "torch_dtype": str(args.torch_dtype),
        "attn_implementation": str(args.attn_implementation),
        "dataset_sources": str(dataset_sources_str),
        "id_normalization": "Verified/Pro: strip instance_ prefix; strip -v.* suffix. Terminal-Bench: identity.",
        "min_models_per_scaffold": int(args.min_models_per_scaffold),
        "seed": int(args.seed),
        "deterministic": True,
        "irt_seeded": True,
        "irt_deterministic": False,
        "irt_epochs": int(args.irt_epochs),
        "irt_device": str(args.irt_device),
        "irt_lr": float(args.irt_lr),
        "irt_model": str(irt_model_label),
        "regressor": regressor_name,
        "ridge_alpha": ridge_alpha,
        "ridge_alphas_searched": [float(x) for x in base.np.asarray(alphas).tolist()],
        "inner_splits": int(args.inner_splits),
        "n_items_total": int(len(task_ids)),
        "n_items_with_responses": int(len(overlap_ids)),
        "n_items_train": int(len(train_items)),
        "n_items_train_labeled": int(len(train_labeled)),
        "n_items_zero_success_in_responses": int(len(zero_success_ids)),
        "embeddings_cache": emb_cache,
    }
    if not use_joint:
        base.save_regression_weights(
            out_dir=str(args.out_dir),
            model=model,
            regressor_name=str(regressor_name),
            feature_dim=int(Xy.shape[1]),
            metadata=weights_meta,
        )
    else:
        if joint_state is None:
            raise RuntimeError("Internal error: joint_state was None after joint training.")
        weights_meta.update(
            {
                "include_judge": True,
                "ridge_alpha": float(args.ridge_alpha),
                "ridge_alphas": str(args.ridge_alphas),
                "ridge_alphas_emb": str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas),
                "ridge_alphas_judge": str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas),
                "verified_judge_features_dir": str(args.verified_judge_features_dir),
                "pro_judge_features_dir": str(args.pro_judge_features_dir),
                "terminal_bench_judge_features_dir": str(args.terminal_bench_judge_features_dir),
                "judge_feature_names": list(judge_feature_names_full),
                "judge_feature_dim": int(judge_dim),
            }
        )
        save_regression_weights_block_ridge(
            out_dir=str(args.out_dir),
            state=joint_state,
            judge_feature_names=judge_feature_names_full,
            metadata=weights_meta,
        )

    # -----------------------------
    # OOD evaluation: 4th benchmark AUROC using learned thetas + regressor
    # -----------------------------
    if eval_mode == "ood":
        ood_agent_results = str(args.ood_agent_results or "").strip()
        if not ood_agent_results:
            raise ValueError("--ood_agent_results is empty (required for --eval_mode=ood).")
        if not os.path.exists(ood_agent_results):
            raise FileNotFoundError(f"OOD agent results JSONL not found: {ood_agent_results}")

        # Collect OOD item ids from the response matrix (so we embed only what's needed).
        ood_item_ids: List[str] = []
        ood_item_set: Set[str] = set()
        for _, resp in iter_subject_responses_jsonl_generic(
            ood_agent_results, normalize_item_ids=bool(args.ood_normalize_item_ids)
        ):
            for tid in resp.keys():
                if tid not in ood_item_set:
                    ood_item_set.add(tid)
                    ood_item_ids.append(tid)
        if not ood_item_ids:
            raise RuntimeError(f"OOD agent results had 0 items after parsing: {ood_agent_results}")

        # Load OOD tasks (statement+solution) for those ids.
        ood_dataset_name = str(args.ood_dataset_name or "").strip()
        ood_dataset_path = str(args.ood_dataset_path or "").strip()
        if not ood_dataset_name and not ood_dataset_path:
            raise ValueError("OOD mode requires --ood_dataset_name or --ood_dataset_path to load the 4th benchmark tasks.")
        if ood_dataset_name and ood_dataset_path:
            raise ValueError("Provide only one of --ood_dataset_name or --ood_dataset_path (OOD mode).")
        ood_treat_as_pro = _is_swebench_pro_benchmark(dataset_name=ood_dataset_name, dataset_path=ood_dataset_path)
        # Heuristic: GSO exports use subject_id as a model name only; assume OpenHands scaffold for all.
        ood_default_scaffold: Optional[str] = None
        hint = " ".join([ood_dataset_name.lower(), os.path.basename(ood_agent_results).lower()])
        if "gso" in hint:
            ood_default_scaffold = "OpenHands"

        ood_items, ood_missing = load_ood_items_by_ids(
            dataset_name=ood_dataset_name,
            split=str(args.ood_split),
            dataset_path=ood_dataset_path,
            item_ids=ood_item_ids,
            normalize_item_ids=bool(args.ood_normalize_item_ids),
        )
        if ood_missing:
            print(f"WARNING: OOD benchmark: {len(ood_missing)}/{len(ood_item_ids)} item_ids were not found in the dataset. Example: {ood_missing[:10]}")
        if not ood_items:
            raise RuntimeError("OOD benchmark: loaded 0 items to embed; cannot evaluate AUROC.")

        # Embed OOD items using the same backbone/settings.
        ood_ids_sorted, ood_emb_by_id, _, _ = base.embed_items(
            items=list(ood_items),
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
        if not ood_ids_sorted:
            raise RuntimeError("OOD benchmark: embeddings produced 0 ids (unexpected).")

        X_ood = base.np.stack([ood_emb_by_id[iid] for iid in ood_ids_sorted], axis=0).astype(base.np.float32)
        if not use_joint:
            z_pred = model.predict(X_ood).astype(base.np.float64)
        else:
            if joint_state is None:
                raise RuntimeError("Internal error: joint_state was None for OOD evaluation in joint mode.")
            if int(judge_dim) != int(joint_state["n_judge"]):
                raise RuntimeError("Joint ridge: internal judge_dim mismatch vs trained model (cannot predict).")

            ood_feat_dir = str(getattr(args, "ood_judge_features_dir", "") or "").strip()
            ood_idx: Dict[str, str] = _build_judge_index(
                ood_feat_dir, normalize_item_ids=bool(args.ood_normalize_item_ids)
            ) if ood_feat_dir else {}

            def _ood_judge_full_vec_for_item(item_id: str):
                """
                Load OOD judge vector from a single directory, assuming it matches one of the known schemas.
                Returns a full-length vector of size `judge_dim`, or None if not found/parsable.
                """
                tid = str(item_id)
                x = base.np.zeros((int(judge_dim),), dtype=base.np.float32)

                v = _load_judge_vector(
                    tid,
                    features_dir=ood_feat_dir,
                    feature_names=VERIFIED_JUDGE_FEATURE_NAMES,
                    index=ood_idx,
                    normalize_item_ids=bool(args.ood_normalize_item_ids),
                )
                if v is not None:
                    x[verified_off:pro_off] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                    return x

                v = _load_judge_vector(
                    tid,
                    features_dir=ood_feat_dir,
                    feature_names=PRO_JUDGE_FEATURE_NAMES,
                    index=ood_idx,
                    normalize_item_ids=bool(args.ood_normalize_item_ids),
                )
                if v is not None:
                    x[pro_off:terminal_off] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                    return x

                v = _load_judge_vector(
                    tid,
                    features_dir=ood_feat_dir,
                    feature_names=TERMINAL_BENCH_JUDGE_FEATURE_NAMES,
                    index=ood_idx,
                    normalize_item_ids=bool(args.ood_normalize_item_ids),
                )
                if v is not None:
                    x[terminal_off:] = base.np.asarray(v, dtype=base.np.float32).reshape(-1)
                    return x

                return None

            # Behave like in-distribution joint mode: only predict items with judge features.
            ood_ids_used: List[str] = []
            ood_emb_rows: List[base.np.ndarray] = []
            ood_judge_rows: List[base.np.ndarray] = []
            for iid in ood_ids_sorted:
                jv = _ood_judge_full_vec_for_item(iid) if ood_feat_dir else None
                if jv is None:
                    continue
                ood_ids_used.append(str(iid))
                ood_emb_rows.append(base.np.asarray(ood_emb_by_id[iid], dtype=base.np.float32).reshape(-1))
                ood_judge_rows.append(base.np.asarray(jv, dtype=base.np.float32).reshape(-1))

            if not ood_ids_used:
                raise RuntimeError(
                    "OOD benchmark: 0 items had OOD judge features; cannot run joint (emb+judge) prediction. "
                    "Provide --ood_judge_features_dir with per-item JSONs or run without --include_judge."
                )

            X_emb_ood_used = base.np.stack(ood_emb_rows, axis=0).astype(base.np.float32)
            X_judge_ood_used = base.np.stack(ood_judge_rows, axis=0).astype(base.np.float32)
            z_pred_used = _predict_block_ridge(joint_state, X_emb=X_emb_ood_used, X_judge=X_judge_ood_used).astype(base.np.float64)
            z_by_item = {iid: float(z) for iid, z in zip(ood_ids_used, z_pred_used.tolist())}
        if not use_joint:
            z_by_item = {iid: float(z) for iid, z in zip(ood_ids_sorted, z_pred.tolist())}

        # Write per-item OOD predictions for the 4th benchmark.
        pred_path = os.path.join(str(args.out_dir), "predictions.csv")
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
            w.writeheader()
            for iid, z in (z_by_item or {}).items():
                w.writerow({"item_id": str(iid), "diff_pred": float(z), "split": "ood", "fold": ""})
        print(f"Wrote predictions: {pred_path}")

        ood_auc, ood_meta = evaluate_ood_auroc(
            ood_agent_results_jsonl=ood_agent_results,
            ood_normalize_item_ids=bool(args.ood_normalize_item_ids),
            ood_treat_as_pro=bool(ood_treat_as_pro),
            ood_default_scaffold=ood_default_scaffold,
            z_by_item=z_by_item,
            theta_by_model=theta_by_model,
            theta_by_scaffold=theta_by_scaffold,
        )
        print(f"OOD ROC-AUC (4th benchmark): {ood_auc}  (agent_results={ood_agent_results})")
        try:
            print(
                "OOD eval details: "
                f"subjects_used={int(ood_meta.get('subjects_used', 0))} "
                f"(assumed_scaffold={int(ood_meta.get('subjects_used_assumed_scaffold', 0))}, "
                f"model_only={int(ood_meta.get('subjects_used_model_only', 0))}) "
                f"obs_scored={int(ood_meta.get('obs_scored', 0))} "
                f"(assumed_scaffold={int(ood_meta.get('obs_scored_assumed_scaffold', 0))}, "
                f"model_only={int(ood_meta.get('obs_scored_model_only', 0))}) "
                f"labels_pos={int(ood_meta.get('labels_pos', 0))} "
                f"labels_neg={int(ood_meta.get('labels_neg', 0))} "
                f"skipped_no_theta={int(ood_meta.get('obs_skipped_no_theta', 0))} "
                f"skipped_no_item={int(ood_meta.get('obs_skipped_no_item', 0))}"
            )
            if not (ood_auc == ood_auc):
                if int(ood_meta.get("obs_scored", 0)) <= 0:
                    print("NOTE: OOD ROC-AUC is NaN because 0 observations were scored (no usable thetas for subjects).")
                elif int(ood_meta.get("labels_pos", 0)) == 0 or int(ood_meta.get("labels_neg", 0)) == 0:
                    print("NOTE: OOD ROC-AUC is NaN because only one label class was present among scored observations.")
        except Exception:
            pass
        base.save_json(
            os.path.join(str(args.out_dir), "metrics.json"),
            {
                "eval_mode": "ood",
                "auc": float(ood_auc),
                "include_judge": bool(use_joint),
                "agent_results": str(ood_agent_results),
                "dataset_name": (ood_dataset_name or None),
                "dataset_path": (ood_dataset_path or None),
                "split": str(args.ood_split),
                "normalize_item_ids": bool(args.ood_normalize_item_ids),
                "ood_judge_features_dir": str(getattr(args, "ood_judge_features_dir", "") or "").strip() or None,
                **ood_meta,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
