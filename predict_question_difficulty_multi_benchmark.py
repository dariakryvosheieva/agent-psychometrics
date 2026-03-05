
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import predict_question_difficulty as base

def _canon_benchmark_name(name: str) -> str:
    s = str(name or "").strip().lower().replace("-", "_")
    if s == "terminalbench":
        s = "terminal_bench"
    if s not in {"verified", "pro", "terminal_bench", "gso"}:
        raise ValueError(f"Unknown benchmark name: {name!r}. Allowed: verified, pro, terminal-bench, gso.")
    return s

def _parse_benchmark_list(spec: str) -> List[str]:
    raw = str(spec or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    out: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        if not p:
            continue
        k = _canon_benchmark_name(p)
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

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

def evaluate_ood_auroc_agent_irt(
    *,
    ood_benchmark: str,
    ood_agent_results_jsonl: str,
    ood_normalize_item_ids: bool,
    z_by_item: Dict[str, float],
    theta_by_agent: Dict[str, float],
) -> Tuple[float, dict]:
    bench = str(ood_benchmark or "").strip()
    if not bench:
        raise ValueError("ood_benchmark was empty")

    scores: List[float] = []
    labels: List[int] = []
    n_subjects_total = 0
    n_subjects_used = 0
    n_obs_total = 0
    n_obs_scored = 0
    n_obs_skipped_no_theta = 0
    n_obs_skipped_no_item = 0

    for sid, resp in iter_subject_responses_jsonl_generic(ood_agent_results_jsonl, normalize_item_ids=bool(ood_normalize_item_ids)):
        n_subjects_total += 1
        agent_key = str(sid)
        th = theta_by_agent.get(str(agent_key), None)
        if th is None:
            n_obs_skipped_no_theta += int(len(resp))
            n_obs_total += int(len(resp))
            continue
        n_subjects_used += 1
        for item_id, y_obs in resp.items():
            n_obs_total += 1
            z = z_by_item.get(str(item_id), None)
            if z is None:
                n_obs_skipped_no_item += 1
                continue
            scores.append(base.base._sigmoid(float(th) - float(z)))
            labels.append(int(y_obs))
            n_obs_scored += 1

    auc = float(base._compute_binary_auroc(scores, labels))
    meta = {
        "subjects_total": int(n_subjects_total),
        "subjects_used": int(n_subjects_used),
        "items_predicted": int(len(z_by_item)),
        "obs_total": int(n_obs_total),
        "obs_scored": int(n_obs_scored),
        "obs_skipped_no_theta": int(n_obs_skipped_no_theta),
        "obs_skipped_no_item": int(n_obs_skipped_no_item),
        "labels_pos": int(sum(int(x) for x in labels)),
        "labels_neg": int(len(labels) - int(sum(int(x) for x in labels))),
    }
    return auc, meta

def _stable_group_kfold(groups: Sequence[str], *, n_splits: int, seed: int) -> List[Tuple[set, set]]:
    k = int(n_splits)
    if k < 2:
        raise ValueError("--cv_folds must be >= 2 for cross-validation")
    uniq = sorted(set([str(g) for g in groups if str(g).strip()]))
    if len(uniq) < k:
        raise RuntimeError(
            f"Not enough unique groups ({len(uniq)}) for {k}-fold CV. Try a smaller --cv_folds or different --split_by."
        )

    xs: List[Tuple[float, str]] = []
    for g in uniq:
        h = hashlib.md5((g + f"::{int(seed)}").encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / float(16**8)
        xs.append((x, g))
    xs.sort()
    ordered = [g for _, g in xs]

    folds: List[List[str]] = [[] for _ in range(k)]
    for i, g in enumerate(ordered):
        folds[i % k].append(g)

    out: List[Tuple[set, set]] = []
    all_set = set(ordered)
    for j in range(k):
        test = set(folds[j])
        train = set([g for g in all_set if g not in test])
        out.append((train, test))
    return out

def _fold_id_for_group(group_id: str, *, n_splits: int, seed: int) -> int:
    k = int(n_splits)
    if k < 2:
        raise ValueError("--cv_folds must be >= 2 for cross-validation")
    g = str(group_id or "")
    h = hashlib.md5((g + f"::{int(seed)}").encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return int(x % k) + 1

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

THETA_COMBINE_CHOICES: Tuple[str, ...] = (
    "sum",
    "product",
    "max",
    "min",
    "l2",
)

def _combine_model_scaffold_theta(theta_model: float, theta_scaffold: float, *, combine: str) -> float:
    tm = float(theta_model)
    ts = float(theta_scaffold)
    c = str(combine or "").strip().lower()
    if c == "sum":
        return tm + ts
    if c == "product":
        s = 1.0 if (tm + ts) >= 0.0 else -1.0
        return s * abs(tm * ts)
    if c == "max":
        return max(tm, ts)
    if c == "min":
        return min(tm, ts)
    if c == "l2":
        s = 1.0 if (tm + ts) >= 0.0 else -1.0
        return s * float(math.hypot(tm, ts))
    raise ValueError(f"Unknown theta combine form {combine!r}. Expected one of {list(THETA_COMBINE_CHOICES)!r}.")

JUDGE_FEATURE_NAMES: List[str] = [
    "solution_hint",
    "problem_clarity",
    "solution_complexity",
    "domain_knowledge_required",
    "logical_reasoning_required",
    "atypicality",
    "verification_difficulty",
    "standard_pattern_available",
    "codebase_scope"
]

def evaluate_ood_auroc(
    *,
    ood_agent_results_jsonl: str,
    ood_normalize_item_ids: bool,
    ood_treat_as_pro: bool,
    ood_default_scaffold: Optional[str],
    z_by_item: Dict[str, float],
    theta_by_model: Dict[str, float],
    theta_by_scaffold: Dict[str, float],
    theta_combine: str = "sum",
) -> Tuple[float, dict]:
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
    n_obs_skipped_unfamiliar_model = 0
    n_obs_skipped_unfamiliar_scaffold = 0
    n_obs_skipped_no_item = 0
    n_obs_skipped_bad_score = 0

    for sid, resp in iter_subject_responses_jsonl_generic(
        str(ood_agent_results_jsonl), normalize_item_ids=bool(ood_normalize_item_ids)
    ):
        n_subjects_total += 1

        assume_scaffold = str(ood_default_scaffold or "").strip()

        used_model_only = False
        used_assumed_scaffold = False
        th: Optional[float] = None

        model_name: Optional[str] = None
        scaffold_name: Optional[str] = None

        if assume_scaffold:
            try:
                model_name = str(split_mod._canonical_model(str(sid)))
            except Exception:
                model_name = str(sid)
            try:
                scaffold_name = str(split_mod._canonical_scaffold(str(assume_scaffold)))
            except Exception:
                scaffold_name = str(assume_scaffold)
            used_assumed_scaffold = True
        else:

            try:
                m = split_mod._model_for_subject(sid, treat_as_pro=bool(ood_treat_as_pro))
            except Exception:
                m = None
            try:
                sc = split_mod._scaffold_for_subject(sid, treat_as_pro=bool(ood_treat_as_pro))
            except Exception:
                sc = None
            model_name = str(m) if m is not None else None
            scaffold_name = str(sc) if sc is not None else None

        model_name = str(model_name or "").strip() or None
        scaffold_name = str(scaffold_name or "").strip() or None

        if model_name is None:
            n_obs_total += int(len(resp))
            n_obs_skipped_unfamiliar_model += int(len(resp))
            n_obs_skipped_no_theta += int(len(resp))
            continue
        if scaffold_name is None:
            n_obs_total += int(len(resp))
            n_obs_skipped_unfamiliar_scaffold += int(len(resp))
            n_obs_skipped_no_theta += int(len(resp))
            continue

        tm = theta_by_model.get(str(model_name), None)
        if tm is None:
            n_obs_total += int(len(resp))
            n_obs_skipped_unfamiliar_model += int(len(resp))
            n_obs_skipped_no_theta += int(len(resp))
            continue
        ts = theta_by_scaffold.get(str(scaffold_name), None)
        if ts is None:
            n_obs_total += int(len(resp))
            n_obs_skipped_unfamiliar_scaffold += int(len(resp))
            n_obs_skipped_no_theta += int(len(resp))
            continue

        th = _combine_model_scaffold_theta(float(tm), float(ts), combine=str(theta_combine))

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
            s = base._sigmoid(th - float(z))
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
        "obs_skipped_unfamiliar_model": int(n_obs_skipped_unfamiliar_model),
        "obs_skipped_unfamiliar_scaffold": int(n_obs_skipped_unfamiliar_scaffold),
        "obs_skipped_no_item": int(n_obs_skipped_no_item),
        "obs_skipped_bad_score": int(n_obs_skipped_bad_score),
        "labels_pos": int(n_pos),
        "labels_neg": int(n_neg),
    }
    return auc, meta

def compute_empirical_success_prob_by_model(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    agent_to_ms_pair: Dict[str, Tuple[str, str]],
    train_item_ids: Set[str],
    keep_agent_keys: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], dict]:
    split_mod = _import_swebench_irt_module("split_agents_model_scaffold")

    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    n_subjects = 0
    n_obs_total = 0
    n_obs_used = 0
    n_obs_skipped_no_model = 0
    n_obs_skipped_not_train_item = 0

    train_set = set([str(x) for x in train_item_ids])
    keep_agents = set([str(x) for x in keep_agent_keys]) if keep_agent_keys is not None else None

    for bench, sid, resp in all_responses_tagged:
        n_subjects += 1
        bench_s = str(bench)
        sid_s = str(sid)
        key = f"{bench_s}::{sid_s}"
        if keep_agents is not None and key not in keep_agents:
            continue

        model_name: Optional[str] = None
        pair = agent_to_ms_pair.get(key, None)
        if pair is not None:
            model_name = str(pair[0])
        if not model_name:

            try:
                m = split_mod._model_for_subject(sid_s, treat_as_pro=bool(bench_s == "pro"))
                if m is not None:
                    model_name = str(m)
            except Exception:
                model_name = None
        if not model_name:
            try:
                model_name = str(split_mod._canonical_model(sid_s))
            except Exception:
                model_name = sid_s
        model_name = str(model_name or "").strip() or None

        for item_id, y_obs in resp.items():
            n_obs_total += 1
            tid = str(item_id)
            if tid not in train_set:
                n_obs_skipped_not_train_item += 1
                continue
            if model_name is None:
                n_obs_skipped_no_model += 1
                continue
            counts[model_name][0] += 1
            counts[model_name][1] += int(y_obs)
            n_obs_used += 1

    probs: Dict[str, float] = {}
    for m, (n, k) in counts.items():
        if int(n) <= 0:
            continue
        probs[str(m)] = float(k) / float(n)

    meta = {
        "subjects_total": int(n_subjects),
        "models_total": int(len(counts)),
        "models_with_prob": int(len(probs)),
        "obs_total": int(n_obs_total),
        "obs_used": int(n_obs_used),
        "obs_skipped_not_train_item": int(n_obs_skipped_not_train_item),
        "obs_skipped_no_model": int(n_obs_skipped_no_model),
    }
    return probs, meta

def compute_empirical_solve_rate_by_item(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    train_item_ids: Set[str],
    keep_agent_keys: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], dict]:

    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    n_subjects = 0
    n_obs_total = 0
    n_obs_used = 0
    n_obs_skipped_not_train_item = 0

    train_set = set([str(x) for x in train_item_ids])
    keep_agents = set([str(x) for x in keep_agent_keys]) if keep_agent_keys is not None else None

    for bench, sid, resp in all_responses_tagged:
        n_subjects += 1
        bench_s = str(bench)
        sid_s = str(sid)
        key = f"{bench_s}::{sid_s}"
        if keep_agents is not None and key not in keep_agents:
            continue

        for item_id, y_obs in resp.items():
            n_obs_total += 1
            tid = str(item_id)
            if tid not in train_set:
                n_obs_skipped_not_train_item += 1
                continue
            counts[tid][0] += 1
            counts[tid][1] += int(y_obs)
            n_obs_used += 1

    probs: Dict[str, float] = {}
    for iid, (n, k) in counts.items():
        if int(n) <= 0:
            continue
        probs[str(iid)] = float(k) / float(n)

    meta = {
        "subjects_total": int(n_subjects),
        "items_total": int(len(counts)),
        "items_with_prob": int(len(probs)),
        "obs_total": int(n_obs_total),
        "obs_used": int(n_obs_used),
        "obs_skipped_not_train_item": int(n_obs_skipped_not_train_item),
    }
    return probs, meta

def evaluate_empirical_model_success_auroc(
    *,
    agent_results_jsonl: str,
    normalize_item_ids: bool,
    treat_as_pro: bool,
    ood_default_scaffold: Optional[str],
    p_success_by_model: Dict[str, float],
) -> Tuple[float, dict]:
    split_mod = _import_swebench_irt_module("split_agents_model_scaffold")

    scores: List[float] = []
    labels: List[int] = []
    n_subjects_total = 0
    n_subjects_used = 0
    n_obs_total = 0
    n_obs_scored = 0
    n_obs_skipped_no_theta = 0
    n_obs_skipped_unfamiliar_model = 0
    n_obs_skipped_bad_score = 0

    for sid, resp in iter_subject_responses_jsonl_generic(str(agent_results_jsonl), normalize_item_ids=bool(normalize_item_ids)):
        n_subjects_total += 1

        assume_scaffold = str(ood_default_scaffold or "").strip()

        model_name: Optional[str] = None
        if assume_scaffold:

            try:
                model_name = str(split_mod._canonical_model(str(sid)))
            except Exception:
                model_name = str(sid)
        else:

            try:
                m = split_mod._model_for_subject(sid, treat_as_pro=bool(treat_as_pro))
            except Exception:
                m = None
            model_name = str(m) if m is not None else None
            if not model_name:
                try:
                    model_name = str(split_mod._canonical_model(str(sid)))
                except Exception:
                    model_name = str(sid)

        model_name = str(model_name or "").strip() or None

        if model_name is None:
            n_obs_total += int(len(resp))
            n_obs_skipped_unfamiliar_model += int(len(resp))
            n_obs_skipped_no_theta += int(len(resp))
            continue

        p = p_success_by_model.get(str(model_name), None)
        if p is None:
            n_obs_total += int(len(resp))
            n_obs_skipped_unfamiliar_model += int(len(resp))
            n_obs_skipped_no_theta += int(len(resp))
            continue

        n_subjects_used += 1
        for _, y_obs in resp.items():
            n_obs_total += 1
            s = float(p)
            if not math.isfinite(s):
                n_obs_skipped_bad_score += 1
                continue
            scores.append(float(s))
            labels.append(int(y_obs))
            n_obs_scored += 1

    auc = float(base._compute_binary_auroc(scores, labels))
    n_pos = int(sum(int(x) for x in labels))
    n_neg = int(len(labels) - n_pos)
    meta = {
        "subjects_total": int(n_subjects_total),
        "subjects_used": int(n_subjects_used),
        "models_with_prob": int(len(p_success_by_model)),
        "obs_total": int(n_obs_total),
        "obs_scored": int(n_obs_scored),
        "obs_skipped_no_theta": int(n_obs_skipped_no_theta),
        "obs_skipped_unfamiliar_model": int(n_obs_skipped_unfamiliar_model),
        "obs_skipped_bad_score": int(n_obs_skipped_bad_score),
        "labels_pos": int(n_pos),
        "labels_neg": int(n_neg),
    }
    return auc, meta


def load_ood_items_by_ids(
    *,
    dataset_name: str,
    split: str,
    item_ids: Sequence[str],
    normalize_item_ids: bool,
    wrap_with_gso_prompt: bool = False,
) -> Tuple[List[base.ItemRecord], List[str]]:
    want = [base.normalize_swebench_item_id(x) for x in list(item_ids)] if bool(normalize_item_ids) else [str(x) for x in list(item_ids)]
    want = [x for x in want if str(x).strip()]
    want_set = set(want)
    if not want:
        return [], []

    dataset_name = str(dataset_name or "").strip()
    if not dataset_name:
        raise ValueError("No dataset provided (set dataset_name).")
    ds = base.load_dataset(str(dataset_name), split=str(split))

    n_total = int(len(ds))
    if n_total == 0:
        raise RuntimeError(f"Loaded empty dataset: {dataset_name} split={split}")

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

        prob_script = str(row.get("prob_script", "") or "")
        qs = base._wrap_gso_problem_statement(prob_script) if bool(wrap_with_gso_prompt) else prob_script
        sol = str(row.get("gt_diff", "") or "")
        found[item_id] = base.ItemRecord(item_id=item_id, question_statement=qs, solution=sol)
        if len(found) >= len(want_set):
            break

    items = [found[tid] for tid in want if tid in found]
    missing = [tid for tid in want if tid not in found]
    return items, missing

def load_swebench_items_by_ids(
    *,
    dataset_name: str,
    split: str,
    item_ids: Sequence[str],
    normalize_item_ids: bool = True,
) -> Tuple[List[base.ItemRecord], List[str]]:
    want_raw = [str(x) for x in item_ids if str(x).strip()]
    if not want_raw:
        return [], []
    want = [base.normalize_swebench_item_id(x) if bool(normalize_item_ids) else str(x).strip() for x in want_raw]
    want = [x for x in want if x]
    want_set = set(want)
    found: Dict[str, base.ItemRecord] = {}
    for it in base.iter_swebench_items(dataset_name=str(dataset_name), split=str(split), dataset_path=""):
        iid = str(it.item_id)
        if iid in want_set and iid not in found:
            found[iid] = it
            if len(found) >= len(want_set):
                break
    items = [found[tid] for tid in want if tid in found]
    missing = [tid for tid in want if tid not in found]
    return items, missing

def load_terminal_bench_items_by_ids(*, tasks_jsonl: str, item_ids: Sequence[str]) -> Tuple[List[base.ItemRecord], List[str]]:
    want = [str(x) for x in item_ids if str(x).strip()]
    if not want:
        return [], []
    want_set = set(want)
    found: Dict[str, base.ItemRecord] = {}
    for it in iter_terminal_bench_items_from_jsonl(path=str(tasks_jsonl)):
        iid = str(it.item_id)
        if iid in want_set and iid not in found:
            found[iid] = it
            if len(found) >= len(want_set):
                break
    items = [found[tid] for tid in want if tid in found]
    missing = [tid for tid in want if tid not in found]
    return items, missing

def load_all_responses_generic(*, path: str, normalize_item_ids: bool) -> List[Tuple[str, Dict[str, int]]]:
    out: List[Tuple[str, Dict[str, int]]] = []
    for sid, resp in iter_subject_responses_jsonl_generic(str(path), normalize_item_ids=bool(normalize_item_ids)):
        if resp:
            out.append((sid, resp))
    return out

def _import_swebench_irt_module(module_name: str):
    here = Path(__file__).resolve().parent
    swe_irt_dir = str(here / "swebench_irt")
    if swe_irt_dir not in sys.path:
        sys.path.insert(0, swe_irt_dir)
    return __import__(str(module_name))

def train_standard_irt_1pl_agents(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    keep_item_ids: Set[str],
    epochs: int,
    device: str,
    seed: int,
    out_dir: str,
    keep_agent_keys: Optional[Set[str]] = None,
    keep_obs_fn: Optional[Callable[[str, str, str], bool]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    import torch

    outp = str(out_dir or "").strip()
    if not outp:
        raise ValueError("out_dir was empty")

    if os.path.exists(outp):
        shutil.rmtree(outp, ignore_errors=True)
    base.ensure_dir(outp)

    items = sorted([str(x) for x in keep_item_ids if str(x).strip()])
    if not items:
        raise ValueError("keep_item_ids was empty")
    item_set = set(items)
    keep_agents = set([str(x) for x in keep_agent_keys]) if keep_agent_keys is not None else None

    subj_to_present: Dict[str, Dict[str, int]] = defaultdict(dict)
    for bench, sid, resp in all_responses_tagged:
        bench_s = str(bench)
        sid_s = str(sid)
        agent_key = sid_s
        if keep_agents is not None and str(agent_key) not in keep_agents:
            continue
        for item_id, y_obs in resp.items():
            tid = str(item_id)
            if tid not in item_set:
                continue
            if keep_obs_fn is not None and not bool(keep_obs_fn(bench_s, sid_s, tid)):
                continue
            subj_to_present[agent_key][tid] = int(y_obs)

    train_jsonl = os.path.join(outp, "train_responses.jsonl")
    n_subjects_written = 0
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for subj in sorted(subj_to_present.keys()):
            present = subj_to_present.get(subj, {})
            if not present:
                continue
            if keep_obs_fn is None:

                out_resp = {tid: int(present.get(tid, 0)) for tid in items}
            else:

                out_resp = {tid: int(v) for tid, v in present.items()}
            f.write(json.dumps({"subject_id": str(subj), "responses": out_resp}) + "\n")
            n_subjects_written += 1
    if n_subjects_written <= 0:
        raise RuntimeError("After filtering, there were 0 observations to train standard IRT on.")

    dev = str(device or "cpu").strip() or "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: --irt_device=cuda requested but CUDA is unavailable; falling back to cpu for IRT.")
        dev = "cpu"

    theta_by_subject, diff_by_item = base.train_irt_1pl(
        responses_jsonl=str(train_jsonl),
        epochs=int(epochs),
        device=str(dev),
        seed=int(seed),
        out_dir=str(outp),
    )
    return theta_by_subject, diff_by_item

def normalize_responses_jsonl(
    *,
    in_path: str,
    out_path: str,
    benchmark: str,
    normalize_item_ids: Optional[bool] = None,
) -> None:
    base.ensure_dir(os.path.dirname(out_path) or ".")
    b = str(benchmark or "").strip().lower()
    if b not in {"verified", "pro", "terminal_bench", "gso"}:
        raise ValueError(f"Unknown benchmark for normalization: {benchmark}")
    norm = normalize_item_ids
    if norm is None:
        norm = (b in {"verified", "pro", "gso"})

    with open(out_path, "w", encoding="utf-8") as f:
        for obj in _iter_jsonl(in_path):
            sid = str(obj.get("subject_id", "") or "").strip()
            resp = obj.get("responses", {}) or {}
            if not sid or not isinstance(resp, dict):
                continue
            out_resp: Dict[str, int] = {}
            for raw_id, v in resp.items():
                if bool(norm):
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
                out_obj: Dict[str, object] = {"subject_id": sid, "responses": out_resp}

                if b == "terminal_bench":
                    for k in ["model", "agent", "date"]:
                        if k in obj:
                            out_obj[k] = obj.get(k)
                f.write(json.dumps(out_obj) + "\n")

def iter_terminal_bench_items_from_jsonl(*, path: str) -> Iterator[base.ItemRecord]:
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
    return _import_swebench_irt_module("train_model_scaffold_shared")

def build_multibench_obs_for_items(
    *,
    obs_full,
    keep_item_ids: Sequence[str],
):
    import torch

    keep: List[str] = [str(x) for x in keep_item_ids if str(x).strip()]
    if not keep:
        raise ValueError("keep_item_ids was empty")

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

    item_map = torch.full((len(full_item_ids),), -1, dtype=torch.long)
    keep_full_idxs_sorted = torch.tensor(sorted(keep_idx_set), dtype=torch.long)
    item_map[keep_full_idxs_sorted] = torch.arange(int(keep_full_idxs_sorted.numel()), dtype=torch.long)
    i_new = item_map[i_old]

    used_m = sorted(set(int(x) for x in m_old.detach().cpu().tolist()))
    used_s = sorted(set(int(x) for x in s_old.detach().cpu().tolist()))

    model_map = torch.full((len(obs_full.model_ids),), -1, dtype=torch.long)
    scaffold_map = torch.full((len(obs_full.scaffold_ids),), -1, dtype=torch.long)
    model_map[torch.tensor(used_m, dtype=torch.long)] = torch.arange(len(used_m), dtype=torch.long)
    scaffold_map[torch.tensor(used_s, dtype=torch.long)] = torch.arange(len(used_s), dtype=torch.long)
    m_new = model_map[m_old]
    s_new = scaffold_map[s_old]

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
        gso_item_ids=set([iid for iid in item_ids if iid in set(getattr(obs_full, "gso_item_ids", set()))]),
        agent_split_df=obs_full.agent_split_df,
    )

def build_multibench_obs_from_tagged_responses(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    agent_to_ms_pair: Dict[str, Tuple[str, str]],
    obs_full_agent_split_df,
    keep_item_ids: Set[str],
    keep_agent_keys: Optional[Set[str]] = None,
    keep_obs_fn: Optional[Callable[[str, str, str], bool]] = None,
):
    import torch

    keep_items = set([str(x) for x in keep_item_ids if str(x).strip()])
    if not keep_items:
        raise ValueError("keep_item_ids was empty")

    keep_agents = set([str(x) for x in keep_agent_keys]) if keep_agent_keys is not None else None

    rows: List[Tuple[str, str, str, int]] = []
    used_agents: List[Tuple[str, str, str, str]] = []
    seen_agents: Set[Tuple[str, str]] = set()

    verified_item_ids: Set[str] = set()
    pro_item_ids: Set[str] = set()
    terminal_item_ids: Set[str] = set()
    gso_item_ids: Set[str] = set()

    for bench, sid, resp in all_responses_tagged:
        bench_s = str(bench)
        sid_s = str(sid)
        agent_key = f"{bench_s}::{sid_s}"
        if keep_agents is not None and agent_key not in keep_agents:
            continue
        pair = agent_to_ms_pair.get(agent_key, None)
        if pair is None:
            continue
        model_name, scaffold = pair

        ak = (bench_s, sid_s)
        if ak not in seen_agents:
            seen_agents.add(ak)
            used_agents.append((bench_s, sid_s, str(model_name), str(scaffold)))

        for item_id, y_obs in resp.items():
            tid = str(item_id)
            if tid not in keep_items:
                continue
            if keep_obs_fn is not None and not bool(keep_obs_fn(bench_s, sid_s, tid)):
                continue
            rows.append((str(model_name), str(scaffold), tid, int(y_obs)))
            if bench_s == "verified":
                verified_item_ids.add(tid)
            elif bench_s == "pro":
                pro_item_ids.add(tid)
            elif bench_s == "terminal_bench":
                terminal_item_ids.add(tid)
            elif bench_s == "gso":
                gso_item_ids.add(tid)

    if not rows:
        raise RuntimeError("After filtering, there were 0 observations to train IRT on.")

    model_ids = sorted(set([m for m, _, _, _ in rows]))
    scaffold_ids = sorted(set([s for _, s, _, _ in rows]))
    item_ids = sorted(set([t for _, _, t, _ in rows]))
    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    scaffold_to_idx = {s: i for i, s in enumerate(scaffold_ids)}
    item_to_idx = {t: i for i, t in enumerate(item_ids)}

    m_list: List[int] = []
    s_list: List[int] = []
    i_list: List[int] = []
    y_list: List[int] = []
    for m, s, t, yv in rows:
        m_list.append(int(model_to_idx[m]))
        s_list.append(int(scaffold_to_idx[s]))
        i_list.append(int(item_to_idx[t]))
        y_list.append(int(yv))

    if not m_list:
        raise RuntimeError("After indexing, there were 0 observations to train IRT on.")

    agent_split_df = obs_full_agent_split_df
    try:
        import pandas as pd

        agent_split_df = pd.DataFrame(
            [{"benchmark": b, "agent": a, "model": m, "scaffold": sc} for (b, a, m, sc) in used_agents]
        )
    except Exception:
        agent_split_df = obs_full_agent_split_df

    ms = _import_shared_irt_module()
    return ms.MultiBenchObs(
        model_idx=torch.tensor(m_list, dtype=torch.long),
        scaffold_idx=torch.tensor(s_list, dtype=torch.long),
        item_idx=torch.tensor(i_list, dtype=torch.long),
        y=torch.tensor(y_list, dtype=torch.float),
        model_ids=model_ids,
        scaffold_ids=scaffold_ids,
        item_ids=item_ids,
        verified_item_ids=verified_item_ids,
        pro_item_ids=pro_item_ids,
        terminal_bench_item_ids=terminal_item_ids,
        gso_item_ids=gso_item_ids,
        agent_split_df=agent_split_df,
    )

def build_agent_only_obs_from_tagged_responses(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    obs_full_agent_split_df,
    keep_item_ids: Set[str],
    keep_agent_keys: Optional[Set[str]] = None,
    keep_obs_fn: Optional[Callable[[str, str, str], bool]] = None,
):
    import torch

    keep_items = set([str(x) for x in keep_item_ids if str(x).strip()])
    if not keep_items:
        raise ValueError("keep_item_ids was empty")

    keep_agents = set([str(x) for x in keep_agent_keys]) if keep_agent_keys is not None else None

    rows: List[Tuple[str, str, int]] = []
    used_agents: List[Tuple[str, str, str]] = []
    seen_agents: Set[Tuple[str, str]] = set()

    verified_item_ids: Set[str] = set()
    pro_item_ids: Set[str] = set()
    terminal_item_ids: Set[str] = set()
    gso_item_ids: Set[str] = set()

    for bench, sid, resp in all_responses_tagged:
        bench_s = str(bench)
        sid_s = str(sid)
        agent_key = f"{bench_s}::{sid_s}"
        if keep_agents is not None and agent_key not in keep_agents:
            continue

        ak = (bench_s, sid_s)
        if ak not in seen_agents:
            seen_agents.add(ak)
            used_agents.append((bench_s, sid_s, agent_key))

        for item_id, y_obs in resp.items():
            tid = str(item_id)
            if tid not in keep_items:
                continue
            if keep_obs_fn is not None and not bool(keep_obs_fn(bench_s, sid_s, tid)):
                continue
            rows.append((agent_key, tid, int(y_obs)))
            if bench_s == "verified":
                verified_item_ids.add(tid)
            elif bench_s == "pro":
                pro_item_ids.add(tid)
            elif bench_s == "terminal_bench":
                terminal_item_ids.add(tid)
            elif bench_s == "gso":
                gso_item_ids.add(tid)

    if not rows:
        raise RuntimeError("After filtering, there were 0 observations to train agent-only IRT on.")

    agent_ids = sorted(set([a for a, _, _ in rows]))
    item_ids = sorted(set([t for _, t, _ in rows]))
    agent_to_idx = {a: i for i, a in enumerate(agent_ids)}
    item_to_idx = {t: i for i, t in enumerate(item_ids)}

    m_list: List[int] = []
    s_list: List[int] = []
    i_list: List[int] = []
    y_list: List[int] = []
    for a, t, yv in rows:
        m_list.append(int(agent_to_idx[a]))
        s_list.append(0)
        i_list.append(int(item_to_idx[t]))
        y_list.append(int(yv))

    agent_split_df = obs_full_agent_split_df
    try:
        import pandas as pd

        agent_split_df = pd.DataFrame(
            [{"benchmark": b, "agent": a, "model": k, "scaffold": "__BASE__"} for (b, a, k) in used_agents]
        )
    except Exception:
        agent_split_df = obs_full_agent_split_df

    ms = _import_shared_irt_module()
    return ms.MultiBenchObs(
        model_idx=torch.tensor(m_list, dtype=torch.long),
        scaffold_idx=torch.tensor(s_list, dtype=torch.long),
        item_idx=torch.tensor(i_list, dtype=torch.long),
        y=torch.tensor(y_list, dtype=torch.float),
        model_ids=agent_ids,
        scaffold_ids=["__BASE__"],
        item_ids=item_ids,
        verified_item_ids=verified_item_ids,
        pro_item_ids=pro_item_ids,
        terminal_bench_item_ids=terminal_item_ids,
        gso_item_ids=gso_item_ids,
        agent_split_df=agent_split_df,
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
    import torch
    import pyro

    ms = _import_shared_irt_module()

    dev = str(device or "cpu").strip() or "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: --irt_device=cuda requested but CUDA is unavailable; falling back to cpu for IRT.")
        dev = "cpu"
    torch_device = torch.device(dev)

    try:
        os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    except Exception:
        pass
    ms.set_seed(int(seed))
    pyro.clear_param_store()

    obs = obs_train

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
        gso_item_ids=set(getattr(obs, "gso_item_ids", set())),
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

    outp = Path(str(out_dir))
    outp.mkdir(parents=True, exist_ok=True)
    ms.save_outputs(out_dir=outp, obs=obs_dev, model_type=model_type)
    try:
        obs_dev.agent_split_df.to_csv(outp / "agent_splits.csv", index=False)
    except Exception:
        pass

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

    try:
        with open(outp / "model_abilities.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model_id", "theta"])
            w.writeheader()
            for mid in sorted(theta_by_model.keys()):
                w.writerow({"model_id": str(mid), "theta": float(theta_by_model[mid])})
    except Exception:
        pass
    try:
        with open(outp / "scaffold_abilities.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["scaffold_id", "theta"])
            w.writeheader()
            for sid in sorted(theta_by_scaffold.keys()):
                w.writerow({"scaffold_id": str(sid), "theta": float(theta_by_scaffold[sid])})
    except Exception:
        pass
    try:
        with open(outp / "items.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "b"])
            w.writeheader()
            for iid in sorted(diff_by_item.keys()):
                w.writerow({"item_id": str(iid), "b": float(diff_by_item[iid])})
    except Exception:
        pass
    return theta_by_model, theta_by_scaffold, diff_by_item

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/multi_benchmark_ood")
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include_zero_success",
        action="store_true",
        help="Include items with 0 successes in CV/IRT and training (not recommended; can destabilize IRT).",
    )

    p.add_argument(
        "--train_benchmarks",
        type=str,
        default="verified,pro,terminal_bench",
        help=(
            "Comma-separated subset of {Verified, Pro, Terminal-Bench, GSO} to use as the TRAIN set. "
            "Must include at least 2 benchmarks. Example: 'verified,pro' or 'pro,terminal-bench,gso'."
        ),
    )
    p.add_argument(
        "--ood_benchmark",
        type=str,
        default="gso",
        help=(
            "Which benchmark to evaluate as OOD (out-of-distribution). Must be one of "
            "{Verified, Pro, Terminal-Bench, GSO} and must NOT appear in --train_benchmarks. "
            "Set to the empty string ('') to disable evaluation entirely (train on --train_benchmarks only, save weights, no eval)."
        ),
    )

    p.add_argument("--verified_dataset_name", type=str, default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--verified_split", type=str, default="test")

    p.add_argument("--pro_dataset_name", type=str, default="ScaleAI/SWE-bench_Pro")
    p.add_argument("--pro_split", type=str, default="test")

    p.add_argument(
        "--terminal_bench_tasks_jsonl",
        type=str,
        default="data/terminalbench/tasks.jsonl",
        help="Terminal-Bench tasks JSONL with fields: task_id, problem_statement, patch.",
    )

    p.add_argument("--gso_dataset_name", type=str, default="gso-bench/gso", help="HF dataset repo for GSO tasks.")
    p.add_argument("--gso_split", type=str, default="test", help="Split name for --gso_dataset_name.")

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
    p.add_argument(
        "--theta_combine",
        type=str,
        default="sum",
        choices=list(THETA_COMBINE_CHOICES),
        help=(
            "Functional form used to combine base LLM and scaffold abilities when scoring: "
            "p(success)=sigmoid(combine(theta_model, theta_scaffold)-b_item). "
            "Default 'sum' matches the historical Rasch-style additive logit."
        ),
    )

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

    p.add_argument(
        "--method",
        type=str,
        default="embedding",
        choices=["embedding", "judge", "combined"],
        help=(
            "Which features to use for difficulty prediction. "
            "'embedding' (default) trains ridge/linear on the embedding vector only. "
            "'combined' concatenates embedding + LLM-judge features and trains a joint (block) ridge. "
            "'judge' trains ridge on judge features only (no embeddings)."
        ),
    )
    p.add_argument(
        "--regressor",
        type=str,
        default="ridge_cv",
        choices=["linear", "ridge", "ridge_cv"],
        help="Regression model (same options as single-benchmark script).",
    )
    p.add_argument("--ridge_alpha", type=float, default=10000.0)
    p.add_argument("--ridge_alphas", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument(
        "--inner_splits",
        type=int,
        default=5,
        help="Inner CV splits for RidgeCV (used when --regressor=ridge_cv). Will be capped by train size; must be >=2.",
    )

    p.add_argument(
        "--verified_judge_features_dir",
        type=str,
        default="llm_judge/features/verified.csv",
        help="Verified judge features (CSV like llm_judge/features/verified.csv, or directory of per-item JSONs).",
    )
    p.add_argument(
        "--pro_judge_features_dir",
        type=str,
        default="llm_judge/features/pro.csv",
        help="Pro judge features (CSV like llm_judge/features/pro.csv, or directory of per-item JSONs).",
    )
    p.add_argument(
        "--terminal_bench_judge_features_dir",
        type=str,
        default="llm_judge/features/terminal_bench.csv",
        help="Terminal-Bench judge features (CSV like llm_judge/features/terminal_bench.csv, or directory of per-item JSONs).",
    )
    p.add_argument(
        "--gso_judge_features_dir",
        type=str,
        default="llm_judge/features/gso.csv",
        help="GSO judge features (CSV like llm_judge/features/gso.csv, or directory of per-item JSONs).",
    )

    p.add_argument(
        "--ridge_alpha_emb",
        type=float,
        default=float("nan"),
        help=(
            "Embedding block ridge alpha (only used when --method=combined and --regressor=ridge). "
            "Defaults to --ridge_alpha when unset."
        ),
    )
    p.add_argument(
        "--ridge_alpha_judge",
        type=float,
        default=float("nan"),
        help=(
            "Judge block ridge alpha (only used when --method=combined and --regressor=ridge). "
            "Defaults to --ridge_alpha when unset."
        ),
    )
    p.add_argument(
        "--ridge_alphas_emb",
        type=str,
        default="",
        help=(
            "Embedding alpha grid for inner CV (only used when --method=combined and --regressor=ridge_cv). "
            "Defaults to --ridge_alphas when unset."
        ),
    )
    p.add_argument(
        "--ridge_alphas_judge",
        type=str,
        default="",
        help=(
            "Judge alpha grid for inner CV (only used when --method=combined and --regressor=ridge_cv). "
            "Defaults to --ridge_alphas when unset."
        ),
    )

    p.add_argument(
        "--split_by",
        type=str,
        default="benchmark",
        choices=["benchmark", "task", "agent", "observation", "none"],
        help=(
            "How to split for evaluation. "
            "'benchmark' (default): train on --train_benchmarks and "
            "evaluate AUROC on a held-out benchmark selected by --ood_benchmark. "
            "'task': K-fold CV holding out items/tasks. "
            "'agent' holds out agents/subjects (in-distribution). "
            "'observation' holds out individual observations (agent,item) (in-distribution). "
            "'none' disables evaluation and uses all eligible items as training data (no baseline/oracle)."
        ),
    )

    p.add_argument(
        "--verified_agent_results",
        type=str,
        default="data/swebench_verified/responses.jsonl",
        help="Verified response-matrix JSONL: {'subject_id': ..., 'responses': {'item_id': 0/1, ...}}",
    )
    p.add_argument(
        "--pro_agent_results",
        type=str,
        default="data/swebench_pro/responses.jsonl",
        help="Pro response-matrix JSONL: {'subject_id': ..., 'responses': {'item_id': 0/1, ...}}",
    )
    p.add_argument(
        "--terminal_bench_agent_results",
        type=str,
        default="data/terminalbench/responses.jsonl",
        help="Terminal-Bench response-matrix JSONL: {'subject_id': ..., 'responses': {'task_id': 0/1, ...}}",
    )
    p.add_argument(
        "--gso_agent_results",
        type=str,
        default="data/gso/responses.jsonl",
        help="GSO response-matrix JSONL: {'subject_id': ..., 'responses': {'item_id': 0/1, ...}}",
    )
    
    args = p.parse_args(argv)
    base.ensure_dir(args.out_dir)
    base.seed_everything(int(args.seed), deterministic=True)

    method = str(getattr(args, "method", "embedding") or "embedding").strip().lower()
    if method not in {"embedding", "judge", "combined"}:
        raise ValueError(f"Unknown --method: {getattr(args, 'method', None)!r}")

    split_by_raw = str(getattr(args, "split_by", "benchmark") or "benchmark").strip().lower()
    if split_by_raw not in {"benchmark", "task", "agent", "observation", "none"}:
        raise ValueError(f"Unknown --split_by: {getattr(args, 'split_by', None)!r}")
    split_by = str(split_by_raw)

    train_benchmarks = _parse_benchmark_list(str(args.train_benchmarks))
    if len(train_benchmarks) < 1:
        raise ValueError("--train_benchmarks must include at least 1 benchmark.")

    ood_benchmark_raw = str(getattr(args, "ood_benchmark", "") or "").strip()
    disable_benchmark_eval = (split_by == "benchmark") and (ood_benchmark_raw == "")
    ood_benchmark: Optional[str] = None
    if not disable_benchmark_eval:
        ood_benchmark = _canon_benchmark_name(ood_benchmark_raw) if ood_benchmark_raw else None

    if split_by == "benchmark" and not disable_benchmark_eval:
        if ood_benchmark is None:
            raise ValueError(
                "--ood_benchmark is required when --split_by=benchmark "
                "(or pass --ood_benchmark '' to disable evaluation)."
            )
        if ood_benchmark in set(train_benchmarks):
            raise ValueError(
                f"--ood_benchmark={ood_benchmark!r} must not be present in --train_benchmarks={train_benchmarks}."
            )

    irt_model = str(args.irt_model or "1d_1pl").strip().lower()
    if irt_model not in {"1d_1pl", "2d_1pl"}:
        raise ValueError(f"Unknown --irt_model: {args.irt_model!r}")

    def _irt_out_dir_name(model_name: str) -> str:
        m = str(model_name or "").strip().lower()
        return m

    irt_model_label = f"model+scaffold {irt_model.replace('_', ' ')} (shared)"

    train_set = set(train_benchmarks)
    use_verified = "verified" in train_set
    use_pro = "pro" in train_set
    use_terminal = "terminal_bench" in train_set
    use_gso = "gso" in train_set

    agent_results_raw_by_bench: Dict[str, str] = {
        "verified": str(args.verified_agent_results or "").strip(),
        "pro": str(args.pro_agent_results or "").strip(),
        "terminal_bench": str(args.terminal_bench_agent_results or "").strip(),
        "gso": str(args.gso_agent_results or "").strip(),
    }

    tmp = tempfile.TemporaryDirectory(prefix="multibench_tmp_")
    tmp_dir = tmp.name

    agent_results_filtered_by_bench: Dict[str, str] = {}
    for b in ["verified", "pro", "terminal_bench", "gso"]:
        agent_results_filtered_by_bench[b] = ""

    verified_agent_results_raw = ""
    pro_agent_results_raw = ""
    terminal_agent_results_raw = ""
    gso_agent_results_raw = ""

    def _require_path_for_train(b: str) -> str:
        pth = str(agent_results_raw_by_bench.get(b, "") or "").strip()
        if not pth:
            raise ValueError(f"Training benchmark {b!r} requires an agent results JSONL path, but it was empty.")
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Training benchmark {b!r}: agent results JSONL not found: {pth}")
        return pth

    if use_verified:
        verified_agent_results_raw = _require_path_for_train("verified")
        verified_agent_results = os.path.join(tmp_dir, "verified.filtered.jsonl")
        agent_results_filtered_by_bench["verified"] = verified_agent_results
        base.ensure_dir(os.path.dirname(verified_agent_results) or ".")
        shutil.copy(verified_agent_results_raw, verified_agent_results)

    if use_pro:
        pro_agent_results_raw = _require_path_for_train("pro")
        pro_agent_results = os.path.join(tmp_dir, "pro.filtered.jsonl")
        agent_results_filtered_by_bench["pro"] = pro_agent_results
        base.ensure_dir(os.path.dirname(pro_agent_results) or ".")
        shutil.copy(pro_agent_results_raw, pro_agent_results)

    if use_terminal:
        terminal_agent_results_raw = _require_path_for_train("terminal_bench")
        terminal_agent_results = os.path.join(tmp_dir, "terminal_bench.filtered.jsonl")
        agent_results_filtered_by_bench["terminal_bench"] = terminal_agent_results
        base.ensure_dir(os.path.dirname(terminal_agent_results) or ".")
        shutil.copy(terminal_agent_results_raw, terminal_agent_results)

    if use_gso:
        gso_agent_results_raw = _require_path_for_train("gso")
        gso_agent_results = os.path.join(tmp_dir, "gso.filtered.jsonl")
        agent_results_filtered_by_bench["gso"] = gso_agent_results
        base.ensure_dir(os.path.dirname(gso_agent_results) or ".")
        shutil.copy(gso_agent_results_raw, gso_agent_results)

    responses_by_bench: Dict[str, List[Tuple[str, Dict[str, int]]]] = {}
    all_responses_tagged: List[Tuple[str, str, Dict[str, int]]] = []

    if use_verified:
        pth = str(agent_results_filtered_by_bench.get("verified", "") or "").strip()
        responses_by_bench["verified"] = base.load_all_responses(pth)
        all_responses_tagged.extend([("verified", sid, resp) for sid, resp in responses_by_bench["verified"]])
    if use_pro:
        pth = str(agent_results_filtered_by_bench.get("pro", "") or "").strip()
        responses_by_bench["pro"] = base.load_all_responses(pth)
        all_responses_tagged.extend([("pro", sid, resp) for sid, resp in responses_by_bench["pro"]])
    if use_terminal:
        pth = str(agent_results_filtered_by_bench.get("terminal_bench", "") or "").strip()
        responses_by_bench["terminal_bench"] = load_all_responses_terminal(pth)
        all_responses_tagged.extend([("terminal_bench", sid, resp) for sid, resp in responses_by_bench["terminal_bench"]])
    if use_gso:
        pth = str(agent_results_filtered_by_bench.get("gso", "") or "").strip()
        responses_by_bench["gso"] = load_all_responses_generic(path=pth, normalize_item_ids=True)
        all_responses_tagged.extend([("gso", sid, resp) for sid, resp in responses_by_bench["gso"]])

    response_items: Set[str] = set()
    for _, _, resp in all_responses_tagged:
        response_items.update(resp.keys())
    if not response_items:
        raise RuntimeError("Parsed 0 response items from the provided agent results JSONLs (after filtering).")

    flat_for_zero_success: List[Tuple[str, Dict[str, int]]] = [(str(sid), resp) for _, sid, resp in all_responses_tagged]
    zero_success_ids = base.compute_zero_success_items(flat_for_zero_success)
    zero_success_set = set(zero_success_ids)
    exclude_zero_success = not bool(args.include_zero_success)

    item_ids_by_bench: Dict[str, List[str]] = {}
    for b, rows in responses_by_bench.items():
        ids: List[str] = []
        seen: Set[str] = set()
        for _, resp in rows:
            for tid in resp.keys():
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
        item_ids_by_bench[str(b)] = ids

    src_parts: List[str] = []
    if use_verified:
        verified_src = (
            f"verified:{str(args.verified_dataset_name)}:{str(args.verified_split)}"
        )
        src_parts.append(verified_src)
    if use_pro:
        pro_src = (
            f"pro:{str(args.pro_dataset_name)}:{str(args.pro_split)}"
        )
        src_parts.append(pro_src)
    if use_terminal:
        terminal_src = f"terminal_jsonl:{os.path.basename(str(args.terminal_bench_tasks_jsonl)) or 'terminal_bench_tasks.jsonl'}"
        src_parts.append(terminal_src)
    if use_gso:
        gso_src = (
            f"gso:{str(args.gso_dataset_name)}:{str(args.gso_split)}"
        )
        src_parts.append(gso_src)
    dataset_sources_str = " | ".join(src_parts)

    safe_backbone = str(args.backbone).replace("/", "__")
    instr_sig = base.prompt_signature(str(args.instruction))

    task_ids: List[str] = []
    X = None
    id_to_row: Dict[str, int] = {}
    Xy = None
    emb_cache = str(args.embeddings_cache or "").strip()

    if (split_by not in {"agent", "observation", "none"}) and method in {"embedding", "combined"}:
        if not emb_cache:

            cache_meta = {
                "backbone": str(args.backbone),
                "max_length": int(args.max_length),
                "batch_size": int(args.batch_size),
                "device_map": str(args.device_map),
                "torch_dtype": str(args.torch_dtype),
                "attn_implementation": str(args.attn_implementation),
                "instruction": str(args.instruction),
                "instruction_sig": str(instr_sig),
                "embedding_layer": int(args.embedding_layer),
                "include_zero_success": bool(args.include_zero_success),
                "normalize_item_ids": True,
                "embed_subset": "response_items",
                "dataset_sources": str(dataset_sources_str),
            }
            cache_key = hashlib.sha1(json.dumps(cache_meta, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            model_short = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(safe_backbone))[:48].strip("_") or "model"
            short_basename = f"embeddings__{model_short}__{cache_key}__maxlen{int(args.max_length)}.npz"
            emb_cache = os.path.join(args.out_dir, short_basename)

            try:
                meta_path = str(emb_cache).replace(".npz", ".meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "cache_path": str(emb_cache),
                            "cache_key": str(cache_key),
                            "basename": str(short_basename),
                            "meta": cache_meta,
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )
            except Exception:
                pass

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

            if use_verified:
                verified_ids = list(item_ids_by_bench.get("verified", []))
                if exclude_zero_success and zero_success_set:
                    verified_ids = [tid for tid in verified_ids if tid not in zero_success_set]
                if not verified_ids:
                    raise RuntimeError(
                        "Verified training benchmark: 0 item_ids remain after response-driven filtering "
                        f"(include_zero_success={bool(args.include_zero_success)})."
                    )
                verified_items, verified_missing = load_swebench_items_by_ids(
                    dataset_name=str(args.verified_dataset_name),
                    split=str(args.verified_split),
                    item_ids=verified_ids,
                    normalize_item_ids=True,
                )
                if verified_missing:
                    print(
                        f"WARNING: Verified training benchmark: {len(verified_missing)}/{len(verified_ids)} item_ids were not found in the dataset. "
                        f"Example: {verified_missing[:10]}"
                    )
                if not verified_items:
                    raise RuntimeError("Verified training benchmark: loaded 0 items to embed; cannot proceed.")
                items.extend(list(verified_items))

            if use_pro:
                pro_ids = list(item_ids_by_bench.get("pro", []))
                if exclude_zero_success and zero_success_set:
                    pro_ids = [tid for tid in pro_ids if tid not in zero_success_set]
                if not pro_ids:
                    raise RuntimeError(
                        "Pro training benchmark: 0 item_ids remain after response-driven filtering "
                        f"(include_zero_success={bool(args.include_zero_success)})."
                    )
                pro_items, pro_missing = load_swebench_items_by_ids(
                    dataset_name=str(args.pro_dataset_name),
                    split=str(args.pro_split),
                    item_ids=pro_ids,
                    normalize_item_ids=True,
                )
                if pro_missing:
                    print(
                        f"WARNING: Pro training benchmark: {len(pro_missing)}/{len(pro_ids)} item_ids were not found in the dataset. "
                        f"Example: {pro_missing[:10]}"
                    )
                if not pro_items:
                    raise RuntimeError("Pro training benchmark: loaded 0 items to embed; cannot proceed.")
                items.extend(list(pro_items))

            if use_terminal:
                terminal_ids = list(item_ids_by_bench.get("terminal_bench", []))
                if exclude_zero_success and zero_success_set:
                    terminal_ids = [tid for tid in terminal_ids if tid not in zero_success_set]
                if not terminal_ids:
                    raise RuntimeError(
                        "Terminal-Bench training benchmark: 0 task_ids remain after response-driven filtering "
                        f"(include_zero_success={bool(args.include_zero_success)})."
                    )
                terminal_items, terminal_missing = load_terminal_bench_items_by_ids(
                    tasks_jsonl=str(args.terminal_bench_tasks_jsonl),
                    item_ids=terminal_ids,
                )
                if terminal_missing:
                    print(
                        f"WARNING: Terminal-Bench training benchmark: {len(terminal_missing)}/{len(terminal_ids)} task_ids were not found in tasks JSONL. "
                        f"Example: {terminal_missing[:10]}"
                    )
                if not terminal_items:
                    raise RuntimeError("Terminal-Bench training benchmark: loaded 0 items to embed; cannot proceed.")
                items.extend(list(terminal_items))

            if use_gso:
                gso_ids = list(item_ids_by_bench.get("gso", []))
                if exclude_zero_success and zero_success_set:
                    gso_ids = [tid for tid in gso_ids if tid not in zero_success_set]
                if not gso_ids:
                    raise RuntimeError(
                        "GSO training benchmark: 0 item_ids remain after response-driven filtering "
                        f"(include_zero_success={bool(args.include_zero_success)})."
                    )
                gso_dataset_name = str(args.gso_dataset_name or "").strip()
                if not gso_dataset_name:
                    raise ValueError("GSO training requires --gso_dataset_name to load tasks.")
                gso_items, gso_missing = load_ood_items_by_ids(
                    dataset_name=gso_dataset_name,
                    split=str(args.gso_split),
                    item_ids=gso_ids,
                    normalize_item_ids=True,
                    wrap_with_gso_prompt=True,
                )
                if gso_missing:
                    print(
                        f"WARNING: GSO training benchmark: {len(gso_missing)}/{len(gso_ids)} item_ids were not found in the dataset. "
                        f"Example: {gso_missing[:10]}"
                    )
                if not gso_items:
                    raise RuntimeError("GSO training benchmark: loaded 0 items to embed; cannot proceed.")
                items.extend(list(gso_items))

            by_id: Dict[str, base.ItemRecord] = {}
            collisions: List[str] = []
            for it in items:
                iid = str(it.item_id)
                if iid in by_id:
                    collisions.append(iid)
                    continue
                by_id[iid] = it
            if collisions:
                print(
                    f"WARNING: {len(collisions)} duplicate item_ids across benchmarks; keeping first occurrence. "
                    f"Example: {collisions[:10]}"
                )
            items = list(by_id.values())

            print(f"Loaded dataset items to embed: {len(items)} (sources={dataset_sources_str})")

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

        if X is None:
            raise RuntimeError("Internal error: embeddings matrix X was None in embedding/combined mode.")
        id_to_row = {tid: int(i) for i, tid in enumerate(task_ids)}

        overlap_ids = [tid for tid in task_ids if tid in response_items]
        if not overlap_ids:
            raise RuntimeError("No overlap between embedded task_ids and item_ids found in the provided responses.")
    else:
        if str(args.embeddings_cache or "").strip():
            print("NOTE: --method=judge ignores --embeddings_cache (no embedding is performed).")
        emb_cache = ""
        task_ids = sorted([str(t) for t in response_items])
        overlap_ids = list(task_ids)

    if exclude_zero_success:
        eligible = [tid for tid in overlap_ids if tid not in zero_success_set]
        print(
            f"Excluding zero-success items from CV/IRT: {len(overlap_ids) - len(eligible)}/{len(overlap_ids)} items "
            f"(agent_results_by_benchmark={ {k: v for k, v in agent_results_filtered_by_bench.items() if str(v).strip()} })"
        )
    else:
        eligible = list(overlap_ids)
    if not eligible:
        raise RuntimeError("After filtering, no items remain for CV/IRT.")

    if (split_by not in {"agent", "observation", "none"}) and method in {"embedding", "combined"}:
        Xy = base.np.stack([X[id_to_row[tid]] for tid in eligible], axis=0).astype(base.np.float32)

    verified_norm = os.path.join(tmp_dir, "verified.normalized.jsonl")
    pro_norm = os.path.join(tmp_dir, "pro.normalized.jsonl")
    terminal_norm = os.path.join(tmp_dir, "terminal_bench.normalized.jsonl")
    gso_norm = os.path.join(tmp_dir, "gso.normalized.jsonl")

    for pth in [verified_norm, pro_norm, terminal_norm, gso_norm]:
        try:
            base.ensure_dir(os.path.dirname(pth) or ".")
            with open(pth, "w", encoding="utf-8"):
                pass
        except Exception:
            pass

    if use_verified:
        normalize_responses_jsonl(
            in_path=str(agent_results_filtered_by_bench["verified"]),
            out_path=verified_norm,
            benchmark="verified",
            normalize_item_ids=True,
        )
    if use_pro:
        normalize_responses_jsonl(
            in_path=str(agent_results_filtered_by_bench["pro"]),
            out_path=pro_norm,
            benchmark="pro",
            normalize_item_ids=True,
        )
    term_path_for_irt: Optional[str] = None
    if use_terminal and str(agent_results_filtered_by_bench.get("terminal_bench", "") or "").strip():
        normalize_responses_jsonl(
            in_path=str(agent_results_filtered_by_bench["terminal_bench"]),
            out_path=terminal_norm,
            benchmark="terminal_bench",
            normalize_item_ids=False,
        )
        term_path_for_irt = terminal_norm
    gso_path_for_irt: Optional[str] = None
    if use_gso and str(agent_results_filtered_by_bench.get("gso", "") or "").strip():
        normalize_responses_jsonl(
            in_path=str(agent_results_filtered_by_bench["gso"]),
            out_path=gso_norm,
            benchmark="gso",
            normalize_item_ids=True,
        )
        gso_path_for_irt = gso_norm

    ms = _import_shared_irt_module()
    obs_full = ms.load_multibench_split_irt_data(
        verified_path=ms.resolve_path(verified_norm),
        pro_path=ms.resolve_path(pro_norm),
        terminal_bench_path=ms.resolve_path(term_path_for_irt) if term_path_for_irt else None,
        gso_path=ms.resolve_path(gso_path_for_irt) if gso_path_for_irt else None,
    )

    agent_to_ms_pair: Dict[str, Tuple[str, str]] = {}
    try:

        for row in obs_full.agent_split_df.to_dict(orient="records"):
            bench = str(row.get("benchmark", "") or "").strip()
            agent = str(row.get("agent", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            scaffold = str(row.get("scaffold", "") or "").strip()
            if bench and agent and model and scaffold:
                agent_to_ms_pair[f"{bench}::{agent}"] = (model, scaffold)
    except Exception:
        agent_to_ms_pair = {}

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

    use_judge = method in {"judge", "combined"}

    if (split_by not in {"agent", "observation", "none"}) and use_judge and str(regressor_name) not in {"ridge", "ridge_cv"}:
        raise ValueError(f"--method={method!r} requires --regressor to be ridge or ridge_cv (linear is not supported).")

    oracle_root = os.path.join(str(args.out_dir), "irt_oracle")
    oracle_theta_by_agent: Dict[str, float] = {}
    oracle_diff_by_item: Dict[str, float] = {}
    if split_by in {"task", "agent", "observation"}:
        print("Training oracle IRT on all eligible items (train+test; leakage).")
        base.set_torch_determinism(False)
        base.seed_everything(int(args.seed), deterministic=False)
        oracle_theta_by_agent, oracle_diff_by_item = train_standard_irt_1pl_agents(
            all_responses_tagged=all_responses_tagged,
            keep_item_ids=set(eligible),
            epochs=int(args.irt_epochs),
            device=str(args.irt_device),
            seed=int(args.seed),
            out_dir=str(oracle_root),
        )
        base.set_torch_determinism(True)
        print(
            "Oracle IRT training complete. "
            f"labeled_items={len(oracle_diff_by_item)} agents={len(oracle_theta_by_agent)}"
        )

    if split_by in {"agent", "observation"}:

        agent_folds: List[Tuple[set, set]] = []
        if split_by == "agent":
            agent_keys_all = [f"{b}::{sid}" for b, sid, _ in all_responses_tagged]
            agent_folds = _stable_group_kfold(agent_keys_all, n_splits=int(args.cv_folds), seed=int(args.seed))
            fold_iter = list(enumerate(agent_folds, start=1))
        else:
            fold_iter = list(enumerate([None] * int(args.cv_folds), start=1))

        cv_test_auc_folds: List[float] = []
        cv_test_auc_folds_empirical_model: List[float] = []
        cv_test_auc_folds_oracle_irt: List[float] = []
        cv_test_n_obs_folds: List[int] = []

        for fold, fold_payload in fold_iter:
            if split_by == "agent":
                train_agents, test_agents = fold_payload
            else:
                train_agents, test_agents = None, None

            p_emp_by_item: Dict[str, float] = {}
            if split_by == "agent":
                p_emp_by_item, _ = compute_empirical_solve_rate_by_item(
                    all_responses_tagged=all_responses_tagged,
                    train_item_ids=set(eligible),
                    keep_agent_keys=set(train_agents or []),
                )

            fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
            base.ensure_dir(fold_root)
            if split_by == "agent":
                base.save_json(os.path.join(fold_root, "train_agents.json"), {"agents": sorted(list(train_agents or []))})
                base.save_json(os.path.join(fold_root, "test_agents.json"), {"agents": sorted(list(test_agents or []))})
            else:
                base.save_json(
                    os.path.join(fold_root, "split.json"),
                    {"split_by": str(split_by), "fold": int(fold), "cv_folds": int(args.cv_folds)},
                )

            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)

            if split_by == "agent":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_agent_keys=set(train_agents or []),
                )
                keep_obs_train = None
            else:
                def _keep_obs_train(b: str, a: str, t: str) -> bool:
                    key = f"{b}::{a}::{t}"
                    return int(_fold_id_for_group(key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold)

                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                )
                keep_obs_train = _keep_obs_train

            theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
                obs_train=obs_train,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, _irt_out_dir_name(irt_model)),
            )
            base.set_torch_determinism(True)

            if not theta_by_model or not theta_by_scaffold:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 model/scaffold thetas (unexpected).")
            if not diff_by_item:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 item difficulties (unexpected).")

            baseline_theta_by_agent: Dict[str, float] = {}
            baseline_b_by_item: Dict[str, float] = {}
            if split_by == "observation":
                if keep_obs_train is None:
                    raise RuntimeError("Internal error: missing keep_obs_train for --split_by=observation")
                base.set_torch_determinism(False)
                base.seed_everything(int(args.seed), deterministic=False)
                baseline_theta_by_agent, baseline_b_by_item = train_standard_irt_1pl_agents(
                    all_responses_tagged=all_responses_tagged,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=keep_obs_train,
                    epochs=int(args.irt_epochs),
                    device=str(args.irt_device),
                    seed=int(args.seed),
                    out_dir=os.path.join(fold_root, "irt_standard"),
                )
                base.set_torch_determinism(True)

            scores: List[float] = []
            labels: List[int] = []
            scores_emp: List[float] = []
            labels_emp: List[int] = []
            scores_oracle: List[float] = []
            labels_oracle: List[int] = []

            for bench, sid, resp in all_responses_tagged:
                if split_by == "agent":
                    if f"{bench}::{sid}" not in set(test_agents or []):
                        continue
                for item_id, y_obs in resp.items():
                    if split_by == "observation":
                        obs_key = f"{bench}::{sid}::{item_id}"
                        if int(_fold_id_for_group(obs_key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold):
                            continue

                    pair = agent_to_ms_pair.get(f"{bench}::{sid}", None)
                    if pair is None:
                        continue
                    model_name, scaffold = pair
                    tm = theta_by_model.get(model_name, None)
                    ts = theta_by_scaffold.get(scaffold, None)
                    if tm is None or ts is None:
                        continue
                    b = diff_by_item.get(str(item_id), None)
                    if b is None:
                        continue
                    th = _combine_model_scaffold_theta(float(tm), float(ts), combine=str(args.theta_combine))
                    scores.append(base._sigmoid(th - float(b)))
                    labels.append(int(y_obs))

                    if split_by == "observation":
                        th_a = baseline_theta_by_agent.get(str(sid), None)
                        b_a = baseline_b_by_item.get(str(item_id), None)
                        if th_a is not None and b_a is not None:
                            scores_emp.append(base._sigmoid(float(th_a) - float(b_a)))
                            labels_emp.append(int(y_obs))
                    else:

                        p_item = p_emp_by_item.get(str(item_id), None)
                        if p_item is not None:
                            scores_emp.append(float(p_item))
                            labels_emp.append(int(y_obs))

                    th_o = oracle_theta_by_agent.get(str(sid), None)
                    b_o = oracle_diff_by_item.get(str(item_id), None)
                    if th_o is not None and b_o is not None:
                        scores_oracle.append(base._sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

            fold_auc = float(base._compute_binary_auroc(scores, labels))
            fold_auc_emp = float(base._compute_binary_auroc(scores_emp, labels_emp))
            fold_auc_oracle = float(base._compute_binary_auroc(scores_oracle, labels_oracle))
            cv_test_auc_folds.append(float(fold_auc))
            cv_test_auc_folds_empirical_model.append(float(fold_auc_emp))
            cv_test_auc_folds_oracle_irt.append(float(fold_auc_oracle))
            cv_test_n_obs_folds.append(int(len(labels)))
            print(
                f"Fold {fold:02d}: auc={fold_auc} baseline_auc={fold_auc_emp} oracle_auc={fold_auc_oracle}"
            )

        auc_arr = base.np.asarray(cv_test_auc_folds, dtype=base.np.float64)
        auc_mean = float(base.np.nanmean(auc_arr)) if auc_arr.size else float("nan")
        auc_std = float(base.np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")

        baseline_auc_arr = base.np.asarray(cv_test_auc_folds_empirical_model, dtype=base.np.float64)
        baseline_auc_mean = float(base.np.nanmean(baseline_auc_arr)) if baseline_auc_arr.size else float("nan")
        baseline_auc_std = float(base.np.nanstd(baseline_auc_arr, ddof=0)) if baseline_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV baseline ROC-AUC: mean={baseline_auc_mean} std={baseline_auc_std}")

        oracle_auc_arr = base.np.asarray(cv_test_auc_folds_oracle_irt, dtype=base.np.float64)
        oracle_auc_mean = float(base.np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
        oracle_auc_std = float(base.np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

        metrics = {
            "method": str(method),
            "split_by": str(split_by),
            "irt_only": True,
            "baseline_type": ("irt_agent_1pl" if str(split_by) == "observation" else "empirical_item_solve_rate"),
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
            "cv_test_auc_mean_baseline": float(baseline_auc_mean),
            "cv_test_auc_std_baseline": float(baseline_auc_std),
            "cv_test_auc_folds_baseline": [float(x) for x in cv_test_auc_folds_empirical_model],
            "cv_test_auc_mean_oracle_irt": float(oracle_auc_mean),
            "cv_test_auc_std_oracle_irt": float(oracle_auc_std),
            "cv_test_auc_folds_oracle_irt": [float(x) for x in cv_test_auc_folds_oracle_irt],
        }
        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)
        print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
        return 0

    if split_by == "task" and method == "combined":

        outer_cv = None
        agent_folds: List[Tuple[set, set]] = []
        if split_by == "task":
            outer_cv = base.KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        elif split_by == "agent":
            agent_keys_all = [f"{b}::{sid}" for b, sid, _ in all_responses_tagged]
            agent_folds = _stable_group_kfold(agent_keys_all, n_splits=int(args.cv_folds), seed=int(args.seed))
        cv_test_auc_folds: List[float] = []
        cv_test_auc_folds_embedding_only: List[float] = []
        cv_test_auc_folds_empirical_model: List[float] = []
        cv_test_auc_folds_oracle_irt: List[float] = []
        cv_test_n_obs_folds: List[int] = []
        cv_test_n_items_scored_folds: List[int] = []
        yhat_oof = base.np.full((int(len(eligible)),), base.np.nan, dtype=base.np.float64) if split_by == "task" else None
        fold_of_item = base.np.full((int(len(eligible)),), -1, dtype=base.np.int32) if split_by == "task" else None

        eligible_index = {tid: i for i, tid in enumerate(eligible)}

        verified_item_set = set(obs_full.verified_item_ids)
        pro_item_set = set(obs_full.pro_item_ids)
        terminal_item_set = set(obs_full.terminal_bench_item_ids)
        gso_item_set = set(getattr(obs_full, "gso_item_ids", set()))

        judge_dim = int(len(JUDGE_FEATURE_NAMES))
        judge_feature_names_full: List[str] = list(JUDGE_FEATURE_NAMES)

        verified_feat_dir = str(args.verified_judge_features_dir)
        pro_feat_dir = str(args.pro_judge_features_dir)
        terminal_feat_dir = str(args.terminal_bench_judge_features_dir)
        verified_idx = base._build_judge_index(verified_feat_dir, normalize_item_ids=True)
        pro_idx = base._build_judge_index(pro_feat_dir, normalize_item_ids=True)
        terminal_idx = base._build_judge_index(terminal_feat_dir, normalize_item_ids=False)
        gso_feat_dir = str(getattr(args, "gso_judge_features_dir", "") or "").strip()
        gso_idx = base._build_judge_index(gso_feat_dir, normalize_item_ids=True) if (gso_item_set and gso_feat_dir) else {}

        def _judge_full_vec_for_item(item_id: str):
            tid = str(item_id)
            if tid in verified_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=verified_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=verified_idx,
                    normalize_item_ids=True,
                )
                return v
            if tid in pro_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=pro_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=pro_idx,
                    normalize_item_ids=True,
                )
                return v
            if tid in terminal_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=terminal_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=terminal_idx,
                    normalize_item_ids=False,
                )
                return v
            if tid in gso_item_set and gso_feat_dir:
                v = base._load_judge_vector(
                    tid,
                    features_dir=gso_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=gso_idx,
                    normalize_item_ids=True,
                )
                return v
            return None

        best_fold_auc = -float("inf")
        best_fold = -1
        best_joint_state = None
        fold_alpha_emb: List[float] = []
        fold_alpha_judge: List[float] = []

        dummy = base.np.zeros((int(len(eligible)), 1), dtype=base.np.float32)
        if split_by == "task":
            fold_iter = list(enumerate(outer_cv.split(dummy), start=1))
        elif split_by == "agent":
            fold_iter = list(enumerate(agent_folds, start=1))
        else:
            fold_iter = list(enumerate([None] * int(args.cv_folds), start=1))

        for fold, fold_payload in fold_iter:
            if split_by == "task":
                tr, te = fold_payload
                train_items = [eligible[int(i)] for i in tr.tolist()]
                test_items = [eligible[int(i)] for i in te.tolist()]
                train_agents = None
                test_agents = None
            elif split_by == "agent":
                train_agents, test_agents = fold_payload
                train_items = list(eligible)
                test_items = list(eligible)
            else:
                train_items = list(eligible)
                test_items = list(eligible)
                train_agents = None
                test_agents = None

            fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
            base.ensure_dir(fold_root)

            if split_by == "task":
                base.save_json(os.path.join(fold_root, "train_items.json"), {"items": list(train_items)})
                base.save_json(os.path.join(fold_root, "test_items.json"), {"items": list(test_items)})
            elif split_by == "agent":
                base.save_json(os.path.join(fold_root, "train_agents.json"), {"agents": sorted(list(train_agents or []))})
                base.save_json(os.path.join(fold_root, "test_agents.json"), {"agents": sorted(list(test_agents or []))})
            else:
                base.save_json(
                    os.path.join(fold_root, "split.json"),
                    {"split_by": str(split_by), "fold": int(fold), "cv_folds": int(args.cv_folds)},
                )

            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)

            if split_by == "task":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(train_items),
                )
            elif split_by == "agent":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_agent_keys=set(train_agents or []),
                )
            else:
                def _keep_obs_train(b: str, a: str, t: str) -> bool:
                    key = f"{b}::{a}::{t}"
                    return int(_fold_id_for_group(key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold)

                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                )
            theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
                obs_train=obs_train,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, _irt_out_dir_name(irt_model)),
            )

            base.set_torch_determinism(True)

            if not theta_by_model or not theta_by_scaffold:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 model/scaffold thetas (unexpected).")
            if not diff_by_item:
                raise RuntimeError(f"Fold {fold}: IRT produced 0 item difficulties (unexpected).")

            baseline_theta_by_agent: Dict[str, float] = {}
            baseline_b_by_item: Dict[str, float] = {}
            if split_by == "observation":
                base.set_torch_determinism(False)
                base.seed_everything(int(args.seed), deterministic=False)
                if "_keep_obs_train" not in locals():
                    raise RuntimeError("Internal error: _keep_obs_train missing for --split_by=observation")
                baseline_theta_by_agent, baseline_b_by_item = train_standard_irt_1pl_agents(
                    all_responses_tagged=all_responses_tagged,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                    epochs=int(args.irt_epochs),
                    device=str(args.irt_device),
                    seed=int(args.seed),
                    out_dir=os.path.join(fold_root, "irt_standard"),
                )
                base.set_torch_determinism(True)

            train_labeled = [tid for tid in train_items if tid in diff_by_item]
            if len(train_labeled) < 2:
                raise RuntimeError(
                    f"Fold {fold}: only {len(train_labeled)} train items had IRT difficulties; cannot fit regressor."
                )

            base.seed_everything(int(args.seed) + int(fold), deterministic=True)
            X_train = base.np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(base.np.float32)
            y_train = base.np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=base.np.float32)
            emb_model = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed) + int(fold))
            emb_model.fit(X_train, y_train)

            X_test = base.np.stack([X[id_to_row[tid]] for tid in test_items], axis=0).astype(base.np.float32)
            emb_pred_test = emb_model.predict(X_test).astype(base.np.float64)
            emb_pred_by_item_test = {tid: float(z) for tid, z in zip(test_items, emb_pred_test.tolist())}

            p_emp_by_model, _ = compute_empirical_success_prob_by_model(
                all_responses_tagged=all_responses_tagged,
                agent_to_ms_pair=agent_to_ms_pair,
                train_item_ids=set(train_items if split_by == "task" else eligible),
                keep_agent_keys=set(train_agents or []) if split_by == "agent" else None,
            )

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
                joint_state = base._fit_block_ridge(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alpha_emb=float(alpha_emb),
                    alpha_judge=float(alpha_judge),
                )
            else:
                ae_grid_s = str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas)
                aj_grid_s = str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas)
                ae_grid = base._parse_alpha_list(ae_grid_s)
                aj_grid = base._parse_alpha_list(aj_grid_s)
                alpha_emb, alpha_judge, _ = base._select_block_alphas_inner_cv(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alphas_emb=ae_grid,
                    alphas_judge=aj_grid,
                    inner_splits=int(args.inner_splits),
                    seed=int(args.seed) + 2000 + int(fold),
                )
                joint_state = base._fit_block_ridge(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alpha_emb=float(alpha_emb),
                    alpha_judge=float(alpha_judge),
                )

            fold_alpha_emb.append(float(joint_state["alpha_emb"]))
            fold_alpha_judge.append(float(joint_state["alpha_judge"]))

            final_pred_by_item: Dict[str, float] = {}
            n_missing_judge = 0
            if split_by in {"agent", "observation"}:
                final_pred_by_item = {str(tid): float(b) for tid, b in diff_by_item.items()}

                emb_pred_by_item_test = dict(final_pred_by_item)
                n_missing_judge = 0
            else:
                for tid in test_items:
                    jv = _judge_full_vec_for_item(tid)
                    if jv is None:
                        n_missing_judge += 1
                        continue
                    x_emb = X[id_to_row[tid]].reshape(1, -1).astype(base.np.float32)
                    x_j = base.np.asarray(jv, dtype=base.np.float32).reshape(1, -1)
                    final_pred_by_item[tid] = float(base._predict_block_ridge(joint_state, X_emb=x_emb, X_judge=x_j)[0])

            if split_by == "task":
                for tid in test_items:
                    i = eligible_index.get(tid, None)
                    if i is None:
                        continue
                    fold_of_item[int(i)] = int(fold)
                    if tid in final_pred_by_item:
                        yhat_oof[int(i)] = float(final_pred_by_item[tid])

            scored_items = set(final_pred_by_item.keys())
            scores_final: List[float] = []
            scores_emb: List[float] = []
            labels: List[int] = []
            scores_emp: List[float] = []
            labels_emp: List[int] = []
            scores_oracle: List[float] = []
            labels_oracle: List[int] = []
            test_set = set(test_items)

            for bench, sid, resp in all_responses_tagged:
                if split_by == "agent":
                    if f"{bench}::{sid}" not in set(test_agents or []):
                        continue
                key = f"{bench}::{sid}"
                pair = agent_to_ms_pair.get(key, None)
                if pair is None:
                    continue
                model_name, scaffold = pair
                tm = theta_by_model.get(model_name, None)
                ts = theta_by_scaffold.get(scaffold, None)
                if tm is None or ts is None:
                    continue
                th = _combine_model_scaffold_theta(float(tm), float(ts), combine=str(args.theta_combine))
                for item_id, y_obs in resp.items():
                    if split_by == "task":
                        if item_id not in test_set:
                            continue
                    elif split_by == "observation":
                        obs_key = f"{bench}::{sid}::{item_id}"
                        if int(_fold_id_for_group(obs_key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold):
                            continue
                    if item_id not in scored_items:
                        continue
                    z = final_pred_by_item.get(item_id, None)
                    z_emb = emb_pred_by_item_test.get(item_id, None)
                    if z is None or z_emb is None:
                        continue
                    scores_final.append(base._sigmoid(th - float(z)))
                    scores_emb.append(base._sigmoid(th - float(z_emb)))
                    labels.append(int(y_obs))

                    if split_by == "observation":
                        th_a = baseline_theta_by_agent.get(str(sid), None)
                        b_a = baseline_b_by_item.get(str(item_id), None)
                        if th_a is not None and b_a is not None:
                            scores_emp.append(base._sigmoid(float(th_a) - float(b_a)))
                            labels_emp.append(int(y_obs))
                    else:
                        p_emp = p_emp_by_model.get(str(model_name), None)
                        if p_emp is not None:
                            scores_emp.append(float(p_emp))
                            labels_emp.append(int(y_obs))

                    th_o = oracle_theta_by_agent.get(str(sid), None)
                    b_o = oracle_diff_by_item.get(item_id, None)
                    if th_o is not None and b_o is not None:
                        scores_oracle.append(base._sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

            fold_auc = float(base._compute_binary_auroc(scores_final, labels))
            fold_auc_emb = float(base._compute_binary_auroc(scores_emb, labels))
            fold_auc_emp = float(base._compute_binary_auroc(scores_emp, labels_emp))
            fold_auc_oracle = float(base._compute_binary_auroc(scores_oracle, labels_oracle))
            cv_test_auc_folds.append(float(fold_auc))
            cv_test_auc_folds_embedding_only.append(float(fold_auc_emb))
            cv_test_auc_folds_empirical_model.append(float(fold_auc_emp))
            cv_test_auc_folds_oracle_irt.append(float(fold_auc_oracle))
            cv_test_n_obs_folds.append(int(len(labels)))
            cv_test_n_items_scored_folds.append(int(len(scored_items)))
            print(
                f"Fold {fold:02d}: auc={fold_auc} baseline_auc={fold_auc_emp} oracle_auc={fold_auc_oracle}"
            )
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

        baseline_auc_arr = base.np.asarray(cv_test_auc_folds_empirical_model, dtype=base.np.float64)
        baseline_auc_mean = float(base.np.nanmean(baseline_auc_arr)) if baseline_auc_arr.size else float("nan")
        baseline_auc_std = float(base.np.nanstd(baseline_auc_arr, ddof=0)) if baseline_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV baseline ROC-AUC: mean={baseline_auc_mean} std={baseline_auc_std}")

        oracle_auc_arr = base.np.asarray(cv_test_auc_folds_oracle_irt, dtype=base.np.float64)
        oracle_auc_mean = float(base.np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
        oracle_auc_std = float(base.np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

        metrics = {
            "split_by": str(split_by),
            "baseline_type": ("irt_agent_1pl" if str(split_by) == "observation" else "empirical_model_success"),
            "method": "combined",
            "n_items_total": int(len(task_ids)),
            "n_items_with_responses": int(len(overlap_ids)),
            "n_items_eligible_cv_irt": int(len(eligible)),
            "exclude_zero_success": bool(exclude_zero_success),
            "seed": int(args.seed),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_auc_folds_embedding_only": [float(x) for x in cv_test_auc_folds_embedding_only],
            "cv_test_auc_folds_empirical_model_success": [float(x) for x in cv_test_auc_folds_empirical_model],
            "cv_test_auc_folds_oracle_irt": [float(x) for x in cv_test_auc_folds_oracle_irt],
            "cv_test_auc_oracle_irt_mean": float(oracle_auc_mean),
            "cv_test_auc_oracle_irt_std": float(oracle_auc_std),
        }
        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

        if split_by == "task":
            pred_path = os.path.join(args.out_dir, "predictions.csv")
            with open(pred_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
                w.writeheader()
                for i, tid in enumerate(eligible):
                    v = float(yhat_oof[int(i)])
                    fold_id = int(fold_of_item[int(i)]) if int(fold_of_item[int(i)]) > 0 else ""
                    split_s = "cv_val" if (v == v) else "missing_judge"
                    w.writerow({"item_id": tid, "diff_pred": (v if v == v else ""), "split": split_s, "fold": fold_id})
            print(f"Wrote predictions: {pred_path}")
        print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
        return 0

    if split_by == "task" and method == "judge":

        outer_cv = None
        agent_folds: List[Tuple[set, set]] = []
        if split_by == "task":
            outer_cv = base.KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        elif split_by == "agent":
            agent_keys_all = [f"{b}::{sid}" for b, sid, _ in all_responses_tagged]
            agent_folds = _stable_group_kfold(agent_keys_all, n_splits=int(args.cv_folds), seed=int(args.seed))
        cv_test_auc_folds: List[float] = []
        cv_test_auc_folds_empirical_model: List[float] = []
        cv_test_auc_folds_oracle_irt: List[float] = []
        cv_test_n_obs_folds: List[int] = []
        cv_test_n_items_scored_folds: List[int] = []
        yhat_oof = base.np.full((int(len(eligible)),), base.np.nan, dtype=base.np.float64)
        fold_of_item = base.np.full((int(len(eligible)),), -1, dtype=base.np.int32)

        eligible_index = {tid: i for i, tid in enumerate(eligible)}

        verified_item_set = set(obs_full.verified_item_ids)
        pro_item_set = set(obs_full.pro_item_ids)
        terminal_item_set = set(obs_full.terminal_bench_item_ids)
        gso_item_set = set(getattr(obs_full, "gso_item_ids", set()))

        judge_dim = int(len(JUDGE_FEATURE_NAMES))
        judge_feature_names_full: List[str] = list(JUDGE_FEATURE_NAMES)

        verified_feat_dir = str(args.verified_judge_features_dir)
        pro_feat_dir = str(args.pro_judge_features_dir)
        terminal_feat_dir = str(args.terminal_bench_judge_features_dir)
        verified_idx = base._build_judge_index(verified_feat_dir, normalize_item_ids=True)
        pro_idx = base._build_judge_index(pro_feat_dir, normalize_item_ids=True)
        terminal_idx = base._build_judge_index(terminal_feat_dir, normalize_item_ids=False)
        gso_feat_dir = str(getattr(args, "gso_judge_features_dir", "") or "").strip()
        gso_idx = base._build_judge_index(gso_feat_dir, normalize_item_ids=True) if (gso_item_set and gso_feat_dir) else {}

        def _judge_full_vec_for_item(item_id: str):
            tid = str(item_id)
            if tid in verified_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=verified_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=verified_idx,
                    normalize_item_ids=True,
                )
                return v
            if tid in pro_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=pro_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=pro_idx,
                    normalize_item_ids=True,
                )
                return v
            if tid in terminal_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=terminal_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=terminal_idx,
                    normalize_item_ids=False,
                )
                return v
            if tid in gso_item_set and gso_feat_dir:
                v = base._load_judge_vector(
                    tid,
                    features_dir=gso_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=gso_idx,
                    normalize_item_ids=True,
                )
                return v
            return None

        best_fold_auc = -float("inf")
        best_fold = -1
        best_model = None

        if split_by == "task":
            fold_iter = list(enumerate(outer_cv.split(Xy), start=1))
        elif split_by == "agent":
            fold_iter = list(enumerate(agent_folds, start=1))
        else:
            fold_iter = list(enumerate([None] * int(args.cv_folds), start=1))

        for fold, fold_payload in fold_iter:
            if split_by == "task":
                tr, te = fold_payload
                train_items = [eligible[int(i)] for i in tr.tolist()]
                test_items = [eligible[int(i)] for i in te.tolist()]
                train_agents = None
                test_agents = None
            elif split_by == "agent":
                train_agents, test_agents = fold_payload
                train_items = list(eligible)
                test_items = list(eligible)
            else:
                train_items = list(eligible)
                test_items = list(eligible)
                train_agents = None
                test_agents = None

            p_emp_by_model, _ = compute_empirical_success_prob_by_model(
                all_responses_tagged=all_responses_tagged,
                agent_to_ms_pair=agent_to_ms_pair,
                train_item_ids=set(train_items if split_by == "task" else eligible),
                keep_agent_keys=set(train_agents or []) if split_by == "agent" else None,
            )

            fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
            base.ensure_dir(fold_root)

            if split_by == "task":
                base.save_json(os.path.join(fold_root, "train_items.json"), {"items": list(train_items)})
                base.save_json(os.path.join(fold_root, "test_items.json"), {"items": list(test_items)})
            elif split_by == "agent":
                base.save_json(os.path.join(fold_root, "train_agents.json"), {"agents": sorted(list(train_agents or []))})
                base.save_json(os.path.join(fold_root, "test_agents.json"), {"agents": sorted(list(test_agents or []))})
            else:
                base.save_json(
                    os.path.join(fold_root, "split.json"),
                    {"split_by": str(split_by), "fold": int(fold), "cv_folds": int(args.cv_folds)},
                )

            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)

            if split_by == "task":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(train_items),
                )
            elif split_by == "agent":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_agent_keys=set(train_agents or []),
                )
            else:
                def _keep_obs_train(b: str, a: str, t: str) -> bool:
                    key = f"{b}::{a}::{t}"
                    return int(_fold_id_for_group(key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold)

                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                )
            theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
                obs_train=obs_train,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, _irt_out_dir_name(irt_model)),
            )

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

            judge_train_rows = []
            judge_y_train_rows = []
            judge_train_items_used: List[str] = []
            for tid in train_labeled:
                jv = _judge_full_vec_for_item(tid)
                if jv is None:
                    continue
                judge_train_rows.append(base.np.asarray(jv, dtype=base.np.float32).reshape(-1))
                judge_y_train_rows.append(float(diff_by_item[tid]))
                judge_train_items_used.append(tid)
            if len(judge_train_items_used) < 2:
                raise RuntimeError(
                    f"Fold {fold}: only {len(judge_train_items_used)} train items had judge features; cannot fit judge ridge."
                )

            base.seed_everything(int(args.seed) + int(fold), deterministic=True)
            X_judge_train = base.np.stack(judge_train_rows, axis=0).astype(base.np.float32)
            y_judge_train = base.np.asarray(judge_y_train_rows, dtype=base.np.float32)

            m = _make_model(n_train=int(len(judge_train_items_used)), fold_seed=int(args.seed) + int(fold))
            m.fit(X_judge_train, y_judge_train)

            final_pred_by_item: Dict[str, float] = {}
            n_missing_judge = 0
            if split_by in {"agent", "observation"}:
                final_pred_by_item = {str(tid): float(b) for tid, b in diff_by_item.items()}
                n_missing_judge = 0
            else:
                test_items_used: List[str] = []
                test_rows: List[base.np.ndarray] = []
                for tid in test_items:
                    jv = _judge_full_vec_for_item(tid)
                    if jv is None:
                        n_missing_judge += 1
                        continue
                    test_items_used.append(tid)
                    test_rows.append(base.np.asarray(jv, dtype=base.np.float32).reshape(-1))
                if test_items_used:
                    X_judge_test = base.np.stack(test_rows, axis=0).astype(base.np.float32)
                    pred = m.predict(X_judge_test).astype(base.np.float64)
                    for tid, z in zip(test_items_used, pred.tolist()):
                        final_pred_by_item[tid] = float(z)

            if split_by == "task":
                for tid in test_items:
                    i = eligible_index.get(tid, None)
                    if i is None:
                        continue
                    fold_of_item[int(i)] = int(fold)
                    if tid in final_pred_by_item:
                        yhat_oof[int(i)] = float(final_pred_by_item[tid])

            scored_items = set(final_pred_by_item.keys())

            baseline_theta_by_agent: Dict[str, float] = {}
            baseline_b_by_item: Dict[str, float] = {}
            if split_by == "observation":
                base.set_torch_determinism(False)
                base.seed_everything(int(args.seed), deterministic=False)
                if "_keep_obs_train" not in locals():
                    raise RuntimeError("Internal error: _keep_obs_train missing for --split_by=observation")
                baseline_theta_by_agent, baseline_b_by_item = train_standard_irt_1pl_agents(
                    all_responses_tagged=all_responses_tagged,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                    epochs=int(args.irt_epochs),
                    device=str(args.irt_device),
                    seed=int(args.seed),
                    out_dir=os.path.join(fold_root, "irt_standard"),
                )
                base.set_torch_determinism(True)

            scores: List[float] = []
            labels: List[int] = []
            scores_emp: List[float] = []
            labels_emp: List[int] = []
            scores_oracle: List[float] = []
            labels_oracle: List[int] = []
            test_set = set(test_items)

            for bench, sid, resp in all_responses_tagged:
                if split_by == "agent":
                    if f"{bench}::{sid}" not in set(test_agents or []):
                        continue
                key = f"{bench}::{sid}"
                pair = agent_to_ms_pair.get(key, None)
                if pair is None:
                    continue
                model_name, scaffold = pair
                tm = theta_by_model.get(model_name, None)
                ts = theta_by_scaffold.get(scaffold, None)
                if tm is None or ts is None:
                    continue
                th = _combine_model_scaffold_theta(float(tm), float(ts), combine=str(args.theta_combine))
                for item_id, y_obs in resp.items():
                    if split_by == "task":
                        if item_id not in test_set:
                            continue
                    elif split_by == "observation":
                        obs_key = f"{bench}::{sid}::{item_id}"
                        if int(_fold_id_for_group(obs_key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold):
                            continue
                    if item_id not in scored_items:
                        continue
                    z = final_pred_by_item.get(item_id, None)
                    if z is None:
                        continue
                    scores.append(base._sigmoid(th - float(z)))
                    labels.append(int(y_obs))

                    if split_by == "observation":
                        th_a = baseline_theta_by_agent.get(str(sid), None)
                        b_a = baseline_b_by_item.get(str(item_id), None)
                        if th_a is not None and b_a is not None:
                            scores_emp.append(base._sigmoid(float(th_a) - float(b_a)))
                            labels_emp.append(int(y_obs))
                    else:
                        p_emp = p_emp_by_model.get(str(model_name), None)
                        if p_emp is not None:
                            scores_emp.append(float(p_emp))
                            labels_emp.append(int(y_obs))

                    th_o = oracle_theta_by_agent.get(str(sid), None)
                    b_o = oracle_diff_by_item.get(item_id, None)
                    if th_o is not None and b_o is not None:
                        scores_oracle.append(base._sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

            fold_auc = float(base._compute_binary_auroc(scores, labels))
            fold_auc_emp = float(base._compute_binary_auroc(scores_emp, labels_emp))
            fold_auc_oracle = float(base._compute_binary_auroc(scores_oracle, labels_oracle))
            cv_test_auc_folds.append(float(fold_auc))
            cv_test_auc_folds_empirical_model.append(float(fold_auc_emp))
            cv_test_auc_folds_oracle_irt.append(float(fold_auc_oracle))
            cv_test_n_obs_folds.append(int(len(labels)))
            cv_test_n_items_scored_folds.append(int(len(scored_items)))

            print(
                f"Fold {fold:02d}: auc={fold_auc} baseline_auc={fold_auc_emp} oracle_auc={fold_auc_oracle}"
            )
            if fold_auc == fold_auc and fold_auc > best_fold_auc:
                best_fold_auc = float(fold_auc)
                best_fold = int(fold)
                best_model = m

        if best_model is None or best_fold < 1:
            raise RuntimeError("Failed to select a best CV fold judge model by ROC-AUC (all folds NaN?).")

        auc_arr = base.np.asarray(cv_test_auc_folds, dtype=base.np.float64)
        auc_mean = float(base.np.nanmean(auc_arr)) if auc_arr.size else float("nan")
        auc_std = float(base.np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")

        baseline_auc_arr = base.np.asarray(cv_test_auc_folds_empirical_model, dtype=base.np.float64)
        baseline_auc_mean = float(base.np.nanmean(baseline_auc_arr)) if baseline_auc_arr.size else float("nan")
        baseline_auc_std = float(base.np.nanstd(baseline_auc_arr, ddof=0)) if baseline_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV baseline ROC-AUC: mean={baseline_auc_mean} std={baseline_auc_std}")

        oracle_auc_arr = base.np.asarray(cv_test_auc_folds_oracle_irt, dtype=base.np.float64)
        oracle_auc_mean = float(base.np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
        oracle_auc_std = float(base.np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

        model = best_model

        metrics = {
            "split_by": str(split_by),
            "baseline_type": ("irt_agent_1pl" if str(split_by) == "observation" else "empirical_model_success"),
            "method": "judge",
            "n_items_total": int(len(task_ids)),
            "n_items_with_responses": int(len(overlap_ids)),
            "n_items_eligible_cv_irt": int(len(eligible)),
            "exclude_zero_success": bool(exclude_zero_success),
            "seed": int(args.seed),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_auc_folds_empirical_model_success": [float(x) for x in cv_test_auc_folds_empirical_model],
            "cv_test_auc_folds_oracle_irt": [float(x) for x in cv_test_auc_folds_oracle_irt],
            "cv_test_auc_oracle_irt_mean": float(oracle_auc_mean),
            "cv_test_auc_oracle_irt_std": float(oracle_auc_std),
        }
        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

        if split_by == "task":

            pred_path = os.path.join(args.out_dir, "predictions.csv")
            with open(pred_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
                w.writeheader()
                for i, tid in enumerate(eligible):
                    v = float(yhat_oof[int(i)])
                    fold_id = int(fold_of_item[int(i)]) if int(fold_of_item[int(i)]) > 0 else ""
                    split_s = "cv_val" if (v == v) else "missing_judge"
                    w.writerow({"item_id": tid, "diff_pred": (v if v == v else ""), "split": split_s, "fold": fold_id})
            print(f"Wrote predictions: {pred_path}")
        print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
        return 0

    if split_by == "task":

        outer_cv = None
        agent_folds: List[Tuple[set, set]] = []
        if split_by == "task":
            outer_cv = base.KFold(n_splits=int(args.cv_folds), shuffle=True, random_state=int(args.seed))
        elif split_by == "agent":
            agent_keys_all = [f"{b}::{sid}" for b, sid, _ in all_responses_tagged]
            agent_folds = _stable_group_kfold(agent_keys_all, n_splits=int(args.cv_folds), seed=int(args.seed))
        cv_test_auc_folds: List[float] = []
        cv_test_auc_folds_empirical_model: List[float] = []
        cv_test_auc_folds_oracle_irt: List[float] = []
        cv_test_n_obs_folds: List[int] = []
        yhat_oof = base.np.full((int(len(eligible)),), base.np.nan, dtype=base.np.float64)
        fold_of_item = base.np.full((int(len(eligible)),), -1, dtype=base.np.int32)

        best_fold_auc = -float("inf")
        best_fold = -1
        best_model = None

        if split_by == "task":
            fold_iter = list(enumerate(outer_cv.split(Xy), start=1))
        elif split_by == "agent":
            fold_iter = list(enumerate(agent_folds, start=1))
        else:
            fold_iter = list(enumerate([None] * int(args.cv_folds), start=1))

        for fold, fold_payload in fold_iter:
            if split_by == "task":
                tr, te = fold_payload
                train_items = [eligible[int(i)] for i in tr.tolist()]
                test_items = [eligible[int(i)] for i in te.tolist()]
                train_agents = None
                test_agents = None
            elif split_by == "agent":
                train_agents, test_agents = fold_payload
                train_items = list(eligible)
                test_items = list(eligible)
            else:
                train_items = list(eligible)
                test_items = list(eligible)
                train_agents = None
                test_agents = None

            p_emp_by_model, _ = compute_empirical_success_prob_by_model(
                all_responses_tagged=all_responses_tagged,
                agent_to_ms_pair=agent_to_ms_pair,
                train_item_ids=set(train_items if split_by == "task" else eligible),
                keep_agent_keys=set(train_agents or []) if split_by == "agent" else None,
            )

            fold_root = os.path.join(str(args.out_dir), "irt_folds", f"fold_{int(fold):02d}")
            base.ensure_dir(fold_root)

            if split_by == "task":
                base.save_json(os.path.join(fold_root, "train_items.json"), {"items": list(train_items)})
                base.save_json(os.path.join(fold_root, "test_items.json"), {"items": list(test_items)})
            elif split_by == "agent":
                base.save_json(os.path.join(fold_root, "train_agents.json"), {"agents": sorted(list(train_agents or []))})
                base.save_json(os.path.join(fold_root, "test_agents.json"), {"agents": sorted(list(test_agents or []))})
            else:
                base.save_json(
                    os.path.join(fold_root, "split.json"),
                    {"split_by": str(split_by), "fold": int(fold), "cv_folds": int(args.cv_folds)},
                )

            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)

            if split_by == "task":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(train_items),
                )
            elif split_by == "agent":
                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_agent_keys=set(train_agents or []),
                )
            else:
                def _keep_obs_train(b: str, a: str, t: str) -> bool:
                    key = f"{b}::{a}::{t}"
                    return int(_fold_id_for_group(key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold)

                obs_train = build_multibench_obs_from_tagged_responses(
                    all_responses_tagged=all_responses_tagged,
                    agent_to_ms_pair=agent_to_ms_pair,
                    obs_full_agent_split_df=obs_full.agent_split_df,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                )
            theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
                obs_train=obs_train,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, _irt_out_dir_name(irt_model)),
            )

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
            if split_by == "task" and yhat_oof is not None and fold_of_item is not None:
                yhat_oof[te] = pred
                fold_of_item[te] = int(fold)

            z_by_item = {tid: float(z) for tid, z in zip(test_items, pred.tolist())}
            if split_by in {"agent", "observation"}:

                z_by_item = {str(tid): float(b) for tid, b in diff_by_item.items()}

            baseline_theta_by_agent: Dict[str, float] = {}
            baseline_b_by_item: Dict[str, float] = {}
            if split_by == "observation":
                base.set_torch_determinism(False)
                base.seed_everything(int(args.seed), deterministic=False)
                if "_keep_obs_train" not in locals():
                    raise RuntimeError("Internal error: _keep_obs_train missing for --split_by=observation")
                baseline_theta_by_agent, baseline_b_by_item = train_standard_irt_1pl_agents(
                    all_responses_tagged=all_responses_tagged,
                    keep_item_ids=set(eligible),
                    keep_obs_fn=_keep_obs_train,
                    epochs=int(args.irt_epochs),
                    device=str(args.irt_device),
                    seed=int(args.seed),
                    out_dir=os.path.join(fold_root, "irt_standard"),
                )
                base.set_torch_determinism(True)
            scores: List[float] = []
            labels: List[int] = []
            scores_emp: List[float] = []
            labels_emp: List[int] = []
            scores_oracle: List[float] = []
            labels_oracle: List[int] = []
            test_set = set(test_items)

            for bench, sid, resp in all_responses_tagged:
                if split_by == "agent":
                    if f"{bench}::{sid}" not in set(test_agents or []):
                        continue
                key = f"{bench}::{sid}"
                pair = agent_to_ms_pair.get(key, None)
                if pair is None:
                    continue
                model, scaffold = pair
                tm = theta_by_model.get(model, None)
                ts = theta_by_scaffold.get(scaffold, None)
                if tm is None or ts is None:
                    continue
                th = _combine_model_scaffold_theta(float(tm), float(ts), combine=str(args.theta_combine))
                for item_id, y_obs in resp.items():
                    if split_by == "task":
                        if item_id not in test_set:
                            continue
                    elif split_by == "observation":
                        obs_key = f"{bench}::{sid}::{item_id}"
                        if int(_fold_id_for_group(obs_key, n_splits=int(args.cv_folds), seed=int(args.seed))) != int(fold):
                            continue
                    z = z_by_item.get(item_id, None)
                    if z is None:
                        continue
                    scores.append(base._sigmoid(th - float(z)))
                    labels.append(int(y_obs))

                    if split_by == "observation":
                        th_a = baseline_theta_by_agent.get(str(sid), None)
                        b_a = baseline_b_by_item.get(str(item_id), None)
                        if th_a is not None and b_a is not None:
                            scores_emp.append(base._sigmoid(float(th_a) - float(b_a)))
                            labels_emp.append(int(y_obs))
                    else:
                        p_emp = p_emp_by_model.get(str(model), None)
                        if p_emp is not None:
                            scores_emp.append(float(p_emp))
                            labels_emp.append(int(y_obs))

                    th_o = oracle_theta_by_agent.get(str(sid), None)
                    b_o = oracle_diff_by_item.get(item_id, None)
                    if th_o is not None and b_o is not None:
                        scores_oracle.append(base._sigmoid(float(th_o) - float(b_o)))
                        labels_oracle.append(int(y_obs))

            fold_auc = float(base._compute_binary_auroc(scores, labels))
            fold_auc_emp = float(base._compute_binary_auroc(scores_emp, labels_emp))
            fold_auc_oracle = float(base._compute_binary_auroc(scores_oracle, labels_oracle))
            cv_test_auc_folds.append(float(fold_auc))
            cv_test_auc_folds_empirical_model.append(float(fold_auc_emp))
            cv_test_auc_folds_oracle_irt.append(float(fold_auc_oracle))
            cv_test_n_obs_folds.append(int(len(labels)))
            print(
                f"Fold {fold:02d}: auc={fold_auc} baseline_auc={fold_auc_emp} oracle_auc={fold_auc_oracle}"
            )
            if fold_auc == fold_auc and fold_auc > best_fold_auc:
                best_fold_auc = float(fold_auc)
                best_fold = int(fold)
                best_model = m

        if split_by == "task":
            if yhat_oof is None or fold_of_item is None:
                raise RuntimeError("Internal error: expected OOF arrays for --split_by=task.")
            if base.np.isnan(yhat_oof).any() or (fold_of_item < 0).any():
                raise RuntimeError("KFold CV produced incomplete out-of-fold predictions (unexpected).")
        if best_model is None or best_fold < 1:
            raise RuntimeError("Failed to select a best CV fold model by ROC-AUC (all folds NaN?).")

        auc_arr = base.np.asarray(cv_test_auc_folds, dtype=base.np.float64)
        auc_mean = float(base.np.nanmean(auc_arr)) if auc_arr.size else float("nan")
        auc_std = float(base.np.nanstd(auc_arr, ddof=0)) if auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV test ROC-AUC: mean={auc_mean} std={auc_std}")

        baseline_auc_arr = base.np.asarray(cv_test_auc_folds_empirical_model, dtype=base.np.float64)
        baseline_auc_mean = float(base.np.nanmean(baseline_auc_arr)) if baseline_auc_arr.size else float("nan")
        baseline_auc_std = float(base.np.nanstd(baseline_auc_arr, ddof=0)) if baseline_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV baseline ROC-AUC: mean={baseline_auc_mean} std={baseline_auc_std}")

        oracle_auc_arr = base.np.asarray(cv_test_auc_folds_oracle_irt, dtype=base.np.float64)
        oracle_auc_mean = float(base.np.nanmean(oracle_auc_arr)) if oracle_auc_arr.size else float("nan")
        oracle_auc_std = float(base.np.nanstd(oracle_auc_arr, ddof=0)) if oracle_auc_arr.size else float("nan")
        print(f"{int(args.cv_folds)}-fold CV oracle ROC-AUC: mean={oracle_auc_mean} std={oracle_auc_std}")

        model = best_model

        metrics = {
            "split_by": str(split_by),
            "baseline_type": ("irt_agent_1pl" if str(split_by) == "observation" else "empirical_model_success"),
            "n_items_total": int(len(task_ids)),
            "n_items_with_responses": int(len(overlap_ids)),
            "n_items_eligible_cv_irt": int(len(eligible)),
            "exclude_zero_success": bool(exclude_zero_success),
            "seed": int(args.seed),
            "cv_best_auc_fold": int(best_fold),
            "cv_best_auc": float(best_fold_auc),
            "cv_test_auc_folds": [float(x) for x in cv_test_auc_folds],
            "cv_test_auc_mean": float(auc_mean),
            "cv_test_auc_std": float(auc_std),
            "cv_test_auc_folds_empirical_model_success": [float(x) for x in cv_test_auc_folds_empirical_model],
            "cv_test_auc_folds_oracle_irt": [float(x) for x in cv_test_auc_folds_oracle_irt],
            "cv_test_auc_oracle_irt_mean": float(oracle_auc_mean),
            "cv_test_auc_oracle_irt_std": float(oracle_auc_std),
        }

        zero_embedded: List[str] = []
        yhat_zero: Optional[base.np.ndarray] = None
        if bool(exclude_zero_success) and zero_success_set:
            zero_embedded = [tid for tid in task_ids if tid in zero_success_set]
            if zero_embedded:
                X_zero = base.np.stack([X[id_to_row[tid]] for tid in zero_embedded], axis=0).astype(base.np.float32)
                yhat_zero = model.predict(X_zero).astype(base.np.float64)
            else:
                print("NOTE: zero-success ids provided, but none were present in embedded task_ids; nothing to predict.")

        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)

        if split_by == "task":
            if yhat_oof is None or fold_of_item is None:
                raise RuntimeError("Internal error: expected OOF arrays for --split_by=task.")

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

    train_items = list(eligible)
    if not train_items:
        raise RuntimeError(
            "Training-only mode: after filtering (include_zero_success="
            f"{bool(args.include_zero_success)}), no train items remain."
        )

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
    print(
        f"IRT training complete. items_train={len(train_items)} labeled_items={len(diff_by_item)} "
        f"models={len(theta_by_model)} scaffolds={len(theta_by_scaffold)}"
    )

    if split_by == "none":
        metrics = {
            "method": str(method),
            "split_by": str(split_by),
            "irt_only": True,
            "items_train": int(len(train_items)),
            "labeled_items": int(len(diff_by_item)),
            "models": int(len(theta_by_model)),
            "scaffolds": int(len(theta_by_scaffold)),
            "irt_out_dir": os.path.join(str(args.out_dir), _irt_out_dir_name(irt_model)),
        }
        base.save_json(os.path.join(args.out_dir, "metrics.json"), metrics)
        print(f"Wrote metrics: {os.path.join(args.out_dir, 'metrics.json')}")
        return 0

    train_labeled = [tid for tid in train_items if tid in diff_by_item]
    if len(train_labeled) < 2:
        raise RuntimeError(
            f"Training-only mode: only {len(train_labeled)} train items had IRT difficulties; cannot fit regressor."
        )

    base.seed_everything(int(args.seed), deterministic=True)
    y_train = base.np.array([float(diff_by_item[tid]) for tid in train_labeled], dtype=base.np.float32)

    model = None
    joint_state = None
    judge_feature_names_full: List[str] = []
    judge_dim = 0

    if method == "embedding":
        if X is None or not id_to_row:
            raise RuntimeError("Internal error: embeddings were not available for --method=embedding.")
        X_train = base.np.stack([X[id_to_row[tid]] for tid in train_labeled], axis=0).astype(base.np.float32)
        model = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed))
        model.fit(X_train, y_train)
    else:
        print(
            f"Building judge feature training matrix for {len(train_labeled)} labeled items "
            f"(verified={len(obs_full.verified_item_ids)}, pro={len(obs_full.pro_item_ids)}, "
            f"terminal_bench={len(obs_full.terminal_bench_item_ids)}, gso={len(getattr(obs_full, 'gso_item_ids', set()))}; "
            f"method={method})"
        )
        verified_item_set = set(obs_full.verified_item_ids)
        pro_item_set = set(obs_full.pro_item_ids)
        terminal_item_set = set(obs_full.terminal_bench_item_ids)
        gso_item_set = set(getattr(obs_full, "gso_item_ids", set()))

        judge_dim = int(len(JUDGE_FEATURE_NAMES))
        judge_feature_names_full = list(JUDGE_FEATURE_NAMES)

        verified_feat_dir = str(args.verified_judge_features_dir)
        pro_feat_dir = str(args.pro_judge_features_dir)
        terminal_feat_dir = str(args.terminal_bench_judge_features_dir)
        verified_idx = base._build_judge_index(verified_feat_dir, normalize_item_ids=True)
        pro_idx = base._build_judge_index(pro_feat_dir, normalize_item_ids=True)
        terminal_idx = base._build_judge_index(terminal_feat_dir, normalize_item_ids=False)
        gso_feat_dir = str(getattr(args, "gso_judge_features_dir", "") or "").strip()
        gso_idx = base._build_judge_index(gso_feat_dir, normalize_item_ids=True) if (gso_item_set and gso_feat_dir) else {}

        def _judge_full_vec_for_item(item_id: str):
            tid = str(item_id)
            if tid in verified_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=verified_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=verified_idx,
                    normalize_item_ids=True,
                )
                return v
            if tid in pro_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=pro_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=pro_idx,
                    normalize_item_ids=True,
                )
                return v
            if tid in terminal_item_set:
                v = base._load_judge_vector(
                    tid,
                    features_dir=terminal_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=terminal_idx,
                    normalize_item_ids=False,
                )
                return v
            if tid in gso_item_set and gso_feat_dir:
                v = base._load_judge_vector(
                    tid,
                    features_dir=gso_feat_dir,
                    feature_names=JUDGE_FEATURE_NAMES,
                    index=gso_idx,
                    normalize_item_ids=True,
                )
                return v
            return None

        joint_judge_train_rows = []
        joint_y_train_rows = []
        joint_train_items_used: List[str] = []
        missing_emb = 0
        for i, tid in enumerate(train_labeled, start=1):
            if i == 1 or i % 200 == 0 or i == len(train_labeled):
                print(
                    f"Loading judge features: {i}/{len(train_labeled)} items "
                    f"(usable_so_far={len(joint_train_items_used)})"
                )
            jv = _judge_full_vec_for_item(tid)
            if jv is None:
                continue

            if method in {"combined"} and (tid not in id_to_row):
                missing_emb += 1
                continue
            joint_judge_train_rows.append(base.np.asarray(jv, dtype=base.np.float32))
            joint_y_train_rows.append(float(diff_by_item[tid]))
            joint_train_items_used.append(tid)
        if len(joint_train_items_used) < 2:
            raise RuntimeError(
                f"Training-only mode: only {len(joint_train_items_used)} train items had judge features; cannot fit judge-based regressor."
            )
        if missing_emb > 0 and method in {"combined"}:
            print(
                f"NOTE: Skipped {int(missing_emb)}/{len(train_labeled)} labeled items with judge features "
                f"but missing embeddings (method={method})."
            )
        print(f"Judge features loaded for {len(joint_train_items_used)}/{len(train_labeled)} labeled items.")

        X_judge_joint_train = base.np.stack(joint_judge_train_rows, axis=0).astype(base.np.float32)
        y_joint_train = base.np.asarray(joint_y_train_rows, dtype=base.np.float32)

        if method == "judge":
            print(f"Fitting judge-only ridge (regressor={regressor_name}).")
            model = _make_model(n_train=int(len(joint_train_items_used)), fold_seed=int(args.seed))
            model.fit(X_judge_joint_train, y_joint_train)
        else:

            print(f"Fitting joint block-ridge (regressor={regressor_name}).")
            X_emb_joint_train = base.np.stack(
                [X[id_to_row[tid]].astype(base.np.float32) for tid in joint_train_items_used], axis=0
            ).astype(base.np.float32)

            reg = str(regressor_name or "ridge_cv").strip()
            if reg == "ridge":
                alpha_emb = float(args.ridge_alpha_emb) if math.isfinite(float(args.ridge_alpha_emb)) else float(args.ridge_alpha)
                alpha_judge = (
                    float(args.ridge_alpha_judge) if math.isfinite(float(args.ridge_alpha_judge)) else float(args.ridge_alpha)
                )
                joint_state = base._fit_block_ridge(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alpha_emb=float(alpha_emb),
                    alpha_judge=float(alpha_judge),
                )
            else:
                ae_grid_s = str(args.ridge_alphas_emb or "").strip() or str(args.ridge_alphas)
                aj_grid_s = str(args.ridge_alphas_judge or "").strip() or str(args.ridge_alphas)
                ae_grid = base._parse_alpha_list(ae_grid_s)
                aj_grid = base._parse_alpha_list(aj_grid_s)
                alpha_emb, alpha_judge, _ = base._select_block_alphas_inner_cv(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alphas_emb=ae_grid,
                    alphas_judge=aj_grid,
                    inner_splits=int(args.inner_splits),
                    seed=int(args.seed) + 2000,
                    verbose=True,
                )
                joint_state = base._fit_block_ridge(
                    X_emb=X_emb_joint_train,
                    X_judge=X_judge_joint_train,
                    y=y_joint_train,
                    alpha_emb=float(alpha_emb),
                    alpha_judge=float(alpha_judge),
                )

    ridge_alpha = None
    if regressor_name in ("ridge", "ridge_cv"):
        if method in {"embedding", "judge"}:
            try:
                ridge_alpha = float(model.named_steps["ridge"].alpha_)
            except Exception:
                ridge_alpha = None

    weights_meta = {
        "method": str(method),
        "split_by": str(split_by),
        "disable_benchmark_eval": bool(disable_benchmark_eval),
        "ood_benchmark": (str(ood_benchmark) if ood_benchmark is not None else ""),
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
    if method == "embedding":
        base.save_regression_weights(
            out_dir=str(args.out_dir),
            model=model,
            regressor_name=str(regressor_name),
            feature_dim=int(Xy.shape[1]),
            metadata=weights_meta,
        )
    elif method == "judge":
        weights_meta.update(
            {
                "verified_judge_features_dir": str(args.verified_judge_features_dir),
                "pro_judge_features_dir": str(args.pro_judge_features_dir),
                "terminal_bench_judge_features_dir": str(args.terminal_bench_judge_features_dir),
                "judge_feature_names": list(judge_feature_names_full),
                "judge_feature_dim": int(judge_dim),
            }
        )
        base.save_regression_weights(
            out_dir=str(args.out_dir),
            model=model,
            regressor_name=str(regressor_name),
            feature_dim=int(judge_dim),
            metadata=weights_meta,
        )
    else:
        if joint_state is None:
            raise RuntimeError("Internal error: joint_state was None after joint training.")
        weights_meta.update(
            {
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
        base.save_regression_weights_block_ridge(
            out_dir=str(args.out_dir),
            state=joint_state,
            judge_feature_names=judge_feature_names_full,
            metadata=weights_meta,
        )

    if split_by == "benchmark" and not disable_benchmark_eval:

        ood_key = str(ood_benchmark)

        ood_agent_results: str = ""
        ood_normalize_item_ids: bool = True
        ood_treat_as_pro: bool = False
        ood_default_scaffold: Optional[str] = None
        ood_feat_dir: str = ""
        ood_dataset_name: str = ""
        ood_split: str = "test"

        if ood_key == "verified":
            ood_agent_results = str(args.verified_agent_results or "").strip()
            ood_normalize_item_ids = True
            ood_treat_as_pro = False
            ood_default_scaffold = None
            ood_feat_dir = str(getattr(args, "verified_judge_features_dir", "") or "").strip()
            ood_dataset_name = str(args.verified_dataset_name or "").strip()
            ood_split = str(args.verified_split)
        elif ood_key == "pro":
            ood_agent_results = str(args.pro_agent_results or "").strip()
            ood_normalize_item_ids = True
            ood_treat_as_pro = True
            ood_default_scaffold = None
            ood_feat_dir = str(getattr(args, "pro_judge_features_dir", "") or "").strip()
            ood_dataset_name = str(args.pro_dataset_name or "").strip()
            ood_split = str(args.pro_split)
        elif ood_key == "terminal_bench":
            ood_agent_results = str(args.terminal_bench_agent_results or "").strip()
            ood_normalize_item_ids = False
            ood_treat_as_pro = False
            ood_default_scaffold = None
            ood_feat_dir = str(getattr(args, "terminal_bench_judge_features_dir", "") or "").strip()
        elif ood_key == "gso":
            ood_agent_results = str(args.gso_agent_results or "").strip()
            ood_normalize_item_ids = True
            ood_treat_as_pro = False
            ood_default_scaffold = "OpenHands"
            ood_feat_dir = str(getattr(args, "gso_judge_features_dir", "") or "").strip()
            ood_dataset_name = str(args.gso_dataset_name or "").strip()
            ood_split = str(args.gso_split)
        else:
            raise ValueError(f"Unsupported OOD benchmark: {ood_key!r}")

        if not ood_agent_results:
            raise ValueError(f"OOD benchmark {ood_key!r}: agent results JSONL path was empty.")
        if not os.path.exists(ood_agent_results):
            raise FileNotFoundError(f"OOD benchmark {ood_key!r}: agent results JSONL not found: {ood_agent_results}")

        ood_subject_responses: List[Tuple[str, Dict[str, int]]] = []
        ood_item_ids: List[str] = []
        ood_item_set: Set[str] = set()
        for sid, resp in iter_subject_responses_jsonl_generic(
            ood_agent_results, normalize_item_ids=bool(ood_normalize_item_ids)
        ):
            ood_subject_responses.append((str(sid), resp))
            for tid in resp.keys():
                if tid not in ood_item_set:
                    ood_item_set.add(tid)
                    ood_item_ids.append(tid)
        if not ood_item_ids:
            raise RuntimeError(f"OOD agent results had 0 items after parsing: {ood_agent_results}")

        ood_zero_success_ids: List[str] = []
        ood_zero_success_set: Set[str] = set()
        try:
            ood_zero_success_ids = base.compute_zero_success_items(ood_subject_responses)
            ood_zero_success_set = set(ood_zero_success_ids)
        except Exception:
            ood_zero_success_ids = []
            ood_zero_success_set = set()

        if bool(exclude_zero_success) and ood_zero_success_set:
            before = int(len(ood_item_ids))
            ood_item_ids = [tid for tid in ood_item_ids if tid not in ood_zero_success_set]
            removed = int(before - len(ood_item_ids))
            if removed > 0:
                print(
                    f"Excluding zero-success items from OOD benchmark: {removed}/{before} items "
                    f"(ood_benchmark={ood_key}, agent_results={ood_agent_results})"
                )
        if not ood_item_ids:
            raise RuntimeError("OOD benchmark: after excluding zero-success items, 0 items remain to embed/evaluate.")

        ood_items: List[base.ItemRecord] = []
        ood_missing: List[str] = []
        if method in {"embedding", "combined"}:
            if ood_key in {"verified", "pro"}:
                if not ood_dataset_name:
                    raise ValueError(f"OOD benchmark {ood_key!r} requires dataset_name to load tasks.")
                ood_items, ood_missing = load_swebench_items_by_ids(
                    dataset_name=ood_dataset_name,
                    split=str(ood_split),
                    item_ids=ood_item_ids,
                    normalize_item_ids=True,
                )
            elif ood_key == "terminal_bench":
                ood_items, ood_missing = load_terminal_bench_items_by_ids(
                    tasks_jsonl=str(args.terminal_bench_tasks_jsonl),
                    item_ids=ood_item_ids,
                )
            else:

                if not ood_dataset_name:
                    raise ValueError("OOD mode requires a dataset_name to load OOD benchmark tasks.")
                ood_items, ood_missing = load_ood_items_by_ids(
                    dataset_name=ood_dataset_name,
                    split=str(ood_split),
                    item_ids=ood_item_ids,
                    normalize_item_ids=bool(ood_normalize_item_ids),
                    wrap_with_gso_prompt=(ood_key == "gso"),
                )
            if ood_missing:
                print(
                    f"WARNING: OOD benchmark: {len(ood_missing)}/{len(ood_item_ids)} item_ids were not found in the dataset. "
                    f"Example: {ood_missing[:10]}"
                )
            if not ood_items:
                raise RuntimeError("OOD benchmark: loaded 0 items to embed; cannot evaluate AUROC.")

        ood_feat_dir_effective = str(ood_feat_dir or "").strip()
        ood_idx: Dict[str, str] = (
            base._build_judge_index(ood_feat_dir_effective, normalize_item_ids=bool(ood_normalize_item_ids))
            if ood_feat_dir_effective
            else {}
        )

        def _ood_judge_full_vec_for_item(item_id: str):
            tid = str(item_id)
            return base._load_judge_vector(
                tid,
                features_dir=ood_feat_dir_effective,
                feature_names=JUDGE_FEATURE_NAMES,
                index=ood_idx,
                normalize_item_ids=bool(ood_normalize_item_ids),
            )

        z_by_item: Dict[str, float] = {}
        if method in {"embedding", "combined"}:

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
            if method == "embedding":
                z_pred = model.predict(X_ood).astype(base.np.float64)
                z_by_item = {iid: float(z) for iid, z in zip(ood_ids_sorted, z_pred.tolist())}
            elif method == "combined":
                if joint_state is None:
                    raise RuntimeError("Internal error: joint_state was None for OOD evaluation in combined mode.")
                if int(judge_dim) != int(joint_state["n_judge"]):
                    raise RuntimeError("Joint ridge: internal judge_dim mismatch vs trained model (cannot predict).")

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
                        "OOD benchmark: 0 items had OOD judge features; cannot run combined (emb+judge) prediction. "
                        "Provide per-item judge feature JSONs or run with --method=embedding."
                    )

                X_emb_ood_used = base.np.stack(ood_emb_rows, axis=0).astype(base.np.float32)
                X_judge_ood_used = base.np.stack(ood_judge_rows, axis=0).astype(base.np.float32)
                z_pred_used = base._predict_block_ridge(joint_state, X_emb=X_emb_ood_used, X_judge=X_judge_ood_used).astype(base.np.float64)
                z_by_item = {iid: float(z) for iid, z in zip(ood_ids_used, z_pred_used.tolist())}
        else:

            ood_ids_used: List[str] = []
            ood_judge_rows: List[base.np.ndarray] = []
            for iid in ood_item_ids:
                jv = _ood_judge_full_vec_for_item(iid) if ood_feat_dir else None
                if jv is None:
                    continue
                ood_ids_used.append(str(iid))
                ood_judge_rows.append(base.np.asarray(jv, dtype=base.np.float32).reshape(-1))

            if not ood_ids_used:
                raise RuntimeError(
                    "OOD benchmark: 0 items had OOD judge features; cannot run judge-only prediction. "
                    "Provide per-item judge feature JSONs or run with --method=embedding."
                )

            X_judge_ood_used = base.np.stack(ood_judge_rows, axis=0).astype(base.np.float32)
            z_pred_used = model.predict(X_judge_ood_used).astype(base.np.float64)
            z_by_item = {iid: float(z) for iid, z in zip(ood_ids_used, z_pred_used.tolist())}

        pred_path = os.path.join(str(args.out_dir), "predictions.csv")
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
            w.writeheader()
            for iid, z in (z_by_item or {}).items():
                w.writerow({"item_id": str(iid), "diff_pred": float(z), "split": "ood", "fold": ""})
        print(f"Wrote predictions: {pred_path}")

        ood_auc, ood_meta = evaluate_ood_auroc(
            ood_agent_results_jsonl=ood_agent_results,
            ood_normalize_item_ids=bool(ood_normalize_item_ids),
            ood_treat_as_pro=bool(ood_treat_as_pro),
            ood_default_scaffold=ood_default_scaffold,
            z_by_item=z_by_item,
            theta_by_model=theta_by_model,
            theta_by_scaffold=theta_by_scaffold,
            theta_combine=str(args.theta_combine),
        )
        print(f"OOD ROC-AUC: {ood_auc}")

        p_emp_by_model, p_emp_meta = compute_empirical_success_prob_by_model(
            all_responses_tagged=all_responses_tagged,
            agent_to_ms_pair=agent_to_ms_pair,
            train_item_ids=set(train_items),
        )
        ood_emp_auc, ood_emp_meta = evaluate_empirical_model_success_auroc(
            agent_results_jsonl=str(ood_agent_results),
            normalize_item_ids=bool(ood_normalize_item_ids),
            treat_as_pro=bool(ood_treat_as_pro),
            ood_default_scaffold=ood_default_scaffold,
            p_success_by_model=p_emp_by_model,
        )
        print(f"Baseline ROC-AUC: {ood_emp_auc}")
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

        ood_oracle_auc = float("nan")
        ood_oracle_meta: Dict[str, object] = {}
        try:
            ood_oracle_filtered = os.path.join(str(tmp_dir), f"{ood_key}.oracle.filtered.jsonl")
            ood_oracle_norm = os.path.join(str(tmp_dir), f"{ood_key}.oracle.normalized.jsonl")
            base.ensure_dir(os.path.dirname(ood_oracle_filtered) or ".")
            shutil.copy(str(ood_agent_results), str(ood_oracle_filtered))
            normalize_responses_jsonl(
                in_path=str(ood_oracle_filtered),
                out_path=str(ood_oracle_norm),
                benchmark=str(ood_key),
                normalize_item_ids=bool(ood_normalize_item_ids),
            )

            oracle_verified_path = str(ood_oracle_norm) if ood_key == "verified" else str(verified_norm)
            oracle_pro_path = str(ood_oracle_norm) if ood_key == "pro" else str(pro_norm)
            oracle_terminal_path = (
                str(ood_oracle_norm) if ood_key == "terminal_bench" else (str(term_path_for_irt) if term_path_for_irt else None)
            )
            oracle_gso_path = str(ood_oracle_norm) if ood_key == "gso" else (str(gso_path_for_irt) if gso_path_for_irt else None)

            obs_oracle_full = ms.load_multibench_split_irt_data(
                verified_path=ms.resolve_path(oracle_verified_path),
                pro_path=ms.resolve_path(oracle_pro_path),
                terminal_bench_path=ms.resolve_path(oracle_terminal_path) if oracle_terminal_path else None,
                gso_path=ms.resolve_path(oracle_gso_path) if oracle_gso_path else None,
            )

            oracle_items = sorted(set([str(x) for x in list(obs_oracle_full.item_ids)]))
            base.set_torch_determinism(False)
            base.seed_everything(int(args.seed), deterministic=False)
            obs_oracle = build_multibench_obs_for_items(obs_full=obs_oracle_full, keep_item_ids=oracle_items)
            oracle_theta_by_model_ood, oracle_theta_by_scaffold_ood, oracle_diff_by_item_ood = train_irt_model_scaffold_1pl(
                obs_train=obs_oracle,
                irt_model=str(irt_model),
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(str(oracle_root), _irt_out_dir_name(irt_model)),
            )
            base.set_torch_determinism(True)

            z_oracle_by_item = {
                str(iid): float(oracle_diff_by_item_ood[str(iid)])
                for iid in list(z_by_item.keys())
                if str(iid) in oracle_diff_by_item_ood
            }
            ood_oracle_auc, ood_oracle_meta = evaluate_ood_auroc(
                ood_agent_results_jsonl=ood_agent_results,
                ood_normalize_item_ids=bool(ood_normalize_item_ids),
                ood_treat_as_pro=bool(ood_treat_as_pro),
                ood_default_scaffold=ood_default_scaffold,
                z_by_item=z_oracle_by_item,
                theta_by_model=oracle_theta_by_model_ood,
                theta_by_scaffold=oracle_theta_by_scaffold_ood,
                theta_combine=str(args.theta_combine),
            )
            print(f"Oracle ROC-AUC: {ood_oracle_auc}")
        except Exception as e:
            try:
                base.set_torch_determinism(True)
            except Exception:
                pass
            ood_oracle_auc = float("nan")
            ood_oracle_meta = {"error": f"{type(e).__name__}: {e}"}
            print(f"WARNING: Failed to compute OOD oracle IRT AUROC: {type(e).__name__}: {e}")

        base.save_json(
            os.path.join(str(args.out_dir), "metrics.json"),
            {
                "method": str(method),
                "ood_benchmark": str(ood_key),
                "theta_combine": str(args.theta_combine),
                "auc": float(ood_auc),
                "empirical_model_success_auc": float(ood_emp_auc),
                "oracle_irt_auc": float(ood_oracle_auc),
                "agent_results": str(ood_agent_results),
                "dataset_name": (ood_dataset_name or None),
                "split": str(ood_split),
                "normalize_item_ids": bool(ood_normalize_item_ids),
                "ood_judge_features_dir": str(ood_feat_dir or "").strip() or None,
                "empirical_model_success_train_meta": dict(p_emp_meta),
                **{f"empirical_model_success_{k}": v for k, v in (ood_emp_meta or {}).items()},
                **ood_meta,
                **{f"oracle_irt_{k}": v for k, v in (ood_oracle_meta or {}).items()},
            },
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())