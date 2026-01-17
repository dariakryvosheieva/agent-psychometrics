#!/usr/bin/env python3
"""
Compute ROC-AUC for an IRT-style success model with *two* ability parameters:

  p(success) = sigmoid( (theta_model[model(agent)] + theta_scaffold[scaffold(agent)]) - z_item )

This is analogous to `auroc.py`, but replaces the single-agent theta with an
additive (model + scaffold) theta.

Inputs
------
- model abilities CSV: from `swebench_irt/train_model_scaffold.py` (model_abilities.csv)
- scaffold abilities CSV: from `swebench_irt/train_model_scaffold.py` (scaffold_abilities.csv)
- agent map CSV: `out/chris_irt/agent_model_scaffold.csv` (columns: agent,model,scaffold,...)
- item difficulty predictions CSV: same format as `auroc.py` expects (predictions.csv with diff_pred, split)
- responses JSONL: agent×task response matrix JSONL (subject_id, responses)

Output
------
Prints ROC-AUC for the "train" vs "test" split from the predictions CSV, computed
over all (agent,item) pairs that have:
  - an observed response
  - a difficulty prediction
  - a model + scaffold ability
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import re


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency '{pkg}': {e}") from e


_require("torch")
_require("torchmetrics")

import torch
from torchmetrics import AUROC


ROOT = Path(__file__).resolve().parent

_V_SUFFIX_RE = re.compile(r"-v.*$")


def normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def _split_multi_arg(values: object) -> List[str]:
    """
    Parse multi-value CLI args that may be provided as:
    - a single string "a,b,c"
    - a list/tuple of strings ["a", "b,c"]
    - repeated flags collected by argparse into a list
    Returns a flat list of non-empty strings.
    """
    if values is None:
        return []
    if isinstance(values, str):
        s = values.strip()
        if not s:
            return []
        return [p.strip() for p in s.split(",") if p.strip()]
    if isinstance(values, (list, tuple)):
        out: List[str] = []
        for v in values:
            out.extend(_split_multi_arg(v))
        return out
    s = str(values).strip()
    return [s] if s else []


def load_thetas_csv(path: Path) -> Dict[str, float]:
    """
    Loads an abilities CSV with columns: <index>,theta,theta_std
    where the first column is the identifier (agent/model/scaffold).
    """
    out: Dict[str, float] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fns = list(r.fieldnames or [])
        if not fns:
            raise ValueError(f"Empty CSV or missing header row: {path}")
        if "theta" not in fns:
            raise ValueError(f"Expected column 'theta' in {path}, got columns={fns}")
        id_col = fns[0]
        for row in r:
            sid = str(row.get(id_col, "") or "").strip()
            if not sid:
                continue
            theta_s = str(row.get("theta", "") or "").strip()
            if not theta_s:
                continue
            out[sid] = float(theta_s)
    return out


def load_agent_map_csv(path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Loads a mapping CSV with columns: agent, model, scaffold
    Returns: agent -> (model, scaffold)
    """
    out: Dict[str, Tuple[str, str]] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fns = set(r.fieldnames or [])
        need = {"agent", "model", "scaffold"}
        missing = sorted(need - fns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}; got columns={sorted(fns)}")
        for row in r:
            agent = str(row.get("agent", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            scaffold = str(row.get("scaffold", "") or "").strip()
            if not agent or not model or not scaffold:
                continue
            out[agent] = (model, scaffold)
    return out


def _get_scaffold_theta(scaffold: str, scaffold_thetas: Dict[str, float]) -> float | None:
    """
    Return theta for scaffold.

    Policy:
    - Normalize any "no scaffold" spellings to the single canonical label "NoScaffold"
    - Otherwise require an exact scaffold id match (no backward-compat fallbacks)
    """
    s = (scaffold or "").strip()
    low = s.lower()
    if low in {"", "none", "null", "nan", "na", "n/a", "noscaffold"}:
        s = "NoScaffold"
    return scaffold_thetas.get(s)


def load_zs_predictions_csv(path: Path) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fns = set(r.fieldnames or [])
        need = {"item_id", "diff_pred", "split"}
        missing = sorted(need - fns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}; got columns={sorted(fns)}")

        for row in r:
            item_id = normalize_swebench_item_id(str(row.get("item_id", "") or "").strip())
            split = str(row.get("split", "") or "").strip().lower()
            z_s = str(row.get("diff_pred", "") or "").strip()
            if not item_id or split not in ("train", "test", "inference") or not z_s:
                continue
            out[item_id] = (float(z_s), split)
    return out


def iter_responses_jsonl(path: Path) -> Iterable[Tuple[str, Dict[str, int]]]:
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            sid = str(obj.get("subject_id", "")).strip()
            responses = obj.get("responses", {}) or {}
            if not sid or not isinstance(responses, dict):
                continue
            out: Dict[str, int] = {}
            for k, v in responses.items():
                try:
                    item_id = normalize_swebench_item_id(str(k))
                    if not item_id:
                        continue
                    out[item_id] = int(v)
                except Exception:
                    continue
            yield sid, out


def _compute_auc(probs: List[float], labels: List[int]) -> float:
    if len(probs) == 0:
        return float("nan")
    uniq = set(int(x) for x in labels)
    if len(uniq) < 2:
        return float("nan")
    auroc = AUROC(task="binary")
    p = torch.tensor(probs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return float(auroc(p, y).item())


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_thetas",
        type=str,
        default="./out/chris_irt/swebench_model_scaffold_shared/1d_1pl/model_abilities.csv",
        help="Path to model_abilities.csv (expects columns: <index>,theta,theta_std).",
    )
    ap.add_argument(
        "--scaffold_thetas",
        type=str,
        default="./out/chris_irt/swebench_model_scaffold_shared/1d_1pl/scaffold_abilities.csv",
        help="Path to scaffold_abilities.csv (expects columns: <index>,theta,theta_std).",
    )
    ap.add_argument(
        "--agent_map",
        type=str,
        default="./out/chris_irt/swebench_model_scaffold_shared/agent_model_scaffold.csv",
        help="Path to agent_model_scaffold.csv (expects columns: agent,model,scaffold,...).",
    )
    ap.add_argument(
        "--zs",
        type=str,
        default="./out/swebench_model_scaffold_shared/predictions.csv",
        help="Path to predictions.csv (expects columns: item_id,diff_true,diff_pred,split).",
    )
    ap.add_argument(
        "--responses",
        type=str,
        nargs="*",
        default=["./out/chris_irt/swebench_verified_20251115_full.jsonl", "./out/chris_irt/swebench_pro.jsonl", "./out/chris_irt/terminal_bench_2.0.jsonl"],
        help=(
            "Path(s) to JSONL(s) with per-agent response dicts (expects keys: subject_id, responses). "
            "Accepts space-separated paths and/or comma-separated within a token."
        ),
    )
    args = ap.parse_args(argv)

    model_thetas_path = resolve_path(args.model_thetas)
    scaffold_thetas_path = resolve_path(args.scaffold_thetas)
    agent_map_path = resolve_path(args.agent_map)
    zs_path = resolve_path(args.zs)
    responses_paths = [resolve_path(p) for p in _split_multi_arg(args.responses)]
    if not responses_paths:
        raise ValueError("Expected at least one --responses JSONL path.")

    model_thetas = load_thetas_csv(model_thetas_path)
    scaffold_thetas = load_thetas_csv(scaffold_thetas_path)
    agent_map = load_agent_map_csv(agent_map_path)
    z_by_item = load_zs_predictions_csv(zs_path)

    train_probs: List[float] = []
    train_labels: List[int] = []
    test_probs: List[float] = []
    test_labels: List[int] = []

    items = list(z_by_item.keys())
    zs = torch.tensor([z_by_item[i][0] for i in items], dtype=torch.float32)
    splits = [z_by_item[i][1] for i in items]

    subjects_used: set[str] = set()
    for responses_path in responses_paths:
        for agent, responses in iter_responses_jsonl(responses_path):
            pair = agent_map.get(agent)
            if pair is None:
                continue
            model, scaffold = pair
            theta_s = _get_scaffold_theta(scaffold, scaffold_thetas)
            if model not in model_thetas or theta_s is None:
                continue

            theta_eff = float(model_thetas[model]) + float(theta_s)

            idxs: List[int] = []
            ys: List[int] = []
            for j, item_id in enumerate(items):
                if item_id not in responses:
                    continue
                idxs.append(j)
                ys.append(int(responses[item_id]))
            if not idxs:
                continue

            z_sub = zs[idxs]
            probs = torch.sigmoid(torch.tensor(theta_eff, dtype=torch.float32) - z_sub).tolist()
            subjects_used.add(agent)

            for p, y, j in zip(probs, ys, idxs):
                if splits[j] == "train":
                    train_probs.append(float(p))
                    train_labels.append(int(y))
                else:
                    test_probs.append(float(p))
                    test_labels.append(int(y))

    train_auc = _compute_auc(train_probs, train_labels)
    test_auc = _compute_auc(test_probs, test_labels)
    union_auc = _compute_auc(train_probs + test_probs, train_labels + test_labels)

    print(f"Used subjects (intersection of responses and agent_map and abilities): {len(subjects_used)}")
    print(f"Train pairs: {len(train_labels)}  Test pairs: {len(test_labels)}")
    print(f"Union pairs (train∪test): {len(train_labels) + len(test_labels)}")
    print(f"Train ROC-AUC: {train_auc}")
    print(f"Test  ROC-AUC: {test_auc}")
    print(f"Union ROC-AUC: {union_auc}")
    if train_auc != train_auc:
        print("WARNING: Train ROC-AUC is NaN (likely only one class present in train labels).")
    if test_auc != test_auc:
        print("WARNING: Test ROC-AUC is NaN (likely only one class present in test labels).")
    if union_auc != union_auc:
        print("WARNING: Union ROC-AUC is NaN (likely only one class present in union labels).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

