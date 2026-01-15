#!/usr/bin/env python3
"""
Compute ROC-AUC for an IRT-style success model using:

  probs = sigmoid(thetas - zs)

where:
- thetas come from an abilities CSV (agents / test-takers)
- zs come from a per-item difficulty prediction CSV (tasks / items)
- labels are the observed binary responses (agent solved task: 1, else 0)

We respect the item train/test split provided in the predictions CSV and report
two separate ROC-AUC scores.

Why labels are binary responses:
The reference implementation computes AUROC from predicted probabilities vs
the observed response matrix (0/1) for each (test-taker, item) pair.

See: https://github.com/sangttruong/reeval/blob/main/calibration/calibration.ipynb
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
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


# Resolve relative paths from the `fulcrum/fellowship/` directory (where this script lives),
# so defaults like `./out/...` work when invoked from anywhere.
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


def load_thetas_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        fns = list(r.fieldnames or [])
        if not fns:
            raise ValueError(f"Empty CSV or missing header row: {path}")
        if "theta" not in fns:
            raise ValueError(f"Expected column 'theta' in {path}, got columns={fns}")
        id_col = fns[0]  # first unnamed column holding agent ids
        for row in r:
            sid = str(row.get(id_col, "") or "").strip()
            if not sid:
                continue
            theta_s = str(row.get("theta", "") or "").strip()
            if not theta_s:
                continue
            out[sid] = float(theta_s)
    return out


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
            # Coerce to int 0/1 best-effort.
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
        # AUROC is undefined if only one class is present.
        return float("nan")
    auroc = AUROC(task="binary")
    p = torch.tensor(probs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return float(auroc(p, y).item())


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--thetas",
        type=str,
        default="./out/chris_irt/swebench_verified_20251120_full/1d_1pl/abilities.csv",
        help="Path to abilities.csv (expects columns: <index>,theta,theta_std).",
    )
    ap.add_argument(
        "--zs",
        type=str,
        default="./out/swebench_verified/predictions.csv",
        help="Path to predictions.csv (expects columns: item_id,diff_true,diff_pred,split).",
    )
    ap.add_argument(
        "--responses",
        type=str,
        default="./out/chris_irt/swebench_verified_20251120_full.jsonl",
        help="Path to JSONL with per-agent response dicts (expects keys: subject_id, responses).",
    )
    args = ap.parse_args(argv)

    thetas_path = resolve_path(args.thetas)
    zs_path = resolve_path(args.zs)
    responses_path = resolve_path(args.responses)

    thetas = load_thetas_csv(thetas_path)
    z_by_item = load_zs_predictions_csv(zs_path)

    # Accumulate per-split predictions and labels across all (agent, item) pairs
    # that have both a z and an observed response.
    train_probs: List[float] = []
    train_labels: List[int] = []
    test_probs: List[float] = []
    test_labels: List[int] = []

    # Pre-materialize item vectors for per-subject vectorized scoring.
    items = list(z_by_item.keys())
    zs = torch.tensor([z_by_item[i][0] for i in items], dtype=torch.float32)
    splits = [z_by_item[i][1] for i in items]

    n_subjects_used = 0
    for subject_id, responses in iter_responses_jsonl(responses_path):
        if subject_id not in thetas:
            continue
        theta = float(thetas[subject_id])
        # Build labels for items in order, but only keep items with observed responses.
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
        # Your requested sign convention:
        probs = torch.sigmoid(torch.tensor(theta, dtype=torch.float32) - z_sub).tolist()
        n_subjects_used += 1

        for p, y, j in zip(probs, ys, idxs):
            if splits[j] == "train":
                train_probs.append(float(p))
                train_labels.append(int(y))
            else:
                test_probs.append(float(p))
                test_labels.append(int(y))

    train_auc = _compute_auc(train_probs, train_labels)
    test_auc = _compute_auc(test_probs, test_labels)

    print(f"Used subjects (intersection of responses and abilities): {n_subjects_used}")
    print(f"Train pairs: {len(train_labels)}  Test pairs: {len(test_labels)}")
    print(f"Train ROC-AUC: {train_auc}")
    print(f"Test  ROC-AUC: {test_auc}")
    if train_auc != train_auc:
        print("WARNING: Train ROC-AUC is NaN (likely only one class present in train labels).")
    if test_auc != test_auc:
        print("WARNING: Test ROC-AUC is NaN (likely only one class present in test labels).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

