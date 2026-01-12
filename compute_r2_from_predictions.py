#!/usr/bin/env python3
"""
Compute R^2 from a predictions CSV like the ones written by:
  `predict_question_difficulty_qs_solution_instruction.py`

Expected default columns:
  - diff_true
  - diff_pred
Optional:
  - split (e.g. train/test) for per-split metrics

Example:
  /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/python \
    /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/compute_r2_from_predictions.py \
    --predictions_csv /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/verified_qs_sol_instr_qwen25coder14b_lr/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"Missing dependency 'numpy'. Original error: {e}") from e


_require_numpy()
import numpy as np  # noqa: E402


@dataclass(frozen=True)
class R2Result:
    n: int
    r2: float


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 = 1 - SS_res / SS_tot. Matches sklearn.metrics.r2_score for 1D inputs.
    Returns NaN if y_true has zero variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_bar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_predictions(
    *,
    predictions_csv: str,
    y_true_col: str,
    y_pred_col: str,
    split_col: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    p = Path(predictions_csv)
    if not p.exists():
        raise FileNotFoundError(f"predictions_csv not found: {predictions_csv}")

    y_true: List[float] = []
    y_pred: List[float] = []
    splits: List[str] = []

    with p.open(newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        for c in (y_true_col, y_pred_col):
            if c not in fieldnames:
                raise ValueError(f"Missing required column {c!r} in {predictions_csv}; got columns={fieldnames}")

        has_split = split_col in fieldnames

        for row in r:
            yt = _to_float(row.get(y_true_col))
            yp = _to_float(row.get(y_pred_col))
            if yt is None or yp is None:
                continue
            y_true.append(float(yt))
            y_pred.append(float(yp))
            if has_split:
                splits.append(str(row.get(split_col, "") or "").strip())

    splits_out = splits if splits else None
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64), splits_out


def compute_r2(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: Optional[List[str]],
) -> Tuple[R2Result, Dict[str, R2Result]]:
    overall = R2Result(n=int(y_true.size), r2=float(r2_score_np(y_true, y_pred)))
    by_split: Dict[str, R2Result] = {}
    if splits is None:
        return overall, by_split

    splits_arr = np.asarray(splits, dtype=object)
    for s in sorted(set(splits_arr.tolist())):
        mask = splits_arr == s
        yt = y_true[mask]
        yp = y_pred[mask]
        by_split[str(s)] = R2Result(n=int(yt.size), r2=float(r2_score_np(yt, yp)))
    return overall, by_split


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", type=str, required=True)
    ap.add_argument("--y_true_col", type=str, default="diff_true")
    ap.add_argument("--y_pred_col", type=str, default="diff_pred")
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--json_out", type=str, default="", help="Optional path to write results as JSON.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    y_true, y_pred, splits = load_predictions(
        predictions_csv=str(args.predictions_csv),
        y_true_col=str(args.y_true_col),
        y_pred_col=str(args.y_pred_col),
        split_col=str(args.split_col),
    )
    overall, by_split = compute_r2(y_true=y_true, y_pred=y_pred, splits=splits)

    print(f"predictions_csv: {args.predictions_csv}")
    print(f"columns: y_true={args.y_true_col!r}, y_pred={args.y_pred_col!r}, split_col={args.split_col!r}")
    print(f"R^2 (overall): {overall.r2}  (n={overall.n})")
    if by_split:
        print("R^2 by split:")
        for k, v in by_split.items():
            print(f"  {k}: R^2={v.r2}  (n={v.n})")

    if str(args.json_out).strip():
        out = {
            "predictions_csv": str(args.predictions_csv),
            "y_true_col": str(args.y_true_col),
            "y_pred_col": str(args.y_pred_col),
            "split_col": str(args.split_col),
            "overall": {"r2": float(overall.r2), "n": int(overall.n)},
            "by_split": {k: {"r2": float(v.r2), "n": int(v.n)} for k, v in by_split.items()},
        }
        Path(str(args.json_out)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.json_out)).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"Wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





