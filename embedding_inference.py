#!/usr/bin/env python3
"""
Apply a saved regressor (exported by `predict_question_difficulty.py`) to a *new* set of tasks.

The "new set" is specified by a CSV containing item ids, like an IRT `items.csv`:
  ,b,b_std
  repo__name-<hash>,6.73,2.91
  ...

This script will:
  - Read item_ids from the CSV (supports explicit `item_id` as well as blank first-column header).
  - Load tasks from a HF dataset hub split (or a local JSON/JSONL via `datasets`).
  - Select only the tasks whose ids appear in the CSV.
  - Embed them using the same prompt formatting (question+solution+instruction) as training.
  - Apply saved regression weights (coef/intercept + optional scaler) to produce diff_pred.
  - Write `predictions.csv` and `metrics.json` to --out_dir.

Example:
  /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/.venv/bin/python \
    /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/predict_question_difficulty_apply_saved_regressor.py \
    --weights_dir /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/qwen25coder14b_qs_sol_instr_lr \
    --items_csv /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/chris_irt/swebench_pro/1d_1pl/items.csv \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --hf_split test \
    --out_dir /orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/apply_saved_regressor_example
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Reuse embedding + id normalization logic from the training script.
import predict_question_difficulty as pqd


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


def load_item_ids_from_csv(path: str, *, allowed_splits: Optional[Set[str]] = None) -> Tuple[List[str], Dict[str, float]]:
    """
    Load item ids (ordered) from a CSV. Also returns optional labels if present.

    Supports schemas:
      - item_id, diff
      - item_id, diff_true
      - item_id, b
      - <blank_first_col>, b, b_std
    """
    ordered: List[str] = []
    labels: Dict[str, float] = {}

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        fns = list(r.fieldnames or [])
        if not fns:
            raise ValueError(f"Empty CSV or missing header row: {path}")

        # Determine id/label columns (mirrors `pqd.load_ground_truth_csv`).
        if "item_id" in fns:
            id_col = "item_id"
        elif "instance_id" in fns:
            id_col = "instance_id"
        elif "id" in fns:
            id_col = "id"
        else:
            id_col = fns[0]

        y_col: Optional[str] = None
        if "diff" in fns:
            y_col = "diff"
        elif "diff_true" in fns:
            # Common in our per-item predictions CSVs.
            y_col = "diff_true"
        elif "b" in fns:
            y_col = "b"
        elif "difficulty" in fns:
            y_col = "difficulty"

        for row in r:
            # Optional filtering on an existing `split` column (e.g. train/test/zero_success).
            if allowed_splits is not None and "split" in fns:
                split_val = str(row.get("split", "") or "").strip().lower()
                if split_val not in allowed_splits:
                    continue
            raw_id = str(row.get(id_col, "") or "").strip()
            if not raw_id:
                continue
            tid = pqd.normalize_swebench_item_id(raw_id)
            if not tid:
                continue
            ordered.append(tid)

            if y_col is not None:
                v = _to_float(row.get(y_col))
                if v is not None:
                    labels[tid] = float(v)

    # De-duplicate while preserving order (common if CSV is concatenated).
    seen = set()
    deduped: List[str] = []
    for tid in ordered:
        if tid in seen:
            continue
        seen.add(tid)
        deduped.append(tid)
    return deduped, labels


def load_saved_weights(weights_dir: str) -> Tuple[dict, dict]:
    weights_json = os.path.join(weights_dir, "regression_weights.json")
    weights_npz = os.path.join(weights_dir, "regression_weights.npz")
    if not os.path.exists(weights_json):
        raise FileNotFoundError(f"Missing saved weights JSON: {weights_json}")
    if not os.path.exists(weights_npz):
        raise FileNotFoundError(f"Missing saved weights NPZ: {weights_npz}")
    meta = json.loads(open(weights_json, "r", encoding="utf-8").read())
    data = np.load(weights_npz, allow_pickle=True)
    arrays = {
        "coef": np.asarray(data["coef"], dtype=np.float32).reshape(-1),
        "intercept": float(np.asarray(data["intercept"], dtype=np.float32).reshape(-1)[0]),
        "scaler_mean": np.asarray(data["scaler_mean"], dtype=np.float32).reshape(-1),
        "scaler_scale": np.asarray(data["scaler_scale"], dtype=np.float32).reshape(-1),
    }
    return meta, arrays


def apply_weights(X: np.ndarray, *, arrays: dict, uses_scaler: bool) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    coef = np.asarray(arrays["coef"], dtype=np.float32).reshape(-1)
    intercept = float(arrays["intercept"])
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D [N,D], got shape={X.shape}")
    if int(X.shape[1]) != int(coef.size):
        raise ValueError(f"Feature dim mismatch: X has D={X.shape[1]} but coef has D={coef.size}")

    if uses_scaler:
        mean = np.asarray(arrays["scaler_mean"], dtype=np.float32).reshape(-1)
        scale = np.asarray(arrays["scaler_scale"], dtype=np.float32).reshape(-1)
        if int(mean.size) != int(coef.size) or int(scale.size) != int(coef.size):
            raise ValueError("Scaler stats dim mismatch with coef.")
        safe_scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32, copy=False)
        X = (X - mean) / safe_scale

    y = X.dot(coef) + intercept
    return np.asarray(y, dtype=np.float64).reshape(-1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", type=str, required=True, help="Training out_dir containing regression_weights.{json,npz}.")
    ap.add_argument(
        "--items_csv",
        type=str,
        required=True,
        help=(
            "CSV of item_ids to predict for. Supports IRT items.csv as well as per-item predictions.csv "
            "(e.g. columns item_id,diff_true,diff_pred,split)."
        ),
    )
    ap.add_argument(
        "--items_splits",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of allowed values in the items_csv `split` column "
            "(e.g. 'train,test'). If set, only those rows are used. "
            "This is the recommended way to exclude 'zero_success' rows."
        ),
    )
    ap.add_argument("--out_dir", type=str, required=True)

    # Dataset source for loading task text/solutions (must include the requested ids).
    ap.add_argument("--dataset_name", type=str, default="")
    ap.add_argument("--hf_split", type=str, default="", help="HF dataset split to load from (e.g. train/test).")
    ap.add_argument("--dataset_path", type=str, default="")
    ap.add_argument(
        "--agent_results",
        type=str,
        default="",
        help=(
            "Optional subject-responses JSONL (schema: {subject_id, responses{item_id:0/1}}). "
            "If provided, tasks with zero successes are excluded from inference."
        ),
    )

    # Embedding settings (default to whatever was used during training, but allow overriding).
    ap.add_argument("--backbone", type=str, default="")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--torch_dtype", type=str, default="", choices=["", "auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default="")
    ap.add_argument("--embedding_layer", type=int, default=10**9, help="Override embedding layer. Omit to use training value.")
    ap.add_argument("--instruction", type=str, default="", help="Override instruction. Omit to use training value.")
    ap.add_argument("--allow_embedding_mismatch", action="store_true", help="Allow differing embedding settings vs training (not recommended).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    pqd.ensure_dir(str(args.out_dir))

    meta, arrays = load_saved_weights(str(args.weights_dir))
    uses_scaler = bool(meta.get("uses_scaler", False))
    feature_dim = int(meta.get("feature_dim", int(arrays["coef"].size)))

    # Resolve embedding/dataset args from training metadata unless explicitly overridden.
    def pick_str(flag: str, key: str, default: str) -> str:
        v = str(getattr(args, flag))
        # For most string flags we trim whitespace, but for `instruction` whitespace
        # (including a trailing newline) is part of the exact prompt and was saved
        # into training metadata. Stripping it can cause a false "signature mismatch"
        # even when the effective embedding prompt is unchanged.
        if flag == "instruction":
            if v != "":
                return v
        else:
            if v.strip():
                return v.strip()
        mv = meta.get(key, None)
        if mv is None:
            return default
        if key == "instruction":
            return str(mv)
        return str(mv).strip()

    def pick_int(flag: str, key: str, default: int) -> int:
        v = int(getattr(args, flag))
        if flag == "embedding_layer" and v == 10**9:
            mv = meta.get(key, None)
            return int(mv) if mv is not None else default
        if v != 0:
            return int(v)
        mv = meta.get(key, None)
        return int(mv) if mv is not None else default

    dataset_name = pick_str("dataset_name", "dataset_name", "princeton-nlp/SWE-bench_Verified")
    hf_split = pick_str("hf_split", "split", "test")
    dataset_path = pick_str("dataset_path", "dataset_path", "")

    backbone = pick_str("backbone", "backbone", "Qwen/Qwen2.5-Coder-14B")
    max_length = pick_int("max_length", "max_length", 1024)
    batch_size = pick_int("batch_size", "batch_size", 1)
    device_map = pick_str("device_map", "device_map", "auto")
    torch_dtype = pick_str("torch_dtype", "torch_dtype", "auto") or "auto"
    attn_implementation = pick_str("attn_implementation", "attn_implementation", "auto") or "auto"
    embedding_layer = pick_int("embedding_layer", "embedding_layer", -1)
    instruction = pick_str("instruction", "instruction", pqd.DIFFICULTY_INSTRUCTION)

    # Check mismatches with training metadata (unless explicitly allowed).
    if not bool(args.allow_embedding_mismatch):
        for k, cur in [
            ("backbone", backbone),
            ("embedding_layer", int(embedding_layer)),
            ("max_length", int(max_length)),
            ("instruction_signature", pqd.prompt_signature(str(instruction))),
        ]:
            trained = meta.get(k, None)
            if trained is None:
                continue
            if str(trained) != str(cur):
                raise RuntimeError(
                    f"Embedding mismatch for {k}: training={trained!r} vs current={cur!r}. "
                    f"Pass --allow_embedding_mismatch to override (not recommended)."
                )

    # Load requested ids (+ optional labels).
    allowed_splits: Optional[Set[str]] = None
    if str(args.items_splits or "").strip():
        allowed_splits = set([s.strip().lower() for s in str(args.items_splits).split(",") if s.strip()])
        if not allowed_splits:
            allowed_splits = None

    requested_ids, labels = load_item_ids_from_csv(str(args.items_csv), allowed_splits=allowed_splits)
    if not requested_ids:
        raise RuntimeError(f"No item_ids found in items_csv={args.items_csv}")
    print(f"Loaded item_ids: {len(requested_ids)} from {args.items_csv}")

    # Exclude zero-success tasks in the *target* benchmark (if a response matrix is provided).
    zero_success_source = str(args.agent_results or "").strip()
    zero_success_set = set()
    n_zero_success_excluded: Optional[int] = None
    if zero_success_source:
        zero_success_ids = pqd.load_zero_success_task_ids_from_subject_responses_jsonl(zero_success_source)
        zero_success_set = set(zero_success_ids)
        if zero_success_set:
            before = len(requested_ids)
            requested_ids = [tid for tid in requested_ids if tid not in zero_success_set]
            # Keep labels aligned with filtered ids.
            labels = {tid: v for tid, v in labels.items() if tid not in zero_success_set}
            removed = before - len(requested_ids)
            n_zero_success_excluded = int(removed)
            print(f"Excluding zero-success items from inference: {removed}/{before} (source={zero_success_source})")
            if not requested_ids:
                raise RuntimeError("After excluding zero-success items, no item_ids remain for inference.")
        else:
            n_zero_success_excluded = 0

    # Load only the tasks we need, in requested order.
    items, missing = pqd.load_items_by_ids(
        dataset_name=str(dataset_name),
        split=str(hf_split),
        dataset_path=str(dataset_path),
        item_ids=requested_ids,
    )
    if missing:
        print(f"WARNING: {len(missing)}/{len(requested_ids)} ids were not found in dataset (e.g. {missing[:5]})")
    if not items:
        raise RuntimeError("No tasks were found in the dataset for the requested item_ids.")

    # Embed and align X in the order of found items.
    _, emb_by_id, _, emb_dim = pqd.embed_items(
        items=items,
        backbone=str(backbone),
        trust_remote_code=bool(args.trust_remote_code) or bool(meta.get("trust_remote_code", False)),
        max_length=int(max_length),
        batch_size=int(batch_size),
        device_map=str(device_map),
        torch_dtype=str(torch_dtype),
        attn_implementation=str(attn_implementation),
        instruction=str(instruction),
        embedding_layer=int(embedding_layer),
    )
    found_ids = [pqd.normalize_swebench_item_id(it.item_id) for it in items if pqd.normalize_swebench_item_id(it.item_id) in emb_by_id]
    if not found_ids:
        raise RuntimeError("No embeddings were produced for the found items.")
    if int(emb_dim) != int(feature_dim):
        raise RuntimeError(f"Embedding dim mismatch: embeddings have dim={emb_dim} but weights expect feature_dim={feature_dim}")

    X = np.stack([emb_by_id[tid].astype(np.float32, copy=False) for tid in found_ids], axis=0).astype(np.float32, copy=False)
    y_pred = apply_weights(X, arrays=arrays, uses_scaler=uses_scaler)

    # Write predictions CSV.
    pred_path = os.path.join(str(args.out_dir), "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_true", "diff_pred", "split"])
        w.writeheader()
        for tid, yp in zip(found_ids, y_pred.tolist()):
            yt = labels.get(tid, None)
            w.writerow(
                {
                    "item_id": tid,
                    "diff_true": "" if yt is None else float(yt),
                    "diff_pred": float(yp),
                    "split": "inference",
                }
            )

    # Metrics (only if we have labels for at least 2 items).
    y_true_list = [labels.get(tid, None) for tid in found_ids]
    y_true = np.asarray([v for v in y_true_list if v is not None], dtype=np.float64)
    y_pred_labeled = np.asarray([yp for (v, yp) in zip(y_true_list, y_pred.tolist()) if v is not None], dtype=np.float64)
    metrics: dict = {
        "weights_dir": str(args.weights_dir),
        "items_csv": str(args.items_csv),
        "out_dir": str(args.out_dir),
        "n_requested": int(len(requested_ids)),
        "n_found_in_dataset": int(len(items)),
        "n_embedded": int(len(found_ids)),
        "n_labeled_in_csv": int(len(y_true)),
        "zero_success_source": (zero_success_source or None),
        "n_items_zero_success_excluded": n_zero_success_excluded,
        "feature_dim": int(feature_dim),
        "uses_scaler": bool(uses_scaler),
        "dataset_name": str(dataset_name),
        "split": str(hf_split),
        "dataset_path": str(dataset_path),
        "backbone": str(backbone),
        "embedding_layer": int(embedding_layer),
        "max_length": int(max_length),
        "instruction_signature": pqd.prompt_signature(str(instruction)),
        "predictions_csv": str(pred_path),
    }
    if int(y_true.size) >= 2:
        metrics.update(
            {
                "r2": float(pqd.r2_score(y_true, y_pred_labeled)),
                "rmse": float(pqd._rmse(y_true.astype(np.float32), y_pred_labeled.astype(np.float32))),
                "pearson": float(pqd._pearsonr(y_true, y_pred_labeled)),
            }
        )
    pqd.save_json(os.path.join(str(args.out_dir), "metrics.json"), metrics)

    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {os.path.join(str(args.out_dir), 'metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

