#!/usr/bin/env python3
"""
K-fold CV pipeline that wires in IRT training per fold.

For each of K folds over items/tasks:
  - Train an IRT model (1PL by default) using ONLY responses for items in the 4 training folds.
  - Use the IRT-trained item difficulties (b / diff) on the training items as supervision to fit a
    regression model from embeddings -> difficulty.
  - Predict difficulty for the held-out fold items (out-of-fold predictions).
  - Evaluate held-out ROC-AUC on the held-out fold using:
        p(success) = sigmoid(theta_subject - z_item_pred)
    where theta_subject comes from the fold's IRT training (fit on train items only).

We intentionally do NOT compute R^2 on held-out items since they do not have IRT-derived
difficulty parameters from that fold's IRT training (no leakage).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the embedding/data-loading utilities from the sibling script.
from predict_question_difficulty import (  # type: ignore
    DIFFICULTY_INSTRUCTION,
    _dataset_sources_signature,
    _npz_scalar,
    _require,
    _split_multi_arg,
    embed_items,
    ensure_dir,
    iter_swebench_items,
    normalize_swebench_item_id,
    prompt_signature,
    save_json,
    save_regression_weights,
)


def iter_subject_responses_jsonls(paths: Sequence[str]) -> Iterator[Tuple[str, Dict[str, int]]]:
    """
    Yield (subject_id, responses) from one or more JSONL files with schema:
      {"subject_id": "...", "responses": {"task_id": 0/1, ...}}
    """
    for path in list(paths or []):
        p = str(path or "").strip()
        if not p:
            continue
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
                yield sid, out


def load_all_responses(paths: Sequence[str]) -> List[Tuple[str, Dict[str, int]]]:
    """
    Materialize all responses from JSONL(s), normalizing item ids.
    """
    out: List[Tuple[str, Dict[str, int]]] = []
    for sid, resp in iter_subject_responses_jsonls(paths):
        if resp:
            out.append((sid, resp))
    return out


def compute_zero_success_items(all_responses: List[Tuple[str, Dict[str, int]]]) -> List[str]:
    """
    Items with 0 successes across all provided subjects.
    """
    counts: Dict[str, int] = {}
    seen: set[str] = set()
    for _, resp in all_responses:
        for tid, v in resp.items():
            seen.add(tid)
            counts[tid] = counts.get(tid, 0) + int(v)
    return sorted([tid for tid in seen if counts.get(tid, 0) == 0])


def write_filtered_responses_jsonl(
    *,
    all_responses: List[Tuple[str, Dict[str, int]]],
    item_ids: Sequence[str],
    out_path: Path,
) -> Tuple[int, int]:
    """
    Write a py_irt-compatible JSONL with responses restricted to `item_ids`.

    Policy:
    - Include only subjects with at least one observed response among `item_ids`.
    - Write a complete response dict over `item_ids`, filling missing with 0.

    Returns: (n_subjects_written, n_items)
    """
    ensure_dir(str(out_path.parent))
    items = [normalize_swebench_item_id(x) for x in list(item_ids)]
    items = [x for x in items if x]
    item_set = set(items)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sid, resp in all_responses:
            # Keep only items in this fold split.
            present = {k: int(v) for k, v in resp.items() if k in item_set}
            if not present:
                continue
            complete = {tid: int(present.get(tid, 0)) for tid in items}
            f.write(json.dumps({"subject_id": sid, "responses": complete}) + "\n")
            n_written += 1
    return n_written, len(items)


def _compute_binary_auroc(scores: List[float], labels: List[int]) -> float:
    """
    ROC-AUC over binary labels. Returns NaN if undefined (e.g. only one class present).
    Mirrors `auroc.py` behavior.
    """
    if len(scores) == 0:
        return float("nan")
    uniq = set(int(x) for x in labels)
    if len(uniq) < 2:
        return float("nan")
    _require("torchmetrics")
    from torchmetrics import AUROC  # type: ignore

    auroc = AUROC(task="binary")
    s = torch.tensor(scores, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return float(auroc(s, y).item())


def train_irt_1pl(
    *,
    responses_jsonl: Path,
    epochs: int,
    device: str,
    seed: int,
    out_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Train a 1PL IRT model via local `py_irt/` on the provided response JSONL.

    Writes:
      - parameters.json (last epoch)
      - best_parameters.json
      - abilities.csv  (subject_id, theta)
      - items.csv      (item_id, b)  where b is difficulty
    Returns:
      - theta_by_subject
      - diff_by_item
    """
    _require("pyro")
    import pyro  # type: ignore

    from py_irt.config import IrtConfig  # type: ignore
    from py_irt.training import IrtModelTrainer  # type: ignore

    ensure_dir(str(out_dir))

    pyro.clear_param_store()
    cfg = IrtConfig(
        model_type="1pl",
        epochs=int(epochs),
        priors="hierarchical",
        dims=1,
        seed=int(seed),
    )
    trainer = IrtModelTrainer(data_path=responses_jsonl, config=cfg, verbose=False)
    trainer.train(device=str(device))

    # Save params.
    trainer.save(out_dir / "parameters.json")
    best = trainer.best_params or {}
    with (out_dir / "best_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, sort_keys=True)

    # Extract + map ids.
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

    # Write abilities.csv / items.csv (same basic schema expected by auroc.py loaders).
    with (out_dir / "abilities.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "theta"])
        w.writeheader()
        for sid, theta in sorted(theta_by_subject.items()):
            w.writerow({"subject_id": sid, "theta": float(theta)})

    with (out_dir / "items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "b"])
        w.writeheader()
        for tid, b in sorted(diff_by_item.items()):
            w.writerow({"item_id": tid, "b": float(b)})

    return theta_by_subject, diff_by_item


def _read_indexed_csv_numeric(path: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Read a CSV where the first column is an index.

    Example (pandas to_csv default with index):
      ,b,b_std
      astropy__astropy-12907,1.23,0.45

    Returns:
      - cols: column names excluding the index column
      - rows: mapping index -> {col -> float}
    """
    rows: Dict[str, Dict[str, float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None or len(header) < 2:
            raise ValueError(f"CSV {path!r} appears empty or has no columns.")
        cols = [str(x) for x in header[1:]]
        for row in r:
            if not row:
                continue
            idx = str(row[0] or "").strip()
            if not idx:
                continue
            d: Dict[str, float] = {}
            for j, col in enumerate(cols, start=1):
                if j >= len(row):
                    continue
                s = str(row[j]).strip()
                if not s:
                    continue
                try:
                    d[str(col)] = float(s)
                except Exception:
                    continue
            rows[idx] = d
    return cols, rows


def load_precomputed_item_difficulties(*, items_csv: str, diff_col: str = "") -> Dict[str, float]:
    cols, rows = _read_indexed_csv_numeric(items_csv)
    diff_col = str(diff_col or "").strip()
    if not diff_col:
        # Prefer 1D b; otherwise use 2D b_sum.
        if "b" in cols:
            diff_col = "b"
        elif "b_sum" in cols:
            diff_col = "b_sum"
        else:
            raise ValueError(f"Could not infer difficulty column from {items_csv!r}. Columns={cols}")

    out: Dict[str, float] = {}
    for raw_item_id, row in rows.items():
        tid = normalize_swebench_item_id(str(raw_item_id))
        if not tid:
            continue
        if diff_col not in row:
            continue
        out[tid] = float(row[diff_col])
    return out


def load_precomputed_thetas(*, thetas_csv: str) -> Dict[str, float]:
    cols, rows = _read_indexed_csv_numeric(thetas_csv)
    # Prefer 1D theta; for 2D use theta_sum (since the model uses summed dims in the logit).
    if "theta" in cols:
        prefer = "theta"
    elif "theta_sum" in cols:
        prefer = "theta_sum"
    else:
        raise ValueError(f"Could not infer theta column from {thetas_csv!r}. Columns={cols}")

    out: Dict[str, float] = {}
    for raw_id, row in rows.items():
        k = str(raw_id or "").strip()
        if not k:
            continue
        if prefer not in row:
            continue
        out[k] = float(row[prefer])
    return out


def load_agent_model_scaffold_map(agent_map_csv: str) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    with open(agent_map_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            agent = str((row or {}).get("agent", "") or "").strip()
            model = str((row or {}).get("model", "") or "").strip()
            scaffold = str((row or {}).get("scaffold", "") or "").strip()
            if not agent or not model or not scaffold:
                continue
            out[agent] = (model, scaffold)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_name",
        type=str,
        nargs="+",
        default=["princeton-nlp/SWE-bench_Verified"],
        help="One or more HF dataset repos to load (space-separated or comma-separated).",
    )
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional local JSON/JSONL dataset path(s), comma-separated.",
    )
    ap.add_argument("--n_inputs", type=int, default=500, help="Number of dataset items to embed (0 means all).")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--backbone", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default="auto")
    ap.add_argument("--embedding_layer", type=int, default=-1)
    ap.add_argument("--instruction", type=str, default=DIFFICULTY_INSTRUCTION, help="Instruction appended when embedding (optional).")

    ap.add_argument("--out_dir", type=str, default="./out/swebench_verified")
    ap.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument(
        "--agent_results",
        type=str,
        nargs="*",
        default=["./out/chris_irt/swebench_verified_20251115_full.jsonl"],
        help="JSONL(s) with per-subject responses: {'subject_id': ..., 'responses': {task_id: 0/1, ...}}",
    )
    ap.add_argument("--exclude_zero_success", action="store_true", help="Exclude items with 0 successes from CV/IRT.")

    ap.add_argument("--cv_folds", type=int, default=5)

    ap.add_argument("--irt_epochs", type=int, default=5000)
    ap.add_argument("--irt_device", type=str, default="cuda")

    # If set, we will NOT train per-fold IRT; instead we will load item difficulties + thetas
    # from a precomputed IRT run (e.g., swebench_irt/train_model_scaffold_shared.py outputs).
    ap.add_argument(
        "--irt_items_csv",
        type=str,
        default="",
        help="Precomputed IRT items CSV (e.g., .../2d_1pl/items_verified.csv or .../1d_1pl/items_verified.csv).",
    )
    ap.add_argument(
        "--irt_items_diff_col",
        type=str,
        default="",
        help="Column to use as item difficulty from --irt_items_csv (default auto: b, else b_sum).",
    )
    ap.add_argument(
        "--irt_agent_map_csv",
        type=str,
        default="",
        help="Agent->(model,scaffold) CSV (e.g., .../agent_model_scaffold.csv). Required with --irt_items_csv.",
    )
    ap.add_argument(
        "--irt_model_thetas_csv",
        type=str,
        default="",
        help="Model abilities CSV (e.g., .../2d_1pl/model_abilities.csv). Required with --irt_items_csv.",
    )
    ap.add_argument(
        "--irt_scaffold_thetas_csv",
        type=str,
        default="",
        help="Scaffold abilities CSV (e.g., .../2d_1pl/scaffold_abilities.csv). Required with --irt_items_csv.",
    )

    ap.add_argument("--regressor", type=str, default="ridge_cv", choices=["linear", "ridge", "ridge_cv"])
    ap.add_argument("--ridge_alpha", type=float, default=10000.0)
    ap.add_argument("--ridge_alphas", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000")
    ap.add_argument("--ridge_cv_folds", type=int, default=5, help="Inner folds for RidgeCV when --regressor=ridge_cv.")

    args = ap.parse_args(argv)
    ensure_dir(str(args.out_dir))

    dataset_names = _split_multi_arg(args.dataset_name)
    dataset_paths = _split_multi_arg(args.dataset_path)
    dataset_sources_str = _dataset_sources_signature(dataset_names=dataset_names, dataset_paths=dataset_paths) or (
        "princeton-nlp/SWE-bench_Verified"
    )

    # Load responses (required for this script).
    agent_results_paths = _split_multi_arg(args.agent_results)
    if not agent_results_paths:
        raise ValueError("This script requires --agent_results JSONL(s) to train IRT per fold.")

    all_responses = load_all_responses(agent_results_paths)
    if not all_responses:
        raise RuntimeError("Loaded 0 subject responses from --agent_results.")

    # Build response task universe.
    response_items: set[str] = set()
    for _, resp in all_responses:
        for tid in resp.keys():
            if tid:
                response_items.add(normalize_swebench_item_id(tid))
    response_items = set([x for x in response_items if x])

    zero_success_ids: List[str] = compute_zero_success_items(all_responses)
    zero_success_set = set(zero_success_ids)
    if args.exclude_zero_success:
        print(f"Excluding zero-success items from CV/IRT: {len(zero_success_ids)}")

    # Embeddings cache path derived from key settings.
    safe_backbone = str(args.backbone).replace("/", "__")
    ds_flag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(dataset_sources_str))[:64]
    split_flag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(args.split))[:32]
    instr_sig = prompt_signature(str(args.instruction or ""))
    layer_flag = "" if int(args.embedding_layer) == -1 else f"__layer{int(args.embedding_layer)}"
    emb_cache = str(args.embeddings_cache or "").strip()
    if not emb_cache:
        emb_cache = os.path.join(
            str(args.out_dir),
            f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__qs-sol-instr__{instr_sig}__{ds_flag}__{split_flag}__n{int(args.n_inputs)}__maxlen{int(args.max_length)}__seed{int(args.seed)}.npz",
        )

    # Load or compute embeddings.
    if os.path.exists(emb_cache) and not args.overwrite:
        data = np.load(emb_cache, allow_pickle=True)
        task_ids = [normalize_swebench_item_id(str(x)) for x in list(data["task_ids"].tolist())]
        task_ids = [x for x in task_ids if x]
        X = data["X"].astype(np.float32)
        cached_layer = int(_npz_scalar(data.get("embedding_layer", None), -1)) if "embedding_layer" in data else -1
        if int(args.embedding_layer) != int(cached_layer):
            raise RuntimeError(
                f"Embeddings cache was created with embedding_layer={cached_layer}, but you requested "
                f"--embedding_layer={int(args.embedding_layer)}. Use --overwrite, or pick a different cache file."
            )
        print(f"Loaded embeddings cache: {emb_cache} (n={len(task_ids)}, dim={X.shape[1]}, embedding_layer={cached_layer})")
    else:
        items = list(
            iter_swebench_items(
                dataset_names=list(dataset_names),
                split=str(args.split),
                dataset_paths=list(dataset_paths),
                n_inputs=int(args.n_inputs),
                seed=int(args.seed),
                shuffle=bool(args.shuffle),
            )
        )
        print(f"Loaded dataset items: {len(items)} (sources={dataset_sources_str})")

        ids_sorted, emb_by_id, _, emb_dim = embed_items(
            items=items,
            backbone=str(args.backbone),
            trust_remote_code=bool(args.trust_remote_code),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            device_map=str(args.device_map),
            torch_dtype=str(args.torch_dtype),
            attn_implementation=str(args.attn_implementation),
            instruction=str(args.instruction or ""),
            embedding_layer=int(args.embedding_layer),
        )
        if not ids_sorted:
            raise RuntimeError("No embeddings were produced (empty ids set).")
        task_ids = [normalize_swebench_item_id(x) for x in ids_sorted]
        X = np.stack([emb_by_id[r] for r in ids_sorted], axis=0).astype(np.float32)

        np.savez_compressed(
            emb_cache,
            task_ids=np.array(task_ids, dtype=object),
            X=X,
            dataset_name=np.array([str(dataset_sources_str)], dtype=object),
            split=np.array([str(args.split)], dtype=object),
            dataset_path=np.array([";".join([str(x) for x in dataset_paths])], dtype=object),
            n_inputs=np.array([int(len(task_ids))], dtype=np.int64),
            instruction=np.array([str(args.instruction or "")], dtype=object),
            instruction_signature=np.array([str(instr_sig)], dtype=object),
            backbone=np.array([str(args.backbone)], dtype=object),
            max_length=np.array([int(args.max_length)], dtype=np.int64),
            embedding_dim=np.array([int(emb_dim)], dtype=np.int64),
            embedding_layer=np.array([int(args.embedding_layer)], dtype=np.int64),
        )
        print(f"Wrote embeddings cache: {emb_cache} (n={len(task_ids)}, dim={X.shape[1]})")

    id_to_row = {tid: i for i, tid in enumerate(task_ids)}

    # Eligible items must have embeddings and appear in responses; optionally exclude zero-success.
    eligible = [tid for tid in task_ids if tid in response_items]
    if args.exclude_zero_success:
        eligible = [tid for tid in eligible if tid not in zero_success_set]
    if not eligible:
        raise RuntimeError("No eligible items found: need overlap between embeddings task_ids and responses task_ids.")

    # Outer CV over items.
    outer_k = int(args.cv_folds)
    if outer_k < 2:
        raise ValueError("--cv_folds must be >= 2")
    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=int(args.seed))

    # Regression model builder.
    regressor_name = str(args.regressor)
    alphas: np.ndarray = np.array([], dtype=np.float64)

    def _make_regressor():
        nonlocal alphas
        if regressor_name == "linear":
            return LinearRegression()
        if regressor_name == "ridge":
            alpha = float(args.ridge_alpha)
            if not (alpha > 0):
                raise ValueError("--ridge_alpha must be > 0")
            return Pipeline(
                steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("ridge", Ridge(alpha=alpha))]
            )
        if regressor_name == "ridge_cv":
            try:
                alphas = np.array(
                    [float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()], dtype=np.float64
                )
            except Exception as e:
                raise ValueError(f"Failed to parse --ridge_alphas={args.ridge_alphas!r}: {e}") from e
            if alphas.size == 0:
                raise ValueError("Expected at least one alpha in --ridge_alphas")
            inner_cv = KFold(n_splits=int(args.ridge_cv_folds), shuffle=True, random_state=int(args.seed))
            return Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("ridge", RidgeCV(alphas=alphas, cv=inner_cv)),
                ]
            )
        raise AssertionError(f"Unhandled regressor: {regressor_name}")

    # OOF predictions for eligible items.
    yhat_oof = {tid: float("nan") for tid in eligible}
    fold_of_item: Dict[str, int] = {}
    fold_test_auc: List[float] = []

    best_auc = -float("inf")
    best_fold = -1
    best_model = None
    best_fold_auc_pairs = 0
    best_fold_theta_by_subject: Dict[str, float] = {}
    using_precomputed_irt = bool(str(args.irt_items_csv or "").strip())

    precomputed_diff_by_item: Dict[str, float] = {}
    precomputed_theta_by_subject: Dict[str, float] = {}
    if using_precomputed_irt:
        if not str(args.irt_agent_map_csv or "").strip():
            raise ValueError("--irt_agent_map_csv is required when --irt_items_csv is set")
        if not str(args.irt_model_thetas_csv or "").strip():
            raise ValueError("--irt_model_thetas_csv is required when --irt_items_csv is set")
        if not str(args.irt_scaffold_thetas_csv or "").strip():
            raise ValueError("--irt_scaffold_thetas_csv is required when --irt_items_csv is set")

        precomputed_diff_by_item = load_precomputed_item_difficulties(
            items_csv=str(args.irt_items_csv), diff_col=str(args.irt_items_diff_col or "")
        )
        if not precomputed_diff_by_item:
            raise RuntimeError(f"Loaded 0 precomputed item difficulties from --irt_items_csv={args.irt_items_csv!r}")

        agent_map = load_agent_model_scaffold_map(str(args.irt_agent_map_csv))
        theta_by_model = load_precomputed_thetas(thetas_csv=str(args.irt_model_thetas_csv))
        theta_by_scaffold = load_precomputed_thetas(thetas_csv=str(args.irt_scaffold_thetas_csv))
        for sid, _ in all_responses:
            pair = agent_map.get(str(sid), None)
            if pair is None:
                continue
            model, scaffold = pair
            tm = theta_by_model.get(model, None)
            ts = theta_by_scaffold.get(scaffold, None)
            if tm is None or ts is None:
                continue
            precomputed_theta_by_subject[str(sid)] = float(tm) + float(ts)
        if not precomputed_theta_by_subject:
            raise RuntimeError(
                "Loaded 0 precomputed subject thetas from the provided IRT CSVs/maps. "
                "Check that --agent_results subject_ids match --irt_agent_map_csv agent names."
            )

    eligible_idx = np.array([id_to_row[tid] for tid in eligible], dtype=np.int64)
    X_elig = X[eligible_idx]

    folds_dir = Path(str(args.out_dir)) / "folds"
    ensure_dir(str(folds_dir))

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_elig), start=1):
        train_items = [eligible[i] for i in tr_idx.tolist()]
        test_items = [eligible[i] for i in te_idx.tolist()]

        fold_dir = folds_dir / f"fold_{fold:02d}"
        irt_dir = fold_dir / "irt"
        reg_dir = fold_dir / "regression"
        ensure_dir(str(irt_dir))
        ensure_dir(str(reg_dir))

        theta_by_subject: Dict[str, float] = {}
        diff_by_item: Dict[str, float] = {}
        if using_precomputed_irt:
            # Use the full precomputed dict; we still subset to train_items below for regression supervision.
            diff_by_item = precomputed_diff_by_item
            theta_by_subject = precomputed_theta_by_subject
        else:
            # 1) Train IRT on training fold items only.
            train_jsonl = irt_dir / "train_responses.jsonl"
            n_subj, n_items = write_filtered_responses_jsonl(
                all_responses=all_responses, item_ids=train_items, out_path=train_jsonl
            )
            if n_subj == 0 or n_items == 0:
                raise RuntimeError(f"Fold {fold}: empty filtered response matrix (subjects={n_subj}, items={n_items}).")

            theta_by_subject, diff_by_item = train_irt_1pl(
                responses_jsonl=train_jsonl,
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed) + int(fold),
                out_dir=irt_dir,
            )
            if not diff_by_item:
                raise RuntimeError(f"Fold {fold}: IRT training produced 0 item difficulties.")

        # 2) Train regression on (embedding -> IRT difficulty) for training items.
        tr_rows = [id_to_row[tid] for tid in train_items if tid in diff_by_item]
        tr_ids = [tid for tid in train_items if tid in diff_by_item]
        if not tr_rows:
            raise RuntimeError(f"Fold {fold}: no train items had IRT difficulty parameters.")
        X_tr = X[np.array(tr_rows, dtype=np.int64)]
        y_tr = np.array([diff_by_item[tid] for tid in tr_ids], dtype=np.float32)

        m = _make_regressor()
        m.fit(X_tr, y_tr)

        # 3) Predict difficulties for held-out fold items.
        te_rows = [id_to_row[tid] for tid in test_items]
        X_te = X[np.array(te_rows, dtype=np.int64)]
        z_te = m.predict(X_te).astype(np.float64)
        for tid, z in zip(test_items, z_te.tolist()):
            yhat_oof[tid] = float(z)
            fold_of_item[tid] = int(fold)

        # 4) Evaluate held-out ROC-AUC on test fold items using theta(train) and z_pred(test).
        z_by_item = {tid: float(z) for tid, z in zip(test_items, z_te.tolist())}
        scores: List[float] = []
        labels: List[int] = []
        for sid, resp in all_responses:
            theta = theta_by_subject.get(sid, None)
            if theta is None:
                continue
            for item_id, y_obs in resp.items():
                z = z_by_item.get(item_id, None)
                if z is None:
                    continue
                scores.append(1.0 / (1.0 + math.exp(-(float(theta) - float(z)))))
                labels.append(int(y_obs))

        auc = float(_compute_binary_auroc(scores, labels))
        fold_test_auc.append(float(auc))
        save_json(
            str(fold_dir / "fold_metrics.json"),
            {
                "fold": int(fold),
                "n_train_items": int(len(train_items)),
                "n_test_items": int(len(test_items)),
                "n_irt_subjects": int(len(theta_by_subject)),
                "n_irt_items": int(len(diff_by_item)),
                "test_auc": float(auc),
                "test_pairs": int(len(labels)),
                "using_precomputed_irt": bool(using_precomputed_irt),
            },
        )

        if auc == auc and auc > best_auc:
            best_auc = float(auc)
            best_fold = int(fold)
            best_model = m
            best_fold_auc_pairs = int(len(labels))
            best_fold_theta_by_subject = dict(theta_by_subject)

    # Sanity: all eligible items should have OOF predictions.
    missing = [tid for tid, z in yhat_oof.items() if not (z == z)]
    if missing:
        raise RuntimeError(f"Missing OOF predictions for {len(missing)} eligible items (e.g. {missing[:3]}).")
    if best_model is None or best_fold < 1:
        raise RuntimeError("Failed to select a best fold model by ROC-AUC (all folds NaN?).")

    auc_mean = float(np.mean(np.array(fold_test_auc, dtype=np.float64))) if fold_test_auc else float("nan")
    print(f"{outer_k}-fold CV test ROC-AUC (mean over folds): {auc_mean}")
    print(f"{outer_k}-fold CV test ROC-AUC per fold: " + ", ".join([str(x) for x in fold_test_auc]))
    print(f"Best fold by held-out ROC-AUC: fold={best_fold} auc={best_auc} (pairs={best_fold_auc_pairs})")

    # Save regression weights for the best fold model.
    weights_meta = {
        "script": os.path.abspath(__file__),
        "seed": int(args.seed),
        "cv_folds": int(outer_k),
        "best_fold": int(best_fold),
        "best_fold_auc": float(best_auc),
        "regressor": regressor_name,
        "ridge_alpha": (float(best_model.named_steps["ridge"].alpha_) if regressor_name in ("ridge", "ridge_cv") else None),
        "ridge_alphas_searched": [float(x) for x in np.asarray(alphas).tolist()],
        "dataset_sources": str(dataset_sources_str),
        "dataset_names": list(dataset_names),
        "dataset_paths": list(dataset_paths),
        "split": str(args.split),
        "backbone": str(args.backbone),
        "embedding_layer": int(args.embedding_layer),
        "max_length": int(args.max_length),
        "embeddings_cache": str(emb_cache),
        "agent_results": list(agent_results_paths),
        "exclude_zero_success": bool(args.exclude_zero_success),
        "zero_success_count": int(len(zero_success_ids)),
        "using_precomputed_irt": bool(using_precomputed_irt),
        "irt_items_csv": str(args.irt_items_csv),
        "irt_items_diff_col": str(args.irt_items_diff_col),
        "irt_agent_map_csv": str(args.irt_agent_map_csv),
        "irt_model_thetas_csv": str(args.irt_model_thetas_csv),
        "irt_scaffold_thetas_csv": str(args.irt_scaffold_thetas_csv),
        "irt_epochs": (None if using_precomputed_irt else int(args.irt_epochs)),
        "irt_device": (None if using_precomputed_irt else str(args.irt_device)),
    }
    weights_json, weights_npz = save_regression_weights(
        out_dir=str(args.out_dir),
        model=best_model,
        regressor_name=str(regressor_name),
        feature_dim=int(X.shape[1]),
        metadata=weights_meta,
    )

    # Write OOF predictions for eligible items.
    pred_path = os.path.join(str(args.out_dir), "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "diff_pred", "split", "fold"])
        w.writeheader()
        for tid in eligible:
            w.writerow(
                {
                    "item_id": tid,
                    "diff_pred": float(yhat_oof[tid]),
                    "split": "cv_val",
                    "fold": int(fold_of_item.get(tid, -1)),
                }
            )

    metrics = {
        "cv_n_splits": int(outer_k),
        "cv_test_auc_folds": [float(x) for x in fold_test_auc],
        "cv_test_auc_mean": float(auc_mean),
        "cv_best_auc_fold": int(best_fold),
        "cv_best_auc": float(best_auc),
        "n_items_eligible": int(len(eligible)),
        "exclude_zero_success": bool(args.exclude_zero_success),
        "n_items_zero_success": int(len(zero_success_ids)),
        "regression_weights_json": str(weights_json),
        "regression_weights_npz": str(weights_npz),
        "predictions_csv": str(pred_path),
    }
    save_json(os.path.join(str(args.out_dir), "metrics.json"), metrics)

    # Optional convenience: also print predictions for excluded zero-success items using best fold regression model.
    if args.exclude_zero_success and zero_success_set:
        zero_embedded = [tid for tid in task_ids if tid in zero_success_set and tid in id_to_row]
        if zero_embedded:
            X_zero = np.stack([X[id_to_row[tid]] for tid in zero_embedded], axis=0).astype(np.float32)
            z0 = best_model.predict(X_zero).astype(np.float64).tolist()
            pairs = sorted(zip(zero_embedded, z0), key=lambda kv: float(kv[1]), reverse=True)
            print(f"\n=== ZERO_SUCCESS_PREDICTIONS_SORTED (task_id, diff_pred) [model=best_fold_by_auc fold={best_fold} auc={best_auc}] ===")
            for tid, score in pairs:
                print(f"{tid}\t{float(score):.6f}")

    print(f"Wrote metrics: {os.path.join(str(args.out_dir), 'metrics.json')}")
    print(f"Wrote predictions: {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

