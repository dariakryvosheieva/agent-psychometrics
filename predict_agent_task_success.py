#!/usr/bin/env python3
"""
Train a PyTorch MLP to predict 0/1 success for an (agent, task) observation using:
  - learned embeddings for model + scaffold IDs (nn.Embedding)
  - task features: embedding of (statement + solution + instruction) (cached .npz)

Input response JSONL schema (same as other scripts here):
  {"subject_id": "<agent_or_model>", "responses": {"<task_id>": 0|1, ...}}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Please install requirements (see "
            f"`fulcrum/fellowship/trajectory_embedding_requirements.txt`). Original error: {e}"
        ) from e


@dataclass(frozen=True)
class Obs:
    benchmark: str
    subject_id: str
    model: str
    scaffold: str
    task_id: str
    task_key: str  # "<bench>::<task_id>"
    y: int


def _canon_benchmark(name: str) -> str:
    # Keep consistent with the "classic" script and multi-benchmark code.
    s = str(name or "").strip().lower().replace("-", "_")
    if s == "terminalbench":
        s = "terminal_bench"
    if s not in {"verified", "pro", "terminal_bench", "gso"}:
        raise ValueError(f"Unknown benchmark: {name!r}. Allowed: verified, pro, terminal-bench, gso.")
    return s


def _parse_benchmarks(spec: str) -> List[str]:
    raw = str(spec or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    out: List[str] = []
    seen = set()
    for p in parts:
        if not p:
            continue
        k = _canon_benchmark(p)
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _import_swebench_irt_module(module_name: str):
    """
    Import a module from `fulcrum/fellowship/swebench_irt/` while preserving sibling imports.
    Mirrors `predict_question_difficulty_multi_benchmark._import_swebench_irt_module`.
    """
    here = Path(__file__).resolve().parent
    swe_irt_dir = str(here / "swebench_irt")
    if swe_irt_dir not in sys.path:
        sys.path.insert(0, swe_irt_dir)
    return __import__(str(module_name))


def _split_subject_to_model_scaffold(*, benchmark: str, subject_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (model, scaffold) for a subject_id under the benchmark's naming convention.

    - verified, terminal_bench: split subject_id as a full agent name
    - pro: subject_id is model-only, scaffold is assumed "SWE-agent 1.0"
    - gso: subject_id is model-only, scaffold is assumed "OpenHands"
    """
    b = _canon_benchmark(benchmark)
    subj = str(subject_id or "").strip()
    if not subj:
        return None, None

    filt = _import_swebench_irt_module("filter_subjects_by_scaffold_count")
    split_mod = _import_swebench_irt_module("split_agents_model_scaffold")

    if b == "pro":
        m = filt._model_for_subject(subj, treat_as_pro=True)  # type: ignore[attr-defined]
        sc = filt._scaffold_for_subject(subj, treat_as_pro=True)  # type: ignore[attr-defined]
        return (str(m) if m is not None else None), (str(sc) if sc is not None else None)

    if b == "gso":
        # GSO exports are model-only strings; assume OpenHands and canonicalize.
        assume = split_mod.assumed_scaffold_for_benchmark("gso")  # type: ignore[attr-defined]
        assume_s = str(assume or "OpenHands").strip()
        try:
            m = str(split_mod._canonical_model(subj))  # type: ignore[attr-defined]
        except Exception:
            m = subj
        try:
            sc = str(split_mod._canonical_scaffold(assume_s))  # type: ignore[attr-defined]
        except Exception:
            sc = assume_s
        return (m or None), (sc or None)

    # verified / terminal_bench
    m = filt._model_for_subject(subj, treat_as_pro=False)  # type: ignore[attr-defined]
    sc = filt._scaffold_for_subject(subj, treat_as_pro=False)  # type: ignore[attr-defined]
    return (str(m) if m is not None else None), (str(sc) if sc is not None else None)


def _iter_obs_from_responses(
    *,
    benchmark: str,
    agent_results_jsonl: str,
    normalize_item_ids: bool,
) -> Iterator[Tuple[str, str, int]]:
    """
    Yield (subject_id, task_id, y) for each observed response entry.
    """
    b = _canon_benchmark(benchmark)
    p = str(agent_results_jsonl or "").strip()
    if not p:
        return
    if not os.path.exists(p):
        raise FileNotFoundError(f"Agent results JSONL not found: {p}")

    norm_fn = None
    if bool(normalize_item_ids):
        import predict_question_difficulty as base

        norm_fn = base.normalize_swebench_item_id

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
            for raw_id, v in resp.items():
                tid = str(raw_id or "").strip()
                if not tid:
                    continue
                if norm_fn is not None and b in {"verified", "pro", "gso"}:
                    tid = str(norm_fn(tid) or "").strip()
                    if not tid:
                        continue
                try:
                    yy = int(v)
                except Exception:
                    yy = 1 if v else 0
                yield sid, tid, int(1 if yy != 0 else 0)


def _stable_group_split(groups: Sequence[str], *, test_fraction: float, seed: int) -> Tuple[set, set]:
    """
    Deterministic split of unique group ids into train/test by hashing.
    Returns (train_group_set, test_group_set).
    """
    if not (0.0 < float(test_fraction) < 1.0):
        raise ValueError("--test_fraction must be between 0 and 1")
    uniq = sorted(set([str(g) for g in groups if str(g).strip()]))
    if len(uniq) == 0:
        return set(), set()

    n_test = int(round(len(uniq) * float(test_fraction)))
    xs: List[Tuple[float, str]] = []
    for g in uniq:
        h = hashlib.md5((g + f"::{int(seed)}").encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / float(16**8)
        xs.append((x, g))
    xs.sort()
    test = set([g for _, g in xs[:n_test]])
    train = set([g for g in uniq if g not in test])
    return train, test


def _stable_group_kfold(groups: Sequence[str], *, n_splits: int, seed: int) -> List[Tuple[set, set]]:
    """
    Deterministic group K-fold split using hashing (no sklearn dependency here).

    Returns a list of (train_group_set, test_group_set) for each fold.
    """
    k = int(n_splits)
    if k < 2:
        raise ValueError("--cv_folds must be >= 2 for cross-validation")
    uniq = sorted(set([str(g) for g in groups if str(g).strip()]))
    if len(uniq) < k:
        raise RuntimeError(f"Not enough unique groups ({len(uniq)}) for {k}-fold CV. Try a smaller --cv_folds or different --split_by.")

    # Stable pseudo-random ordering of groups based on hash(seed, group).
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


def _sigmoid(x: float) -> float:
    # numerically-stable-ish sigmoid for moderate magnitudes
    try:
        import math

        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    except Exception:
        return 0.5


def _load_or_build_task_embeddings(
    *,
    out_dir: str,
    embeddings_cache: str,
    overwrite: bool,
    backbone: str,
    trust_remote_code: bool,
    max_length: int,
    batch_size: int,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    instruction: str,
    embedding_layer: int,
    # benchmark-specific task sources:
    verified_dataset_name: str,
    verified_split: str,
    pro_dataset_name: str,
    pro_split: str,
    terminal_bench_tasks_jsonl: str,
    gso_dataset_name: str,
    gso_split: str,
    # ids needed:
    task_ids_by_benchmark: Dict[str, List[str]],
) -> Tuple["Dict[str, Any]", int, str]:
    """
    Returns: (embedding_by_task_key, embedding_dim, cache_path)
    where task_key is always "<bench>::<task_id>".
    """
    os.makedirs(str(out_dir), exist_ok=True)
    cache_path = str(embeddings_cache or "").strip()
    if not cache_path:
        raise ValueError("embeddings_cache path is empty")

    # Lazy imports (heavy deps).
    _require("numpy")
    import numpy as np

    # These scripts bring in torch/transformers/datasets; we defer until we're sure we need them.
    import predict_question_difficulty as base
    import predict_question_difficulty_multi_benchmark as mb

    if os.path.exists(cache_path) and not bool(overwrite):
        _p(f"[progress] Loading task embeddings cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)  # type: ignore[no-untyped-call]
        task_keys = [str(x) for x in list(data["task_ids"].tolist())]
        X = data["X"].astype(np.float32, copy=False)
        emb_dim = int(X.shape[1]) if X.ndim == 2 else 0
        by_key = {k: X[i] for i, k in enumerate(task_keys)}
        _p(f"[progress] Loaded embeddings: n={len(task_keys)} dim={emb_dim}")
        return by_key, int(emb_dim), cache_path

    # Build items list (with globally-unique prefixed ids).
    items: List[base.ItemRecord] = []
    _p("[progress] Building task embeddings (cache miss or --overwrite).")
    for bench, raw_ids in task_ids_by_benchmark.items():
        b = _canon_benchmark(bench)
        want = [str(x) for x in raw_ids if str(x).strip()]
        if not want:
            continue
        _p(f"[progress]  - {b}: {len(want)} task ids requested")

        loaded: List[base.ItemRecord] = []
        missing: List[str] = []
        if b in {"verified", "pro"}:
            ds = verified_dataset_name if b == "verified" else pro_dataset_name
            split = verified_split if b == "verified" else pro_split
            loaded, missing = mb.load_swebench_items_by_ids(dataset_name=str(ds), split=str(split), item_ids=want, normalize_item_ids=True)
        elif b == "terminal_bench":
            loaded, missing = mb.load_terminal_bench_items_by_ids(tasks_jsonl=str(terminal_bench_tasks_jsonl), item_ids=want)
        elif b == "gso":
            loaded, missing = mb.load_ood_items_by_ids(
                dataset_name=str(gso_dataset_name),
                split=str(gso_split),
                item_ids=want,
                normalize_item_ids=True,
                wrap_with_gso_prompt=True,
            )
        else:
            raise AssertionError(f"Unhandled benchmark: {b}")

        if missing:
            # Missing tasks can happen when response JSONLs contain ids outside the task source.
            # We'll simply skip embedding them; downstream rows will be dropped if they need them.
            _p(f"[progress]  - {b}: missing {len(missing)}/{len(want)} tasks in task source (will be dropped if needed)")
        _p(f"[progress]  - {b}: loaded {len(loaded)} task records")

        for it in loaded:
            key = f"{b}::{str(it.item_id)}"
            items.append(base.ItemRecord(item_id=key, question_statement=str(it.question_statement), solution=str(it.solution)))

    if not items:
        raise RuntimeError("No tasks were loaded for embedding (empty items list). Check task sources and response JSONLs.")

    _p(f"[progress] Embedding {len(items)} tasks with backbone={backbone!r} max_length={int(max_length)} batch_size={int(batch_size)}")
    ids_sorted, emb_by_id, counts_by_id, emb_dim = base.embed_items(
        items=items,
        backbone=str(backbone),
        trust_remote_code=bool(trust_remote_code),
        max_length=int(max_length),
        batch_size=int(batch_size),
        device_map=str(device_map),
        torch_dtype=str(torch_dtype),
        attn_implementation=str(attn_implementation),
        instruction=str(instruction),
        embedding_layer=int(embedding_layer),
    )
    if not ids_sorted:
        raise RuntimeError("No embeddings were produced (empty ids set).")

    _p(f"[progress] Finished embeddings: n={len(ids_sorted)} dim={int(emb_dim)}. Saving cache: {cache_path}")
    X = np.stack([emb_by_id[k] for k in ids_sorted], axis=0).astype(np.float32, copy=False)
    counts_arr = np.array([int(counts_by_id.get(k, 0)) for k in ids_sorted], dtype=np.int64)

    np.savez_compressed(
        cache_path,
        task_ids=np.array(ids_sorted, dtype=object),
        X=X,
        counts_kind=np.array(["text_len_chars"], dtype=object),
        counts=counts_arr,
        instruction=np.array([str(instruction)], dtype=object),
        instruction_signature=np.array([str(base.prompt_signature(str(instruction)))], dtype=object),
        backbone=np.array([str(backbone)], dtype=object),
        max_length=np.array([int(max_length)], dtype=np.int64),
        embedding_dim=np.array([int(emb_dim)], dtype=np.int64),
        embedding_layer=np.array([int(embedding_layer)], dtype=np.int64),
    )
    by_key = {k: np.asarray(emb_by_id[k], dtype=np.float32) for k in ids_sorted}
    return by_key, int(emb_dim), cache_path


def _set_seeds(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def _p(msg: str) -> None:
    # Slurm stdout can be buffered; force flush for progress visibility.
    print(str(msg), flush=True)


def _set_torch_determinism(*, deterministic: bool) -> None:
    """
    Determinism policy:
      - IRT: deterministic=False (requested)
      - everything else: deterministic=True

    NOTE: When deterministic=True on CUDA, PyTorch may require CUBLAS_WORKSPACE_CONFIG to be set;
    we set it early in main (before importing torch) when running on CUDA.
    """
    try:
        import torch

        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
        # This is what triggers the CuBLAS reproducibility requirement.
        # We keep it strict here; main ensures the env var is set on CUDA.
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

    # -----------------------------
    # Data selection
    # -----------------------------
    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/agent_task_success_embed_ms")
    p.add_argument("--train_benchmarks", type=str, default="verified,pro,terminal_bench", help="Comma-separated subset of {verified, pro, terminal-bench, gso}.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # Response matrices (per benchmark)
    p.add_argument("--verified_agent_results_jsonl", type=str, default="")
    p.add_argument("--pro_agent_results_jsonl", type=str, default="")
    p.add_argument("--terminal_bench_agent_results_jsonl", type=str, default="")
    p.add_argument("--gso_agent_results_jsonl", type=str, default="")

    # Task sources
    p.add_argument("--verified_dataset_name", type=str, default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--verified_split", type=str, default="test")
    p.add_argument("--pro_dataset_name", type=str, default="ScaleAI/SWE-bench_Pro")
    p.add_argument("--pro_split", type=str, default="test")
    p.add_argument(
        "--terminal_bench_tasks_jsonl",
        type=str,
        default="out/chris_irt/terminal_bench_tasks.jsonl",
        help="Terminal-Bench tasks JSONL with fields: task_id, problem_statement, patch.",
    )
    p.add_argument("--gso_dataset_name", type=str, default="gso-bench/gso")
    p.add_argument("--gso_split", type=str, default="test")

    # -----------------------------
    # Task embedding settings (copied from existing scripts)
    # -----------------------------
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to embeddings cache (.npz).")
    p.add_argument("--backbone", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto", help="HF device_map (e.g. auto). Use 'none' to force single-device .to(device).")
    p.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="auto", help="e.g. auto, flash_attention_2")
    p.add_argument("--embedding_layer", type=int, default=-1, help="Which hidden layer to pool embeddings from (-1 means last).")
    p.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Instruction text appended last in the embedding input. If empty, defaults to the difficulty script's instruction.",
    )

    # -----------------------------
    # Train/test split settings
    # -----------------------------
    p.add_argument("--split_by", type=str, default="task", choices=["task", "agent", "observation"], help="Hold-out split unit.")
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--cv_folds", type=int, default=5, help="If >=2, run group K-fold CV (default: 5). If <2, use --test_fraction holdout.")

    # -----------------------------
    # Learned embedding + MLP settings
    # -----------------------------
    p.add_argument("--model_emb_dim", type=int, default=64)
    p.add_argument("--scaffold_emb_dim", type=int, default=64)
    p.add_argument("--hidden_dims", type=str, default="256,128", help="Comma-separated hidden layer sizes.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda", help="torch device for training/eval (e.g. cuda, cpu).")

    # -----------------------------
    # Baselines
    # -----------------------------
    p.add_argument("--baseline_alpha", type=float, default=1.0, help="Beta prior alpha for empirical-rate baselines (Laplace smoothing).")
    p.add_argument("--baseline_beta", type=float, default=1.0, help="Beta prior beta for empirical-rate baselines (Laplace smoothing).")
    p.add_argument(
        "--ridge_regressor",
        type=str,
        default="ridge_cv",
        choices=["ridge", "ridge_cv"],
        help="Embeddings-only Ridge baseline variant (fixed alpha vs CV sweep).",
    )
    p.add_argument("--ridge_alpha", type=float, default=1.0, help="L2 penalty for Ridge when --ridge_regressor=ridge.")
    p.add_argument("--ridge_alphas", type=str, default="1e-4,1e-3,0.01,0.1,1,10,100,1e3,1e4", help="Comma-separated alpha grid for RidgeCV.")
    p.add_argument("--ridge_inner_splits", type=int, default=5, help="Inner CV splits for RidgeCV (capped by train size; must be >=2).")

    # -----------------------------
    # Oracle: IRT trained on ALL data
    # -----------------------------
    p.add_argument("--oracle_irt_model", type=str, default="1d_1pl", choices=["1d_1pl", "2d_1pl"])
    p.add_argument("--oracle_irt_epochs", type=int, default=5000)
    p.add_argument("--oracle_irt_device", type=str, default="cuda")
    p.add_argument("--oracle_irt_lr", type=float, default=0.01)

    # -----------------------------
    # Debug / exports
    # -----------------------------
    p.add_argument("--write_rows_csv", type=str, default="", help="Optional CSV path to write expanded (agent, task, y) rows.")
    p.add_argument(
        "--pca",
        action="store_true",
        help="Enable PCA diagnostics: PC1/PC2(model/scaffold learned embeddings) vs oracle IRT ability (theta).",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure deterministic training on CUDA doesn't crash due to CuBLAS reproducibility settings.
    # Must be set before torch initializes CuBLAS.
    dev_s = str(getattr(args, "device", "cuda") or "cuda")
    if dev_s.startswith("cuda"):
        if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    _set_seeds(int(args.seed))

    # Resolve benchmarks + per-benchmark jsonl paths.
    train_benchmarks = _parse_benchmarks(str(args.train_benchmarks))
    if len(train_benchmarks) < 1:
        raise ValueError("--train_benchmarks is empty")

    bench_to_jsonl: Dict[str, str] = {}
    bench_to_norm: Dict[str, bool] = {}
    for b in train_benchmarks:
        if b == "verified":
            bench_to_jsonl[b] = str(args.verified_agent_results_jsonl or "").strip()
            bench_to_norm[b] = True
        elif b == "pro":
            bench_to_jsonl[b] = str(args.pro_agent_results_jsonl or "").strip()
            bench_to_norm[b] = True
        elif b == "terminal_bench":
            bench_to_jsonl[b] = str(args.terminal_bench_agent_results_jsonl or "").strip()
            bench_to_norm[b] = False
        elif b == "gso":
            bench_to_jsonl[b] = str(args.gso_agent_results_jsonl or "").strip()
            bench_to_norm[b] = True
        else:
            raise AssertionError(f"Unhandled benchmark: {b}")

    missing_paths = [b for b in train_benchmarks if not bench_to_jsonl.get(b)]
    if missing_paths:
        raise ValueError(f"Missing required agent-results JSONL path(s) for: {missing_paths}. Set the corresponding --*_agent_results_jsonl.")
    for b, pth in bench_to_jsonl.items():
        if not os.path.exists(pth):
            raise FileNotFoundError(f"{b} agent_results_jsonl not found: {pth}")

    # Heavy deps needed from here on.
    _require("numpy")
    _require("sklearn")
    _require("torch")
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    import predict_question_difficulty as diff_base

    def _pca_pc2_vs_theta(
        *,
        out_dir: str,
        entity: str,
        emb_weight: "torch.Tensor",
        id_to_idx: Dict[str, int],
        theta_by_id: Dict[str, float],
        seed: int,
    ) -> Dict[str, Any]:
        """
        PCA learned embedding vectors and save PC2 vs IRT theta scatter.
        Returns a small dict with summary stats + output paths.
        """
        try:
            from sklearn.decomposition import PCA  # type: ignore
        except Exception as e:
            return {"enabled": True, "entity": str(entity), "error": f"sklearn PCA import failed: {e}"}

        ids = [i for i in sorted(id_to_idx.keys()) if i != "__UNK__" and i in theta_by_id]
        if len(ids) < 3:
            return {
                "enabled": True,
                "entity": str(entity),
                "n": int(len(ids)),
                "error": f"Not enough IDs with both embedding and theta (need >=3, got {len(ids)}).",
            }

        idx = np.asarray([int(id_to_idx[i]) for i in ids], dtype=np.int64)
        X = emb_weight.detach().float().cpu().numpy()[idx]
        theta = np.asarray([float(theta_by_id[i]) for i in ids], dtype=np.float64)

        pca = PCA(n_components=2, random_state=int(seed))
        Z = pca.fit_transform(X)
        pc1 = np.asarray(Z[:, 0], dtype=np.float64)
        pc2 = np.asarray(Z[:, 1], dtype=np.float64)

        # Optional linear fit of PC2 on theta (handy for metadata; doesn't affect plots).
        try:
            if float(np.nanstd(theta)) > 0:
                a2, b2 = np.polyfit(theta, pc2, deg=1)
            else:
                a2, b2 = float("nan"), float("nan")
        except Exception:
            a2, b2 = float("nan"), float("nan")

        def _pearson(x: np.ndarray, y: np.ndarray) -> float:
            if x.size < 2:
                return float("nan")
            c = np.corrcoef(x, y)
            return float(c[0, 1])

        r_pc2 = _pearson(pc2, theta)
        r_pc1 = _pearson(pc1, theta)

        csv_path = os.path.join(out_dir, f"embed_pca_vs_oracle_theta__{str(entity)}.csv")
        import csv

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "theta", "pc1", "pc2"])
            for i, th, a, b in zip(ids, theta.tolist(), pc1.tolist(), pc2.tolist()):
                w.writerow([str(i), float(th), float(a), float(b)])

        png_pc1_path = os.path.join(out_dir, f"embed_pc1_vs_oracle_theta__{str(entity)}.png")
        png_pc2_path = os.path.join(out_dir, f"embed_pc2_vs_oracle_theta__{str(entity)}.png")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

            def _short_label(s: str, *, max_len: int = 28) -> str:
                ss = str(s)
                if len(ss) <= int(max_len):
                    return ss
                return ss[: int(max_len) - 1] + "…"

            evr = getattr(pca, "explained_variance_ratio_", None)
            evr1 = float(evr[0]) if isinstance(evr, np.ndarray) and evr.size >= 1 else float("nan")
            evr2 = float(evr[1]) if isinstance(evr, np.ndarray) and evr.size >= 2 else float("nan")

            plt.figure(figsize=(7, 5))
            plt.scatter(theta, pc1, s=12, alpha=0.8)
            plt.xlabel("Oracle IRT ability (theta)")
            plt.ylabel("PCA dim 1 (learned embedding)")
            plt.title(f"{str(entity)} embeddings: PC1 vs theta (r={r_pc1:.3f}, EVR1={evr1:.3f}, n={len(ids)})")
            plt.tight_layout()
            plt.savefig(png_pc1_path, dpi=200)
            plt.close()

            plt.figure(figsize=(7, 5))
            plt.scatter(theta, pc2, s=12, alpha=0.8)
            # Label points (n is typically small: ~40-50).
            for x, yv, lab in zip(theta.tolist(), pc2.tolist(), ids):
                plt.annotate(
                    _short_label(str(lab)),
                    (float(x), float(yv)),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                    alpha=0.85,
                )
            plt.xlabel("Oracle IRT ability (theta)")
            plt.ylabel("PCA dim 2 (learned embedding)")
            plt.title(f"{str(entity)} embeddings: PC2 vs theta (r={r_pc2:.3f}, EVR2={evr2:.3f}, n={len(ids)})")
            plt.tight_layout()
            plt.savefig(png_pc2_path, dpi=200)
            plt.close()
        except Exception as e:
            png_pc1_path = ""
            png_pc2_path = ""
            plot_err = str(e)
        else:
            plot_err = ""

        evr = getattr(pca, "explained_variance_ratio_", None)
        return {
            "enabled": True,
            "entity": str(entity),
            "n": int(len(ids)),
            "pearson_r_pc2_vs_theta": float(r_pc2),
            "pearson_r_pc1_vs_theta": float(r_pc1),
            "pc2_on_theta_slope": float(a2),
            "explained_variance_ratio": (evr.tolist() if isinstance(evr, np.ndarray) else None),
            "csv_path": str(csv_path),
            # Backwards-compatible key (was PC2).
            "png_path": str(png_pc2_path) if png_pc2_path else "",
            "png_pc1_path": str(png_pc1_path) if png_pc1_path else "",
            "png_pc2_path": str(png_pc2_path) if png_pc2_path else "",
            **({"plot_error": str(plot_err)} if plot_err else {}),
        }

    if not str(args.instruction or "").strip():
        args.instruction = str(diff_base.DIFFICULTY_INSTRUCTION)

    # Global policy: deterministic ON for everything except IRT blocks.
    _set_torch_determinism(deterministic=True)

    # Expand response matrices into per-observation rows, splitting subject -> (model, scaffold).
    obs: List[Obs] = []
    n_dropped_unsplittable = 0
    task_ids_by_bench: Dict[str, List[str]] = {b: [] for b in train_benchmarks}
    seen_task_ids_by_bench: Dict[str, set] = {b: set() for b in train_benchmarks}

    for b in train_benchmarks:
        jsonl_path = bench_to_jsonl[b]
        for subj, tid, yy in _iter_obs_from_responses(benchmark=b, agent_results_jsonl=jsonl_path, normalize_item_ids=bool(bench_to_norm[b])):
            m, sc = _split_subject_to_model_scaffold(benchmark=b, subject_id=subj)
            if not m or not sc:
                n_dropped_unsplittable += 1
                continue
            task_id = str(tid)
            task_key = f"{b}::{task_id}"
            obs.append(Obs(benchmark=b, subject_id=str(subj), model=str(m), scaffold=str(sc), task_id=task_id, task_key=task_key, y=int(yy)))
            if task_id not in seen_task_ids_by_bench[b]:
                seen_task_ids_by_bench[b].add(task_id)
                task_ids_by_bench[b].append(task_id)

    if not obs:
        raise RuntimeError("No observations were loaded (after dropping unsplittable subjects). Check inputs.")
    _p(
        f"[progress] Loaded observations: n={len(obs)} "
        f"(benchmarks={train_benchmarks}, dropped_unsplittable={int(n_dropped_unsplittable)})"
    )

    # Load/build task embeddings (by task_key).
    emb_cache = str(args.embeddings_cache or "").strip()
    if not emb_cache:
        emb_cache = os.path.join(out_dir, f"task_embeddings__{diff_base.prompt_signature(str(args.instruction))}.npz")

    emb_by_key, emb_dim, cache_path = _load_or_build_task_embeddings(
        out_dir=out_dir,
        embeddings_cache=emb_cache,
        overwrite=bool(args.overwrite),
        backbone=str(args.backbone),
        trust_remote_code=bool(args.trust_remote_code),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        device_map=str(args.device_map),
        torch_dtype=str(args.torch_dtype),
        attn_implementation=str(args.attn_implementation),
        instruction=str(args.instruction),
        embedding_layer=int(args.embedding_layer),
        verified_dataset_name=str(args.verified_dataset_name),
        verified_split=str(args.verified_split),
        pro_dataset_name=str(args.pro_dataset_name),
        pro_split=str(args.pro_split),
        terminal_bench_tasks_jsonl=str(args.terminal_bench_tasks_jsonl),
        gso_dataset_name=str(args.gso_dataset_name),
        gso_split=str(args.gso_split),
        task_ids_by_benchmark=task_ids_by_bench,
    )

    # Drop observations whose task_key was not embedded (missing in task source).
    obs2: List[Obs] = []
    n_dropped_missing_task = 0
    for r in obs:
        if r.task_key not in emb_by_key:
            n_dropped_missing_task += 1
            continue
        obs2.append(r)
    obs = obs2
    if not obs:
        raise RuntimeError("All observations were dropped due to missing task embeddings. Check task sources.")

    # Optional row export.
    if str(args.write_rows_csv or "").strip():
        csv_path = str(args.write_rows_csv)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["benchmark", "subject_id", "model", "scaffold", "task_id", "task_key", "y"],
            )
            w.writeheader()
            for r in obs:
                w.writerow(
                    {
                        "benchmark": r.benchmark,
                        "subject_id": r.subject_id,
                        "model": r.model,
                        "scaffold": r.scaffold,
                        "task_id": r.task_id,
                        "task_key": r.task_key,
                        "y": int(r.y),
                    }
                )
        print(f"Wrote rows CSV: {csv_path} (n={len(obs)})")

    # Groups for splitting.
    groups: List[str] = []
    y: List[int] = []
    for r in obs:
        y.append(int(r.y))
        if str(args.split_by) == "task":
            groups.append(str(r.task_key))
        elif str(args.split_by) == "agent":
            groups.append(f"{r.benchmark}::{r.subject_id}")
        else:
            groups.append(f"{r.benchmark}::{r.subject_id}::{r.task_id}")

    # Parse hidden dims.
    try:
        hidden_dims = tuple(int(x.strip()) for x in str(args.hidden_dims).split(",") if x.strip())
    except Exception as e:
        raise ValueError(f"Failed to parse --hidden_dims={args.hidden_dims!r}: {e}") from e
    if not hidden_dims:
        raise ValueError("--hidden_dims must contain at least one integer size, e.g. '256,128'")

    # Metrics helpers.
    def _eval_probs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        preds = (probs >= 0.5).astype(np.int64)
        out: Dict[str, float] = {}
        try:
            from sklearn.metrics import roc_auc_score

            out["roc_auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            out["roc_auc"] = float("nan")
        out["accuracy"] = float(accuracy_score(y_true, preds))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, preds))
        try:
            from sklearn.metrics import log_loss

            out["log_loss"] = float(log_loss(y_true, np.stack([1.0 - probs, probs], axis=1), labels=[0, 1]))
        except Exception:
            out["log_loss"] = float("nan")
        return out

    def _mean_std(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}

    def _beta_smooth_rate(successes: int, trials: int, *, alpha: float, beta: float) -> float:
        a = float(alpha)
        b = float(beta)
        s = float(max(0, int(successes)))
        n = float(max(0, int(trials)))
        denom = n + a + b
        if denom <= 0:
            return 0.5
        return float((s + a) / denom)

    class AgentTaskSuccessNet(nn.Module):
        def __init__(
            self,
            *,
            n_models: int,
            n_scaffolds: int,
            task_dim: int,
            model_emb_dim: int,
            scaffold_emb_dim: int,
            hidden: Tuple[int, ...],
            dropout: float,
        ) -> None:
            super().__init__()
            self.model_emb = nn.Embedding(int(n_models), int(model_emb_dim))
            self.scaffold_emb = nn.Embedding(int(n_scaffolds), int(scaffold_emb_dim))

            in_dim = int(model_emb_dim) + int(scaffold_emb_dim) + int(task_dim)
            layers: List[nn.Module] = []
            d = in_dim
            for h in hidden:
                layers.append(nn.Linear(d, int(h)))
                layers.append(nn.ReLU())
                if float(dropout) > 0:
                    layers.append(nn.Dropout(float(dropout)))
                d = int(h)
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, model_idx: torch.Tensor, scaffold_idx: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
            m = self.model_emb(model_idx)
            s = self.scaffold_emb(scaffold_idx)
            x = torch.cat([m, s, task_emb], dim=-1)
            return self.mlp(x).squeeze(-1)

    def _build_id_maps(train_idx: List[int]) -> Tuple[Dict[str, int], Dict[str, int]]:
        # Train-only mapping with UNK=0 to handle unseen in test folds.
        models = sorted(set([obs[i].model for i in train_idx]))
        scaffolds = sorted(set([obs[i].scaffold for i in train_idx]))
        model_to_idx = {"__UNK__": 0, **{m: (j + 1) for j, m in enumerate(models)}}
        scaffold_to_idx = {"__UNK__": 0, **{s: (j + 1) for j, s in enumerate(scaffolds)}}
        return model_to_idx, scaffold_to_idx

    def _tensorize(indices: List[int], model_to_idx: Dict[str, int], scaffold_to_idx: Dict[str, int], *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m_idx: List[int] = []
        s_idx: List[int] = []
        t_emb: List[np.ndarray] = []
        yy: List[int] = []
        for i in indices:
            r = obs[i]
            m_idx.append(int(model_to_idx.get(r.model, 0)))
            s_idx.append(int(scaffold_to_idx.get(r.scaffold, 0)))
            t_emb.append(np.asarray(emb_by_key[r.task_key], dtype=np.float32))
            yy.append(int(r.y))
        m_t = torch.tensor(m_idx, dtype=torch.long, device=device)
        s_t = torch.tensor(s_idx, dtype=torch.long, device=device)
        t_t = torch.tensor(np.stack(t_emb, axis=0), dtype=torch.float32, device=device)
        y_t = torch.tensor(yy, dtype=torch.float32, device=device)
        return m_t, s_t, t_t, y_t

    @torch.no_grad()
    def _predict_probs(
        net: AgentTaskSuccessNet,
        idxs: List[int],
        model_to_idx: Dict[str, int],
        scaffold_to_idx: Dict[str, int],
        *,
        device: torch.device,
        batch_size: int,
    ) -> np.ndarray:
        net.eval()
        out: List[np.ndarray] = []
        bs = int(batch_size)
        for off in range(0, len(idxs), bs):
            chunk = idxs[off : off + bs]
            m_t, s_t, t_t, _ = _tensorize(chunk, model_to_idx, scaffold_to_idx, device=device)
            logits = net(m_t, s_t, t_t)
            probs = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)
            out.append(probs.astype(np.float64, copy=False))
        if not out:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(out, axis=0)

    def _fit_eval_fold(train_idx: List[int], test_idx: List[int], *, fold_seed: int, fold_idx: int) -> Dict[str, Any]:
        _set_seeds(int(fold_seed))
        _set_torch_determinism(deterministic=True)
        device = torch.device(str(args.device))

        model_to_idx, scaffold_to_idx = _build_id_maps(train_idx)
        net = AgentTaskSuccessNet(
            n_models=len(model_to_idx),
            n_scaffolds=len(scaffold_to_idx),
            task_dim=int(emb_dim),
            model_emb_dim=int(args.model_emb_dim),
            scaffold_emb_dim=int(args.scaffold_emb_dim),
            hidden=hidden_dims,
            dropout=float(args.dropout),
        ).to(device)

        opt = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_bs = int(args.train_batch_size)
        idx_train = list(train_idx)
        for _epoch in range(int(args.epochs)):
            net.train()
            random.shuffle(idx_train)
            loss_sum = 0.0
            n_seen = 0
            for off in range(0, len(idx_train), train_bs):
                chunk = idx_train[off : off + train_bs]
                m_t, s_t, t_t, y_t = _tensorize(chunk, model_to_idx, scaffold_to_idx, device=device)
                logits = net(m_t, s_t, t_t)
                loss = loss_fn(logits, y_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                bs = int(y_t.shape[0])
                loss_sum += float(loss.detach().item()) * float(bs)
                n_seen += bs
            if int(args.epochs) <= 20 or ((_epoch + 1) % 5 == 0) or (_epoch == 0) or (_epoch + 1 == int(args.epochs)):
                avg = (loss_sum / float(max(1, n_seen))) if n_seen else float("nan")
                _p(f"[progress] train epoch {_epoch + 1}/{int(args.epochs)} loss={avg:.4f}")

        # Model scores on test.
        t0 = time.time()
        _p(f"[progress] eval: embed-MLP probs on test (n={len(test_idx)}, batch={int(args.eval_batch_size)})")
        y_test = np.array([y[i] for i in test_idx], dtype=np.int64)
        probs_test = _predict_probs(
            net,
            list(test_idx),
            model_to_idx,
            scaffold_to_idx,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        _p(f"[progress] eval: embed-MLP probs done in {time.time() - t0:.1f}s")
        emb_scores = _eval_probs(y_test, probs_test)

        # Baselines (fit on train only), matching `predict_agent_task_success.py`.
        alpha = float(args.baseline_alpha)
        beta = float(args.baseline_beta)
        model_counts: Dict[str, List[int]] = {}  # model -> [n_obs, n_success]
        task_counts: Dict[str, List[int]] = {}  # task_key -> [n_obs, n_success]
        n_train_obs = 0
        n_train_succ = 0
        for i in train_idx:
            r = obs[i]
            n_train_obs += 1
            n_train_succ += int(r.y)
            model_counts.setdefault(r.model, [0, 0])
            model_counts[r.model][0] += 1
            model_counts[r.model][1] += int(r.y)
            task_counts.setdefault(r.task_key, [0, 0])
            task_counts[r.task_key][0] += 1
            task_counts[r.task_key][1] += int(r.y)

        global_p = _beta_smooth_rate(n_train_succ, n_train_obs, alpha=alpha, beta=beta)
        p_by_model = {m: _beta_smooth_rate(succ, n, alpha=alpha, beta=beta) for m, (n, succ) in model_counts.items()}
        p_by_task = {t: _beta_smooth_rate(succ, n, alpha=alpha, beta=beta) for t, (n, succ) in task_counts.items()}

        split_by = str(args.split_by)
        if split_by == "agent":
            baseline_name = "baseline_task_empirical_rate"
            baseline_probs = np.array([float(p_by_task.get(obs[i].task_key, global_p)) for i in test_idx], dtype=np.float64)
        else:
            baseline_name = "baseline_model_empirical_rate"
            baseline_probs = np.array([float(p_by_model.get(obs[i].model, global_p)) for i in test_idx], dtype=np.float64)
        baseline_scores = _eval_probs(y_test, baseline_probs)

        # Ridge baseline in the same setup as `predict_question_difficulty.py`:
        #   1) train IRT on TRAIN split only -> difficulties b_train and abilities theta_{model,scaffold}
        #   2) fit Ridge/RidgeCV: task_embedding -> b_train (on TRAIN tasks)
        #   3) predict b_hat for TEST tasks and score TEST obs with sigmoid(theta_model+theta_scaffold - b_hat)
        try:
            from sklearn.linear_model import Ridge, RidgeCV
            from sklearn.model_selection import KFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler as _SkStandardScaler

            _require("pandas")
            _require("pyro")
            import pandas as pd  # type: ignore

            import predict_question_difficulty_multi_benchmark as pqmb

            ms = _import_swebench_irt_module("train_model_scaffold_shared")

            obs_train = [obs[i] for i in train_idx]
            model_ids = sorted(set([r.model for r in obs_train]))
            scaffold_ids = sorted(set([r.scaffold for r in obs_train]))
            item_ids = sorted(set([r.task_key for r in obs_train]))
            model_to_idx = {m: i for i, m in enumerate(model_ids)}
            scaffold_to_idx = {s: i for i, s in enumerate(scaffold_ids)}
            item_to_idx = {t: i for i, t in enumerate(item_ids)}

            m_list: List[int] = []
            s_list: List[int] = []
            i_list: List[int] = []
            y_list: List[int] = []
            agent_rows: List[dict] = []
            seen_agent_rows: set = set()

            verified_item_ids: set = set()
            pro_item_ids: set = set()
            terminal_item_ids: set = set()
            gso_item_ids: set = set()
            for r in obs_train:
                m_list.append(int(model_to_idx[r.model]))
                s_list.append(int(scaffold_to_idx[r.scaffold]))
                i_list.append(int(item_to_idx[r.task_key]))
                y_list.append(int(r.y))

                if r.benchmark == "verified":
                    verified_item_ids.add(r.task_key)
                elif r.benchmark == "pro":
                    pro_item_ids.add(r.task_key)
                elif r.benchmark == "terminal_bench":
                    terminal_item_ids.add(r.task_key)
                elif r.benchmark == "gso":
                    gso_item_ids.add(r.task_key)

                ak = (r.benchmark, r.subject_id)
                if ak not in seen_agent_rows:
                    seen_agent_rows.add(ak)
                    agent_rows.append({"benchmark": r.benchmark, "agent": r.subject_id, "model": r.model, "scaffold": r.scaffold})

            agent_split_df = pd.DataFrame(agent_rows)
            try:
                agent_split_df = agent_split_df.sort_values(["benchmark", "model", "scaffold", "agent"])
            except Exception:
                pass

            obs_full = ms.MultiBenchObs(
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

            ridge_irt_root = os.path.join(out_dir, "ridge_irt_folds", f"fold_{int(fold_idx):02d}")
            irt_dir = os.path.join(ridge_irt_root, "irt_1pl")
            if bool(args.overwrite) and os.path.exists(irt_dir):
                shutil.rmtree(irt_dir, ignore_errors=True)
            os.makedirs(ridge_irt_root, exist_ok=True)

            _p(
                f"[progress] ridge+irt: training IRT on train split "
                f"(n_obs={len(obs_train)} n_models={len(model_ids)} n_scaffolds={len(scaffold_ids)} n_train_tasks={len(item_ids)})"
            )
            # Requested policy: IRT uses deterministic=False.
            _set_torch_determinism(deterministic=False)
            _set_seeds(int(args.seed) + int(fold_idx))
            theta_by_model, theta_by_scaffold, diff_by_item = pqmb.train_irt_model_scaffold_1pl(
                obs_train=obs_full,
                irt_model=str(args.oracle_irt_model),
                epochs=int(args.oracle_irt_epochs),
                device=str(args.oracle_irt_device),
                seed=int(args.seed) + int(fold_idx),
                lr=float(args.oracle_irt_lr),
                out_dir=str(irt_dir),
            )
            # Restore deterministic=True for the remainder of the fold.
            _set_torch_determinism(deterministic=True)
            _set_seeds(int(fold_seed))
            _p("[progress] ridge+irt: IRT done; fitting ridge to predict difficulties from embeddings")

            train_tasks_labeled = sorted(set([r.task_key for r in obs_train if r.task_key in diff_by_item and r.task_key in emb_by_key]))
            if len(train_tasks_labeled) < 2:
                raise RuntimeError(f"ridge+irt: only {len(train_tasks_labeled)} train tasks had IRT difficulties; cannot fit ridge.")

            X_task_train = np.stack([emb_by_key[k] for k in train_tasks_labeled], axis=0).astype(np.float32, copy=False)
            y_train_r = np.array([float(diff_by_item[k]) for k in train_tasks_labeled], dtype=np.float32)

            reg = str(getattr(args, "ridge_regressor", "ridge_cv") or "ridge_cv").strip()
            if reg == "ridge":
                _p(
                    f"[progress] ridge: fitting Ridge(alpha={float(args.ridge_alpha):g}) "
                    f"on task embeddings (n_tasks={X_task_train.shape[0]} d={X_task_train.shape[1]})"
                )
                ridge_alpha = float(args.ridge_alpha)
                if not (ridge_alpha > 0):
                    raise ValueError("--ridge_alpha must be > 0")
                ridge_model = Pipeline(
                    steps=[
                        ("scaler", _SkStandardScaler(with_mean=True, with_std=True)),
                        ("ridge", Ridge(alpha=ridge_alpha, fit_intercept=True, random_state=None)),
                    ]
                )
            else:
                try:
                    alphas = np.array([float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()], dtype=np.float64)
                except Exception as e:
                    raise ValueError(f"Failed to parse --ridge_alphas={args.ridge_alphas!r}: {e}") from e
                if alphas.size == 0:
                    raise ValueError("Expected at least one alpha in --ridge_alphas")
                req_inner = int(args.ridge_inner_splits)
                if req_inner < 2:
                    raise ValueError("--ridge_inner_splits must be >= 2")
                inner_splits = int(min(req_inner, max(2, int(len(train_tasks_labeled)))))
                _p(
                    f"[progress] ridge: fitting RidgeCV (n_tasks={X_task_train.shape[0]} d={X_task_train.shape[1]} "
                    f"alphas={alphas.size} inner_splits={inner_splits})"
                )
                inner_cv = KFold(n_splits=int(inner_splits), shuffle=True, random_state=int(args.seed) + int(fold_seed))
                ridge_model = Pipeline(
                    steps=[
                        ("scaler", _SkStandardScaler(with_mean=True, with_std=True)),
                        ("ridge", RidgeCV(alphas=alphas, cv=inner_cv)),
                    ]
                )
            t1 = time.time()
            ridge_model.fit(X_task_train, y_train_r)
            try:
                rr = ridge_model.named_steps.get("ridge", None)
                ridge_alpha = float(getattr(rr, "alpha_", getattr(rr, "alpha", float(args.ridge_alpha))))
            except Exception:
                ridge_alpha = float(getattr(args, "ridge_alpha", 1.0))
            _p(f"[progress] ridge: fit done in {time.time() - t1:.1f}s (alpha={float(ridge_alpha):g})")

            test_tasks = sorted(set([obs[i].task_key for i in test_idx if obs[i].task_key in emb_by_key]))
            X_task_test = np.stack([emb_by_key[k] for k in test_tasks], axis=0).astype(np.float32, copy=False)
            b_hat = ridge_model.predict(X_task_test).astype(np.float64, copy=False).reshape(-1)
            b_hat_by_task = {k: float(v) for k, v in zip(test_tasks, b_hat.tolist())}

            ridge_probs: List[float] = []
            for i in test_idx:
                r = obs[i]
                tm = theta_by_model.get(r.model, None)
                ts = theta_by_scaffold.get(r.scaffold, None)
                bh = b_hat_by_task.get(r.task_key, None)
                if tm is None or ts is None or bh is None:
                    ridge_probs.append(float(global_p))
                else:
                    ridge_probs.append(float(_sigmoid((float(tm) + float(ts)) - float(bh))))
            ridge_probs_arr = np.asarray(ridge_probs, dtype=np.float64)
            ridge_scores = _eval_probs(y_test, ridge_probs_arr)
        except Exception:
            ridge_alpha = float(getattr(args, "ridge_alpha", 1.0))
            ridge_scores = {"roc_auc": float("nan"), "accuracy": float("nan"), "balanced_accuracy": float("nan"), "log_loss": float("nan")}

        # Split diagnostics.
        train_models = sorted(set([obs[i].model for i in train_idx]))
        test_models = sorted(set([obs[i].model for i in test_idx]))
        train_scaffolds = sorted(set([obs[i].scaffold for i in train_idx]))
        test_scaffolds = sorted(set([obs[i].scaffold for i in test_idx]))
        train_tasks = sorted(set([obs[i].task_key for i in train_idx]))
        test_tasks = sorted(set([obs[i].task_key for i in test_idx]))
        unseen_models_in_test = sorted(set(test_models) - set(train_models))
        unseen_scaffolds_in_test = sorted(set(test_scaffolds) - set(train_scaffolds))
        unseen_tasks_in_test = sorted(set(test_tasks) - set(train_tasks))

        return {
            "n_obs_train": int(len(train_idx)),
            "n_obs_test": int(len(test_idx)),
            "embed_mlp": dict(emb_scores),
            "baseline": {"name": str(baseline_name), "alpha": float(alpha), "beta": float(beta), **baseline_scores},
            "ridge": {"name": "Ridge", "alpha": float(ridge_alpha), **dict(ridge_scores)},
            "global_p_train": float(global_p),
            "split_diagnostics": {
                "n_models_train": int(len(train_models)),
                "n_models_test": int(len(test_models)),
                "n_models_test_unseen": int(len(unseen_models_in_test)),
                "n_scaffolds_train": int(len(train_scaffolds)),
                "n_scaffolds_test": int(len(test_scaffolds)),
                "n_scaffolds_test_unseen": int(len(unseen_scaffolds_in_test)),
                "n_tasks_train": int(len(train_tasks)),
                "n_tasks_test": int(len(test_tasks)),
                "n_tasks_test_unseen": int(len(unseen_tasks_in_test)),
            },
            "task_embedding_dim": int(emb_dim),
            "model_emb_dim": int(args.model_emb_dim),
            "scaffold_emb_dim": int(args.scaffold_emb_dim),
            "x_total_dim": int(args.model_emb_dim) + int(args.scaffold_emb_dim) + int(emb_dim),
            "n_models_train_plus_unk": int(len(model_to_idx)),
            "n_scaffolds_train_plus_unk": int(len(scaffold_to_idx)),
        }

    # Choose CV vs holdout.
    cv_folds = int(args.cv_folds)
    if cv_folds >= 2:
        fold_splits = _stable_group_kfold(groups, n_splits=cv_folds, seed=int(args.seed))
    else:
        train_g, test_g = _stable_group_split(groups, test_fraction=float(args.test_fraction), seed=int(args.seed))
        fold_splits = [(train_g, test_g)]

    # Run folds.
    fold_results: List[Dict[str, Any]] = []
    for fold_idx, (train_g, test_g) in enumerate(fold_splits):
        _p(f"[progress] Fold {fold_idx + 1}/{len(fold_splits)}: fitting embed-MLP/baselines")
        train_idx = [i for i, g in enumerate(groups) if g in train_g]
        test_idx = [i for i, g in enumerate(groups) if g in test_g]
        if not train_idx or not test_idx:
            raise RuntimeError(
                f"Fold {fold_idx}: empty train or test set (train={len(train_idx)}, test={len(test_idx)}). "
                f"Try adjusting --cv_folds/--test_fraction or --split_by."
            )
        fr = _fit_eval_fold(train_idx, test_idx, fold_seed=int(args.seed) + int(fold_idx), fold_idx=int(fold_idx))
        fr["fold_idx"] = int(fold_idx)
        fr["n_groups_train"] = int(len(train_g))
        fr["n_groups_test"] = int(len(test_g))
        fold_results.append(fr)

    # -----------------------------
    # Oracle: IRT trained on ALL data (evaluated per fold test)
    # -----------------------------
    oracle_details: Dict[str, Any] = {}
    oracle_fold_scores: List[Dict[str, float]] = []
    oracle_theta_by_model: Dict[str, float] = {}
    oracle_theta_by_scaffold: Dict[str, float] = {}
    try:
        _require("pandas")
        _require("pyro")
        import pandas as pd  # type: ignore

        import predict_question_difficulty_multi_benchmark as pqmb

        ms = _import_swebench_irt_module("train_model_scaffold_shared")

        model_ids = sorted(set([r.model for r in obs]))
        scaffold_ids = sorted(set([r.scaffold for r in obs]))
        item_ids = sorted(set([r.task_key for r in obs]))
        model_to_idx = {m: i for i, m in enumerate(model_ids)}
        scaffold_to_idx = {s: i for i, s in enumerate(scaffold_ids)}
        item_to_idx = {t: i for i, t in enumerate(item_ids)}

        m_list: List[int] = []
        s_list: List[int] = []
        i_list: List[int] = []
        y_list: List[int] = []
        agent_rows: List[dict] = []
        seen_agent_rows: set = set()

        verified_item_ids: set = set()
        pro_item_ids: set = set()
        terminal_item_ids: set = set()
        gso_item_ids: set = set()
        for r in obs:
            m_list.append(int(model_to_idx[r.model]))
            s_list.append(int(scaffold_to_idx[r.scaffold]))
            i_list.append(int(item_to_idx[r.task_key]))
            y_list.append(int(r.y))

            if r.benchmark == "verified":
                verified_item_ids.add(r.task_key)
            elif r.benchmark == "pro":
                pro_item_ids.add(r.task_key)
            elif r.benchmark == "terminal_bench":
                terminal_item_ids.add(r.task_key)
            elif r.benchmark == "gso":
                gso_item_ids.add(r.task_key)

            ak = (r.benchmark, r.subject_id)
            if ak not in seen_agent_rows:
                seen_agent_rows.add(ak)
                agent_rows.append({"benchmark": r.benchmark, "agent": r.subject_id, "model": r.model, "scaffold": r.scaffold})

        agent_split_df = pd.DataFrame(agent_rows)
        try:
            agent_split_df = agent_split_df.sort_values(["benchmark", "model", "scaffold", "agent"])
        except Exception:
            pass

        obs_full = ms.MultiBenchObs(
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

        oracle_out_dir = os.path.join(out_dir, "oracle_irt_all_data")
        _p(
            f"[progress] Oracle IRT: training on ALL data "
            f"(n_obs={len(obs)}, n_models={len(model_ids)}, n_scaffolds={len(scaffold_ids)}, n_items={len(item_ids)})"
        )
        _set_torch_determinism(deterministic=False)
        _set_seeds(int(args.seed))
        theta_by_model, theta_by_scaffold, diff_by_item = pqmb.train_irt_model_scaffold_1pl(
            obs_train=obs_full,
            irt_model=str(args.oracle_irt_model),
            epochs=int(args.oracle_irt_epochs),
            device=str(args.oracle_irt_device),
            seed=int(args.seed),
            lr=float(args.oracle_irt_lr),
            out_dir=str(oracle_out_dir),
        )
        oracle_theta_by_model = {str(k): float(v) for k, v in (theta_by_model or {}).items()}
        oracle_theta_by_scaffold = {str(k): float(v) for k, v in (theta_by_scaffold or {}).items()}
        _set_torch_determinism(deterministic=True)
        _p("[progress] Oracle IRT: training done; scoring folds")

        oracle_details = {
            "enabled": True,
            "oracle_out_dir": str(oracle_out_dir),
            "oracle_irt_model": str(args.oracle_irt_model),
            "oracle_irt_epochs": int(args.oracle_irt_epochs),
            "oracle_irt_device": str(args.oracle_irt_device),
            "oracle_irt_lr": float(args.oracle_irt_lr),
            "oracle_models": int(len(theta_by_model)),
            "oracle_scaffolds": int(len(theta_by_scaffold)),
            "oracle_items": int(len(diff_by_item)),
        }

        # Evaluate per fold.
        for fr in fold_results:
            fold_idx = int(fr["fold_idx"])
            _train_g, test_g = fold_splits[fold_idx]
            test_idx = [i for i, g in enumerate(groups) if g in test_g]
            y_test = np.array([y[i] for i in test_idx], dtype=np.int64)
            global_p = float(fr.get("global_p_train", 0.5))
            oracle_probs: List[float] = []
            for i in test_idx:
                r = obs[i]
                tm = oracle_theta_by_model.get(r.model, None)
                ts = oracle_theta_by_scaffold.get(r.scaffold, None)
                b = diff_by_item.get(r.task_key, None)
                if tm is None or ts is None or b is None:
                    oracle_probs.append(float(global_p))
                else:
                    oracle_probs.append(float(_sigmoid((float(tm) + float(ts)) - float(b))))
            oracle_probs_arr = np.asarray(oracle_probs, dtype=np.float64)
            oracle_fold_scores.append(_eval_probs(y_test, oracle_probs_arr))
    except Exception as e:
        oracle_details = {"enabled": True, "error": str(e)}

    # Aggregate fold metrics.
    emb_roc = [float(fr["embed_mlp"].get("roc_auc", float("nan"))) for fr in fold_results]
    emb_acc = [float(fr["embed_mlp"].get("accuracy", float("nan"))) for fr in fold_results]
    emb_bacc = [float(fr["embed_mlp"].get("balanced_accuracy", float("nan"))) for fr in fold_results]
    emb_ll = [float(fr["embed_mlp"].get("log_loss", float("nan"))) for fr in fold_results]

    base_roc = [float(fr["baseline"].get("roc_auc", float("nan"))) for fr in fold_results]
    base_acc = [float(fr["baseline"].get("accuracy", float("nan"))) for fr in fold_results]
    base_bacc = [float(fr["baseline"].get("balanced_accuracy", float("nan"))) for fr in fold_results]
    base_ll = [float(fr["baseline"].get("log_loss", float("nan"))) for fr in fold_results]

    ridge_roc = [float(fr.get("ridge", {}).get("roc_auc", float("nan"))) for fr in fold_results]
    ridge_acc = [float(fr.get("ridge", {}).get("accuracy", float("nan"))) for fr in fold_results]
    ridge_bacc = [float(fr.get("ridge", {}).get("balanced_accuracy", float("nan"))) for fr in fold_results]
    ridge_ll = [float(fr.get("ridge", {}).get("log_loss", float("nan"))) for fr in fold_results]

    oracle_roc = [float(d.get("roc_auc", float("nan"))) for d in oracle_fold_scores]
    oracle_acc = [float(d.get("accuracy", float("nan"))) for d in oracle_fold_scores]
    oracle_bacc = [float(d.get("balanced_accuracy", float("nan"))) for d in oracle_fold_scores]
    oracle_ll = [float(d.get("log_loss", float("nan"))) for d in oracle_fold_scores]

    cv_summary = {
        "enabled": bool(cv_folds >= 2),
        "n_folds": int(len(fold_results)),
        "split_by": str(args.split_by),
        "test_fraction_ignored_when_cv": float(args.test_fraction),
        "embed_mlp": {
            "roc_auc": _mean_std(emb_roc),
            "accuracy": _mean_std(emb_acc),
            "balanced_accuracy": _mean_std(emb_bacc),
            "log_loss": _mean_std(emb_ll),
        },
        "baseline": {
            "roc_auc": _mean_std(base_roc),
            "accuracy": _mean_std(base_acc),
            "balanced_accuracy": _mean_std(base_bacc),
            "log_loss": _mean_std(base_ll),
        },
        "ridge": {
            "name": "Ridge",
            "roc_auc": _mean_std(ridge_roc),
            "accuracy": _mean_std(ridge_acc),
            "balanced_accuracy": _mean_std(ridge_bacc),
            "log_loss": _mean_std(ridge_ll),
        },
        "oracle_irt_all_data": {
            **{k: v for k, v in (oracle_details or {}).items() if k != "error"},
            "roc_auc": _mean_std(oracle_roc),
            "accuracy": _mean_std(oracle_acc),
            "balanced_accuracy": _mean_std(oracle_bacc),
            "log_loss": _mean_std(oracle_ll),
            **({"error": oracle_details.get("error")} if isinstance(oracle_details, dict) and oracle_details.get("error") else {}),
        },
        "folds": fold_results,
    }

    metrics: Dict[str, Any] = {
        "n_obs_total": int(len(obs)),
        "n_dropped_unsplittable_subject": int(n_dropped_unsplittable),
        "n_dropped_missing_task_embedding": int(n_dropped_missing_task),
        "split_by": str(args.split_by),
        "test_fraction": float(args.test_fraction),
        "cv_folds": int(args.cv_folds),
        "cv": cv_summary,
        # Backwards-compatible top-level aliases (primary model metrics).
        "roc_auc": float(cv_summary["embed_mlp"]["roc_auc"]["mean"]),
        "accuracy": float(cv_summary["embed_mlp"]["accuracy"]["mean"]),
        "balanced_accuracy": float(cv_summary["embed_mlp"]["balanced_accuracy"]["mean"]),
        "log_loss": float(cv_summary["embed_mlp"]["log_loss"]["mean"]),
        "embed_mlp": dict(cv_summary["embed_mlp"]),
        "baseline": dict(cv_summary["baseline"]),
        "ridge": dict(cv_summary["ridge"]),
        "oracle_irt_all_data": dict(cv_summary["oracle_irt_all_data"]),
        "train_benchmarks": list(train_benchmarks),
        "embeddings_cache": str(cache_path),
        "model_emb_dim": int(args.model_emb_dim),
        "scaffold_emb_dim": int(args.scaffold_emb_dim),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "device": str(args.device),
    }

    # Print a compact, human-readable summary (stdout), then the full JSON.
    try:
        mlp_auc = cv_summary["embed_mlp"]["roc_auc"]
        base_auc = cv_summary["baseline"]["roc_auc"]
        ridge_auc = cv_summary["ridge"]["roc_auc"]
        oracle_auc = cv_summary["oracle_irt_all_data"]["roc_auc"]
        oracle_enabled = bool(cv_summary.get("oracle_irt_all_data", {}).get("enabled", True))
        oracle_err = cv_summary.get("oracle_irt_all_data", {}).get("error", None)
        _p("=== Key metrics (CV mean ROC-AUC) ===")
        _p(f"MLP:      {float(mlp_auc.get('mean', float('nan'))):.4f} ± {float(mlp_auc.get('std', float('nan'))):.4f}")
        _p(f"Baseline: {float(base_auc.get('mean', float('nan'))):.4f} ± {float(base_auc.get('std', float('nan'))):.4f}")
        _p(f"Ridge:    {float(ridge_auc.get('mean', float('nan'))):.4f} ± {float(ridge_auc.get('std', float('nan'))):.4f}")
        if not oracle_enabled:
            _p("Oracle:   disabled")
        elif oracle_err:
            _p(f"Oracle:   error ({str(oracle_err)})")
        else:
            _p(
                f"Oracle:   {float(oracle_auc.get('mean', float('nan'))):.4f} ± {float(oracle_auc.get('std', float('nan'))):.4f}"
            )
        _p("")
    except Exception:
        pass

    # If PCA isn't requested, stop here (no final-train / no .pt bundle).
    if not bool(getattr(args, "pca", False)):
        print(json.dumps(metrics, indent=2, sort_keys=True))
        meta_path = os.path.join(out_dir, "agent_task_success_embed_ms_metrics.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote metrics: {meta_path}")
        return 0

    # -----------------------------
    # Train final model on ALL data (and save) [PCA only]
    # -----------------------------
    device = torch.device(str(args.device))

    # All-data mapping (no UNK needed, but keep UNK=0 for safety / future inference).
    all_models = sorted(set([r.model for r in obs]))
    all_scaffolds = sorted(set([r.scaffold for r in obs]))
    model_to_idx_all = {"__UNK__": 0, **{m: (j + 1) for j, m in enumerate(all_models)}}
    scaffold_to_idx_all = {"__UNK__": 0, **{s: (j + 1) for j, s in enumerate(all_scaffolds)}}

    net = AgentTaskSuccessNet(
        n_models=len(model_to_idx_all),
        n_scaffolds=len(scaffold_to_idx_all),
        task_dim=int(emb_dim),
        model_emb_dim=int(args.model_emb_dim),
        scaffold_emb_dim=int(args.scaffold_emb_dim),
        hidden=hidden_dims,
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    all_idx = list(range(len(obs)))
    train_bs = int(args.train_batch_size)
    for _epoch in range(int(args.epochs)):
        net.train()
        random.shuffle(all_idx)
        loss_sum = 0.0
        n_seen = 0
        for off in range(0, len(all_idx), train_bs):
            chunk = all_idx[off : off + train_bs]
            m_t, s_t, t_t, y_t = _tensorize(chunk, model_to_idx_all, scaffold_to_idx_all, device=device)
            logits = net(m_t, s_t, t_t)
            loss = loss_fn(logits, y_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = int(y_t.shape[0])
            loss_sum += float(loss.detach().item()) * float(bs)
            n_seen += bs
        if int(args.epochs) <= 20 or ((_epoch + 1) % 5 == 0) or (_epoch == 0) or (_epoch + 1 == int(args.epochs)):
            avg = (loss_sum / float(max(1, n_seen))) if n_seen else float("nan")
            _p(f"[progress] final-train epoch {_epoch + 1}/{int(args.epochs)} loss={avg:.4f}")

    # PCA diagnostics: learned embeddings vs oracle IRT theta.
    if oracle_theta_by_model and oracle_theta_by_scaffold:
        try:
            pca_diag = {
                "model": _pca_pc2_vs_theta(
                    out_dir=out_dir,
                    entity="model",
                    emb_weight=net.model_emb.weight,
                    id_to_idx=model_to_idx_all,
                    theta_by_id=oracle_theta_by_model,
                    seed=int(args.seed),
                ),
                "scaffold": _pca_pc2_vs_theta(
                    out_dir=out_dir,
                    entity="scaffold",
                    emb_weight=net.scaffold_emb.weight,
                    id_to_idx=scaffold_to_idx_all,
                    theta_by_id=oracle_theta_by_scaffold,
                    seed=int(args.seed),
                ),
            }
        except Exception as e:
            pca_diag = {"enabled": True, "error": str(e)}
        metrics["embed_mlp_embedding_pca_vs_oracle_theta"] = pca_diag

    # Now print the full JSON (including PCA diagnostics, if any).
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Save bundle.
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "model_to_idx": model_to_idx_all,
        "scaffold_to_idx": scaffold_to_idx_all,
        "task_embedding_dim": int(emb_dim),
        "model_emb_dim": int(args.model_emb_dim),
        "scaffold_emb_dim": int(args.scaffold_emb_dim),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "metrics": metrics,
        "embeddings_cache": str(cache_path),
        "train_benchmarks": list(train_benchmarks),
        "split_by": str(args.split_by),
    }

    model_path = os.path.join(out_dir, "agent_task_success_embed_ms.pt")
    torch.save(bundle, model_path)

    meta_path = os.path.join(out_dir, "agent_task_success_embed_ms_metrics.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote model bundle: {model_path}")
    print(f"Wrote metrics: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

