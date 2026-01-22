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
  - Optional (`--evaluate_in_distribution`): K-fold CV over items. For each fold,
    train IRT on train-fold items only (no leakage), fit a regressor, predict held-out
    item difficulties, and evaluate held-out AUROC using the fold's IRT abilities.
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
    epochs: int,
    device: str,
    seed: int,
    lr: float,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Train shared model+scaffold 1PL on `obs_train`.

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

    model_obj = ms.ModelScaffold1PL(len(obs_dev.model_ids), len(obs_dev.scaffold_ids), len(obs_dev.item_ids))
    _ = ms.train_svi(model_obj.model, model_obj.guide, obs_dev, epochs=int(epochs), lr=float(lr))

    # Save fold artifacts in the same spirit as the single-benchmark script.
    outp = Path(str(out_dir))
    outp.mkdir(parents=True, exist_ok=True)
    ms.save_outputs(out_dir=outp, obs=obs_dev, model_type="1pl")
    try:
        obs_dev.agent_split_df.to_csv(outp / "agent_splits.csv", index=False)
    except Exception:
        pass

    # Extract centered abilities + item difficulties.
    theta_m_raw = pyro.param("loc_theta_model_raw").detach().cpu()
    theta_s_raw = pyro.param("loc_theta_scaffold_raw").detach().cpu()
    theta_m = (theta_m_raw - theta_m_raw.mean()).numpy().tolist()
    theta_s = (theta_s_raw - theta_s_raw.mean()).numpy().tolist()

    b_loc = pyro.param("loc_b").detach().cpu().numpy().tolist()

    theta_by_model: Dict[str, float] = {str(mid): float(theta_m[i]) for i, mid in enumerate(obs_dev.model_ids)}
    theta_by_scaffold: Dict[str, float] = {str(sid): float(theta_s[i]) for i, sid in enumerate(obs_dev.scaffold_ids)}
    diff_by_item: Dict[str, float] = {str(iid): float(b_loc[i]) for i, iid in enumerate(obs_dev.item_ids)}
    return theta_by_model, theta_by_scaffold, diff_by_item


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()

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

    # -----------------------------
    # Embedding model settings (identical)
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

    p.add_argument("--out_dir", type=str, default="/orcd/scratch/orcd/001/daria_k/fulcrum/fellowship/out/multi_benchmark")
    p.add_argument("--embeddings_cache", type=str, default="", help="Optional path to existing embeddings cache (.npz).")
    p.add_argument("--overwrite", action="store_true")

    # -----------------------------
    # Multi-benchmark response matrices (for IRT + evaluation)
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

    p.add_argument(
        "--include_zero_success",
        action="store_true",
        help="Include items with 0 successes in CV/IRT (not recommended; can destabilize IRT).",
    )
    p.add_argument(
        "--evaluate_in_distribution",
        action="store_true",
        help=(
            "If set, run the existing in-distribution K-fold CV + AUROC evaluation pipeline. "
            "If not set (default), do a single split into train vs zero-success, train IRT + regressor on train, "
            "and only save the learned regression weights (no evaluation / predictions)."
        ),
    )
    p.add_argument("--irt_epochs", type=int, default=5000)
    p.add_argument("--irt_device", type=str, default="cuda", help="Device for IRT training (cuda or cpu).")
    p.add_argument("--irt_lr", type=float, default=0.01, help="Learning rate for Pyro SVI (shared model+scaffold IRT).")
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

    args = p.parse_args(argv)
    base.ensure_dir(args.out_dir)
    base.seed_everything(int(args.seed), deterministic=True)

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

    if bool(args.evaluate_in_distribution):
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
                epochs=int(args.irt_epochs),
                device=str(args.irt_device),
                seed=int(args.seed),
                lr=float(args.irt_lr),
                out_dir=os.path.join(fold_root, "irt_model_scaffold_1pl"),
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
            "evaluate_in_distribution": True,
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
            "irt_model": "model+scaffold 1pl (shared)",
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
            "evaluate_in_distribution": True,
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
            "irt_model": "model+scaffold 1pl (shared)",
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
            "NOTE: --include_zero_success is ignored when --evaluate_in_distribution is not set. "
            "Training-only mode always splits into train vs zero-success and trains only on the train split."
        )

    train_items = [tid for tid in overlap_ids if tid not in zero_success_set]
    if not train_items:
        raise RuntimeError("Training-only mode: after excluding zero-success items, no train items remain.")

    # IRT on train items only.
    base.set_torch_determinism(False)
    base.seed_everything(int(args.seed), deterministic=False)
    obs_train = build_multibench_obs_for_items(obs_full=obs_full, keep_item_ids=train_items)
    _, _, diff_by_item = train_irt_model_scaffold_1pl(
        obs_train=obs_train,
        epochs=int(args.irt_epochs),
        device=str(args.irt_device),
        seed=int(args.seed),
        lr=float(args.irt_lr),
        out_dir=os.path.join(str(args.out_dir), "irt_model_scaffold_1pl"),
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

    model = _make_model(n_train=int(len(train_labeled)), fold_seed=int(args.seed))
    model.fit(X_train, y_train)

    ridge_alpha = None
    if regressor_name in ("ridge", "ridge_cv"):
        try:
            ridge_alpha = float(model.named_steps["ridge"].alpha_)
        except Exception:
            ridge_alpha = None

    weights_meta = {
        "evaluate_in_distribution": False,
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
        "irt_model": "model+scaffold 1pl (shared)",
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
    base.save_regression_weights(
        out_dir=str(args.out_dir),
        model=model,
        regressor_name=str(regressor_name),
        feature_dim=int(Xy.shape[1]),
        metadata=weights_meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
