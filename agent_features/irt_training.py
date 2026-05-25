"""Shared IRT training helpers used by the split experiments.

This module contains the small subset of shared IRT code that the newer
experiment packages need.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple


_V_SUFFIX_RE = re.compile(r"-v(?:\d+|[0-9a-f]{6,}|nan)$", re.IGNORECASE)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_swebench_item_id(raw_item_id: str) -> str:
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_") :]
    s = _V_SUFFIX_RE.sub("", s)
    return s.strip()


def _import_swebench_irt_module(module_name: str):
    swe_irt_dir = str(Path(__file__).resolve().parents[1] / "swebench_irt")
    if swe_irt_dir not in sys.path:
        sys.path.insert(0, swe_irt_dir)
    return __import__(str(module_name))


def _import_shared_irt_module():
    return _import_swebench_irt_module("train_model_scaffold_shared")


def train_irt_1pl(
    *,
    responses_jsonl: str,
    epochs: int,
    device: str,
    seed: int,
    out_dir: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train a standard agent/item 1PL IRT model and write legacy outputs."""

    import pyro
    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    _ensure_dir(str(out_dir))
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
        tid = _normalize_swebench_item_id(str(item_map.get(i, "")).strip())
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
    """Train a standard 1PL IRT model over agents and items."""

    import torch

    outp = str(out_dir or "").strip()
    if not outp:
        raise ValueError("out_dir was empty")

    if os.path.exists(outp):
        shutil.rmtree(outp, ignore_errors=True)
    _ensure_dir(outp)

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

    return train_irt_1pl(
        responses_jsonl=str(train_jsonl),
        epochs=int(epochs),
        device=str(dev),
        seed=int(seed),
        out_dir=str(outp),
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
    """Build model/scaffold/item observations from tagged response rows."""

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
    """Train a model+scaffold 1PL IRT model and write legacy outputs."""

    import pyro
    import torch

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
