"""
Train a 1D IRT model on multiple benchmarks with *shared* model and scaffold abilities.

Use case (Fulcrum SWE-bench + Terminal-Bench):
----------------------------
We want one shared set of (theta_model, theta_scaffold) that explains agent performance
across multiple benchmarks (e.g., SWE-bench Verified, SWE-bench Pro, Terminal-Bench 2.0), while keeping
*item parameters* benchmark-specific (because the tasks differ).

Model:
  p(y=1 | m, s, i) = sigmoid( a_i * ( theta_model[m] + theta_scaffold[s] - b_i ) )

where items i are identified by their task name / instance id. We assume the
benchmarks provide distinct task names, so explicit benchmark prefixes are not needed.

Input format:
-------------
Each benchmark is a JSONL with one record per subject/agent:
  {"subject_id": "<agent_or_model_name>", "responses": {"<instance_id>": 0|1, ...}}

Notes for SWE-bench Pro:
------------------------
In the Pro JSONL used here, `subject_id` is a model name (not a dated agent id).
Per analysis convention:
  - All Pro subjects are assigned scaffold="SWE-agent 1.0"
  - Pro model names are canonicalized/merged so they align with Verified where desired:
      * Merge "Claude 4 Sonnet" paper+dated variants -> "Claude 4 Sonnet"
      * Merge "Gemini 2.5 Pro Preview" paper+debug variants -> "Gemini 2.5 Pro Preview"
      * "GPT-5" -> "GPT-5" (merges with Verified)
      * "Claude 4 Sonnet" -> "Claude 4 Sonnet" (merges with Verified)
      * "Claude 4.5 Sonnet" -> "claude-sonnet-4-5" (merges with Verified's token)
      * "Kimi" -> "kimi_k2_instruct" (merges with Verified)

Usage:
------
python fulcrum/fellowship/swebench_irt/train_multibench_shared_model_scaffold.py \
  --verified_path fulcrum/fellowship/out/chris_irt/swebench_verified_20251115_full.jsonl \
  --pro_path fulcrum/fellowship/out/chris_irt/swebench_pro.jsonl \
  --terminal_bench_path fulcrum/fellowship/out/chris_irt/terminal_bench_2.0.jsonl \
  --output_dir clean_data/training_results_shared_verified_pro \
  --epochs 5000 \
  --model 1pl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def resolve_output_dir(path_str: str) -> Path:
    """
    Match swebench_irt/train.py behavior:
    - absolute paths used as-is
    - relative paths with separators resolved from repo ROOT
    - bare names go under chris_output/clean_data/<name>
    """
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    if "/" in path_str or "\\" in path_str:
        return ROOT / p
    return ROOT / "chris_output" / "clean_data" / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


"""
Verified agent splitting conventions.

IMPORTANT: to keep conventions exactly aligned with `split_agents_model_scaffold.py`
and downstream analyses (e.g. auroc_model_scaffold.py), we import the splitter
from that script rather than duplicating logic here.
"""

from split_agents_model_scaffold import split_agent_name, _version_scaffold_for_agent  # type: ignore


# -----------------------------
# Pro model canonicalization
# -----------------------------

_PRO_TRAILING_SUFFIX_RE = re.compile(r"\s+(?:--|-)\s+.+$")  # strip " - paper", " - 10132025", " -- debug-oct22", ...


def canonicalize_pro_model(model_name: str) -> str:
    """
    Canonicalize SWE-bench Pro `subject_id` model strings so they align with Verified
    naming conventions where requested.
    """
    raw = (model_name or "").strip()
    if not raw:
        return raw

    # Remove paper/date/debug-ish suffixes (analysis convention).
    base = re.sub(_PRO_TRAILING_SUFFIX_RE, "", raw).strip()

    low = base.lower()

    # Merge "Claude Sonnet 4" and "Claude 4 Sonnet" -> Verified's "Claude 4 Sonnet".
    if low in {"claude 4 sonnet", "claude sonnet 4"}:
        return "Claude 4 Sonnet"

    # Merge Gemini 2.5 Pro Preview variants (paper/debug) -> a single label.
    if low == "gemini 2.5 pro preview":
        return "Gemini 2.5 Pro Preview"

    # Merge GPT-5 (base) with Verified.
    if low == "gpt-5" or low == "gpt 5":
        return "GPT-5"

    # Merge GPT-5 Codex naming variants.
    if low in {"gpt 5 codex", "gpt-5-codex", "gpt-5 codex", "gpt5-codex", "gpt5 codex"}:
        return "GPT-5 Codex"

    # Treat Pro "GPT OSS" as the 120B model (matches Verified/Terminal-Bench canon).
    if low == "gpt oss" or low == "gpt-oss" or low == "gptoss":
        return "GPT OSS 120B"

    # Merge GLM-4.5 variants (if present in Pro exports)
    if low in {"glm4-5", "glm-4.5", "glm-4-5"}:
        return "GLM-4.5"

    # Merge Claude 4.5 Sonnet with Verified's token (kept as-is in Verified results).
    # NOTE: user request mentions "Claude 4.5 Connet" (typo) — we handle Sonnet here.
    if low in {"claude 4.5 sonnet", "claude 4.5 connet"}:
        return "claude-sonnet-4-5"

    # Merge Kimi (paper) with Verified's "kimi_k2_instruct".
    if low == "kimi":
        return "kimi_k2_instruct"

    # Default: return the stripped base name.
    return base


# -----------------------------
# Data loading (multi-benchmark)
# -----------------------------

@dataclass(frozen=True)
class MultiBenchObs:
    model_idx: torch.Tensor
    scaffold_idx: torch.Tensor
    item_idx: torch.Tensor
    y: torch.Tensor
    model_ids: list[str]
    scaffold_ids: list[str]
    item_ids: list[str]
    verified_item_ids: set[str]
    pro_item_ids: set[str]
    terminal_bench_item_ids: set[str]
    agent_split_df: pd.DataFrame


def _agent_key(benchmark: str, agent: str) -> str:
    return f"{benchmark}::{agent}"


def load_multibench_split_irt_data(
    *,
    verified_path: Path,
    pro_path: Path,
    terminal_bench_path: Optional[Path] = None,
) -> MultiBenchObs:
    """
    Load Verified + Pro (+ optional Terminal-Bench) JSONL and share model/scaffold vocab across all.
    """
    agent_rows: list[dict] = []
    model_set: set[str] = set()
    scaffold_set: set[str] = set()
    item_set: set[str] = set()
    verified_item_ids: set[str] = set()
    pro_item_ids: set[str] = set()
    terminal_bench_item_ids: set[str] = set()

    # Pre-load Pro records so we can enforce "prefer paper" for specific models.
    pro_records = list(_iter_jsonl(pro_path))

    # For these canonical models, if a "paper" variant exists, drop the non-paper variants.
    prefer_paper_models = {"Claude 4 Sonnet", "Gemini 2.5 Pro Preview"}
    pro_is_paper = {str(r["subject_id"]): ("paper" in str(r["subject_id"]).lower()) for r in pro_records}
    pro_canon = {str(r["subject_id"]): canonicalize_pro_model(str(r["subject_id"])) for r in pro_records}
    have_paper_for_model = {
        m for subj, m in pro_canon.items() if (m in prefer_paper_models and pro_is_paper.get(subj, False))
    }

    filtered_pro_records: list[dict] = []
    for r in pro_records:
        subj = str(r["subject_id"])
        m = pro_canon[subj]
        if m in have_paper_for_model and m in prefer_paper_models and not pro_is_paper.get(subj, False):
            continue
        filtered_pro_records.append(r)

    # Pass 1: build vocabs and per-agent splits
    for r in _iter_jsonl(verified_path):
        agent = str(r["subject_id"])
        split = split_agent_name(agent)
        if split is None:
            continue
        model, scaffold, model_raw, scaffold_raw = split
        scaffold = _version_scaffold_for_agent(agent, scaffold)

        model_set.add(model)
        scaffold_set.add(scaffold)
        agent_rows.append(
            {
                "benchmark": "verified",
                "agent": agent,
                "model": model,
                "scaffold": scaffold,
                "model_raw": model_raw,
                "scaffold_raw": scaffold_raw,
            }
        )
        verified_item_ids.update(str(it) for it in r["responses"].keys())
        item_set.update(str(it) for it in r["responses"].keys())

    for r in filtered_pro_records:
        agent = str(r["subject_id"])
        model_raw = agent
        scaffold_raw = "SWE-agent 1.0"
        model = canonicalize_pro_model(agent)
        scaffold = "SWE-agent 1.0"

        model_set.add(model)
        scaffold_set.add(scaffold)
        agent_rows.append(
            {
                "benchmark": "pro",
                "agent": agent,
                "model": model,
                "scaffold": scaffold,
                "model_raw": model_raw,
                "scaffold_raw": scaffold_raw,
            }
        )
        pro_item_ids.update(str(it) for it in r["responses"].keys())
        item_set.update(str(it) for it in r["responses"].keys())

    if terminal_bench_path is not None:
        for r in _iter_jsonl(terminal_bench_path):
            agent = str(r["subject_id"])
            split = split_agent_name(agent)
            if split is None:
                continue
            model, scaffold, model_raw, scaffold_raw = split
            scaffold = _version_scaffold_for_agent(agent, scaffold)

            model_set.add(model)
            scaffold_set.add(scaffold)
            agent_rows.append(
                {
                    "benchmark": "terminal_bench",
                    "agent": agent,
                    "model": model,
                    "scaffold": scaffold,
                    "model_raw": model_raw,
                    "scaffold_raw": scaffold_raw,
                }
            )
            terminal_bench_item_ids.update(str(it) for it in r["responses"].keys())
            item_set.update(str(it) for it in r["responses"].keys())

    model_ids = sorted(model_set)
    scaffold_ids = sorted(scaffold_set)
    item_ids = sorted(item_set)

    model_to_idx = {m: i for i, m in enumerate(model_ids)}
    scaffold_to_idx = {s: i for i, s in enumerate(scaffold_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}

    # Map agent to (model_idx, scaffold_idx)
    agent_to_pair: dict[str, tuple[int, int]] = {}
    for row in agent_rows:
        agent_to_pair[_agent_key(row["benchmark"], row["agent"])] = (
            model_to_idx[row["model"]],
            scaffold_to_idx[row["scaffold"]],
        )

    # Pass 2: build observation arrays
    m_list: list[int] = []
    s_list: list[int] = []
    i_list: list[int] = []
    y_list: list[int] = []

    for r in _iter_jsonl(verified_path):
        benchmark = "verified"
        agent = str(r["subject_id"])
        pair = agent_to_pair.get(_agent_key(benchmark, agent))
        if pair is None:
            continue
        m_idx, s_idx = pair
        for item_id, y in r["responses"].items():
            it = str(item_id)
            m_list.append(m_idx)
            s_list.append(s_idx)
            i_list.append(item_to_idx[it])
            y_list.append(int(y))

    for r in filtered_pro_records:
        benchmark = "pro"
        agent = str(r["subject_id"])
        pair = agent_to_pair.get(_agent_key(benchmark, agent))
        if pair is None:
            continue
        m_idx, s_idx = pair
        for item_id, y in r["responses"].items():
            it = str(item_id)
            m_list.append(m_idx)
            s_list.append(s_idx)
            i_list.append(item_to_idx[it])
            y_list.append(int(y))

    if terminal_bench_path is not None:
        for r in _iter_jsonl(terminal_bench_path):
            benchmark = "terminal_bench"
            agent = str(r["subject_id"])
            pair = agent_to_pair.get(_agent_key(benchmark, agent))
            if pair is None:
                continue
            m_idx, s_idx = pair
            for item_id, y in r["responses"].items():
                it = str(item_id)
                m_list.append(m_idx)
                s_list.append(s_idx)
                i_list.append(item_to_idx[it])
                y_list.append(int(y))

    agent_split_df = pd.DataFrame(agent_rows).sort_values(["benchmark", "model", "scaffold", "agent"])

    return MultiBenchObs(
        model_idx=torch.tensor(m_list, dtype=torch.long),
        scaffold_idx=torch.tensor(s_list, dtype=torch.long),
        item_idx=torch.tensor(i_list, dtype=torch.long),
        y=torch.tensor(y_list, dtype=torch.float),
        model_ids=model_ids,
        scaffold_ids=scaffold_ids,
        item_ids=item_ids,
        verified_item_ids=verified_item_ids,
        pro_item_ids=pro_item_ids,
        terminal_bench_item_ids=terminal_bench_item_ids,
        agent_split_df=agent_split_df,
    )


# -----------------------------
# Pyro models (same as train_model_scaffold.py)
# -----------------------------

class ModelScaffold1PL:
    """1D Rasch (1PL) with theta_model + theta_scaffold."""

    def __init__(self, num_models: int, num_scaffolds: int, num_items: int):
        self.num_models = num_models
        self.num_scaffolds = num_scaffolds
        self.num_items = num_items

    def model(self, m_idx, s_idx, items, y):
        sigma_theta_m = pyro.sample("sigma_theta_model", dist.HalfNormal(1.0))
        sigma_theta_s = pyro.sample("sigma_theta_scaffold", dist.HalfNormal(1.0))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(1.0))

        with pyro.plate("models", self.num_models):
            theta_m_raw = pyro.sample("theta_model_raw", dist.Normal(0.0, sigma_theta_m))
        with pyro.plate("scaffolds", self.num_scaffolds):
            theta_s_raw = pyro.sample("theta_scaffold_raw", dist.Normal(0.0, sigma_theta_s))
        with pyro.plate("items", self.num_items):
            b = pyro.sample("b", dist.Normal(0.0, sigma_b))

        theta_m = theta_m_raw - theta_m_raw.mean()
        theta_s = theta_s_raw - theta_s_raw.mean()

        with pyro.plate("obs", y.size(0)):
            logits = (theta_m[m_idx] + theta_s[s_idx]) - b[items]
            pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    def guide(self, m_idx, s_idx, items, y):
        sigma_theta_m_q = pyro.param("sigma_theta_model_q", torch.tensor(1.0), constraint=constraints.positive)
        sigma_theta_s_q = pyro.param(
            "sigma_theta_scaffold_q", torch.tensor(1.0), constraint=constraints.positive
        )
        sigma_b_q = pyro.param("sigma_b_q", torch.tensor(1.0), constraint=constraints.positive)

        pyro.sample("sigma_theta_model", dist.Delta(sigma_theta_m_q))
        pyro.sample("sigma_theta_scaffold", dist.Delta(sigma_theta_s_q))
        pyro.sample("sigma_b", dist.Delta(sigma_b_q))

        loc_theta_m = pyro.param("loc_theta_model_raw", torch.zeros(self.num_models))
        scale_theta_m = pyro.param(
            "scale_theta_model_raw",
            torch.ones(self.num_models),
            constraint=constraints.positive,
        )
        loc_theta_s = pyro.param("loc_theta_scaffold_raw", torch.zeros(self.num_scaffolds))
        scale_theta_s = pyro.param(
            "scale_theta_scaffold_raw",
            torch.ones(self.num_scaffolds),
            constraint=constraints.positive,
        )
        loc_b = pyro.param("loc_b", torch.zeros(self.num_items))
        scale_b = pyro.param("scale_b", torch.ones(self.num_items), constraint=constraints.positive)

        with pyro.plate("models", self.num_models):
            pyro.sample("theta_model_raw", dist.Normal(loc_theta_m, scale_theta_m))
        with pyro.plate("scaffolds", self.num_scaffolds):
            pyro.sample("theta_scaffold_raw", dist.Normal(loc_theta_s, scale_theta_s))
        with pyro.plate("items", self.num_items):
            pyro.sample("b", dist.Normal(loc_b, scale_b))


class ModelScaffold2PL:
    """1D 2PL with theta_model + theta_scaffold and positive discrimination."""

    def __init__(self, num_models: int, num_scaffolds: int, num_items: int):
        self.num_models = num_models
        self.num_scaffolds = num_scaffolds
        self.num_items = num_items

    def model(self, m_idx, s_idx, items, y):
        sigma_theta_m = pyro.sample("sigma_theta_model", dist.HalfNormal(1.0))
        sigma_theta_s = pyro.sample("sigma_theta_scaffold", dist.HalfNormal(1.0))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(1.0))
        mu_log_a = pyro.sample("mu_log_a", dist.Normal(0.0, 1.0))
        sigma_log_a = pyro.sample("sigma_log_a", dist.HalfNormal(1.0))

        with pyro.plate("models", self.num_models):
            theta_m_raw = pyro.sample("theta_model_raw", dist.Normal(0.0, sigma_theta_m))
        with pyro.plate("scaffolds", self.num_scaffolds):
            theta_s_raw = pyro.sample("theta_scaffold_raw", dist.Normal(0.0, sigma_theta_s))
        with pyro.plate("items", self.num_items):
            b = pyro.sample("b", dist.Normal(0.0, sigma_b))
            a = pyro.sample("a", dist.LogNormal(mu_log_a, sigma_log_a))

        theta_m = theta_m_raw - theta_m_raw.mean()
        theta_s = theta_s_raw - theta_s_raw.mean()

        with pyro.plate("obs", y.size(0)):
            logits = a[items] * ((theta_m[m_idx] + theta_s[s_idx]) - b[items])
            pyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    def guide(self, m_idx, s_idx, items, y):
        sigma_theta_m_q = pyro.param("sigma_theta_model_q", torch.tensor(1.0), constraint=constraints.positive)
        sigma_theta_s_q = pyro.param(
            "sigma_theta_scaffold_q", torch.tensor(1.0), constraint=constraints.positive
        )
        sigma_b_q = pyro.param("sigma_b_q", torch.tensor(1.0), constraint=constraints.positive)
        loc_mu_log_a = pyro.param("loc_mu_log_a", torch.tensor(0.0))
        scale_mu_log_a = pyro.param("scale_mu_log_a", torch.tensor(1.0), constraint=constraints.positive)
        sigma_log_a_q = pyro.param("sigma_log_a_q", torch.tensor(1.0), constraint=constraints.positive)

        pyro.sample("sigma_theta_model", dist.Delta(sigma_theta_m_q))
        pyro.sample("sigma_theta_scaffold", dist.Delta(sigma_theta_s_q))
        pyro.sample("sigma_b", dist.Delta(sigma_b_q))
        pyro.sample("mu_log_a", dist.Normal(loc_mu_log_a, scale_mu_log_a))
        pyro.sample("sigma_log_a", dist.Delta(sigma_log_a_q))

        loc_theta_m = pyro.param("loc_theta_model_raw", torch.zeros(self.num_models))
        scale_theta_m = pyro.param(
            "scale_theta_model_raw",
            torch.ones(self.num_models),
            constraint=constraints.positive,
        )
        loc_theta_s = pyro.param("loc_theta_scaffold_raw", torch.zeros(self.num_scaffolds))
        scale_theta_s = pyro.param(
            "scale_theta_scaffold_raw",
            torch.ones(self.num_scaffolds),
            constraint=constraints.positive,
        )
        loc_b = pyro.param("loc_b", torch.zeros(self.num_items))
        scale_b = pyro.param("scale_b", torch.ones(self.num_items), constraint=constraints.positive)

        loc_a = pyro.param("loc_log_a", torch.zeros(self.num_items))
        scale_a = pyro.param("scale_log_a", torch.ones(self.num_items), constraint=constraints.positive)

        with pyro.plate("models", self.num_models):
            pyro.sample("theta_model_raw", dist.Normal(loc_theta_m, scale_theta_m))
        with pyro.plate("scaffolds", self.num_scaffolds):
            pyro.sample("theta_scaffold_raw", dist.Normal(loc_theta_s, scale_theta_s))
        with pyro.plate("items", self.num_items):
            pyro.sample("b", dist.Normal(loc_b, scale_b))
            pyro.sample("a", dist.LogNormal(loc_a, scale_a))


def train_svi(model_fn, guide_fn, obs: MultiBenchObs, epochs: int, lr: float = 0.01) -> list[float]:
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    optimizer = ClippedAdam({"lr": lr, "betas": (0.9, 0.999), "clip_norm": 5.0})
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())

    losses: list[float] = []
    for ep in range(1, epochs + 1):
        loss = float(svi.step(obs.model_idx, obs.scaffold_idx, obs.item_idx, obs.y))
        losses.append(loss)
        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"epoch {ep:5d}  loss={loss:,.2f}")
    return losses


def _centered_loc(loc_raw: torch.Tensor) -> torch.Tensor:
    return loc_raw - loc_raw.mean()


def save_outputs(*, out_dir: Path, obs: MultiBenchObs, model_type: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    b_loc = pyro.param("loc_b").detach().cpu().numpy()
    b_scale = pyro.param("scale_b").detach().cpu().numpy()

    items_df = pd.DataFrame({"b": b_loc, "b_std": b_scale}, index=obs.item_ids)
    if model_type == "2pl":
        a_loc = pyro.param("loc_log_a").detach().cpu().numpy()
        a_scale = pyro.param("scale_log_a").detach().cpu().numpy()
        items_df["log_a"] = a_loc
        items_df["log_a_std"] = a_scale
        items_df["a_mean"] = np.exp(a_loc + 0.5 * (a_scale**2))
    items_df.to_csv(out_dir / "items.csv")

    # Also write per-benchmark item views.
    def _write_items_subset(item_ids: set[str], out_name: str) -> None:
        if not item_ids:
            return
        idx = [iid for iid in items_df.index if iid in item_ids]
        if not idx:
            return
        items_df.loc[idx].to_csv(out_dir / out_name)

    _write_items_subset(obs.verified_item_ids, "items_verified.csv")
    _write_items_subset(obs.pro_item_ids, "items_pro.csv")
    _write_items_subset(obs.terminal_bench_item_ids, "items_terminal_bench.csv")

    theta_m_loc_raw = pyro.param("loc_theta_model_raw").detach().cpu()
    theta_m_scale = pyro.param("scale_theta_model_raw").detach().cpu().numpy()
    theta_m_loc = _centered_loc(theta_m_loc_raw).numpy()
    model_df = pd.DataFrame({"theta": theta_m_loc, "theta_std": theta_m_scale}, index=obs.model_ids).sort_values(
        "theta", ascending=False
    )
    model_df.to_csv(out_dir / "model_abilities.csv")

    theta_s_loc_raw = pyro.param("loc_theta_scaffold_raw").detach().cpu()
    theta_s_scale = pyro.param("scale_theta_scaffold_raw").detach().cpu().numpy()
    theta_s_loc = _centered_loc(theta_s_loc_raw).numpy()
    scaffold_df = pd.DataFrame(
        {"theta": theta_s_loc, "theta_std": theta_s_scale}, index=obs.scaffold_ids
    ).sort_values("theta", ascending=False)
    scaffold_df.to_csv(out_dir / "scaffold_abilities.csv")
    # NOTE: Intentionally do not write agent_splits.csv (contains per-agent metadata that
    # isn't needed for downstream analyses and can be large/noisy).


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shared model+scaffold IRT across multiple benchmarks")
    parser.add_argument(
        "--verified_path",
        type=str,
        default="out/chris_irt/swebench_verified_20251115_full.jsonl",
        help="Path to Verified agent×task JSONL (subject_id, responses)",
    )
    parser.add_argument(
        "--pro_path",
        type=str,
        default="out/chris_irt/swebench_pro.jsonl",
        help="Path to Pro agent×task JSONL (subject_id, responses)",
    )
    parser.add_argument(
        "--terminal_bench_path",
        type=str,
        default="out/chris_irt/terminal_bench_2.0.jsonl",
        help="Path to Terminal-Bench 2.0 agent×task JSONL (subject_id, responses)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clean_data/training_results_shared_verified_pro",
        help="Directory to save results to",
    )
    parser.add_argument("--epochs", type=int, default=5000, help="SVI epochs")
    parser.add_argument(
        "--model",
        type=str,
        default="2pl",
        choices=["1pl", "2pl"],
        help="IRT model type (1pl=Rasch, 2pl=discrimination+difficulty)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for SVI (default: 0.01)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
        set_seed(args.seed)

    verified_path = resolve_path(args.verified_path)
    pro_path = resolve_path(args.pro_path)
    terminal_bench_path = resolve_path(args.terminal_bench_path) if args.terminal_bench_path else None
    out_root = resolve_output_dir(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading Verified from: {verified_path}")
    print(f"Loading Pro from:       {pro_path}")
    if terminal_bench_path is not None:
        print(f"Loading Terminal-Bench: {terminal_bench_path}")
    obs = load_multibench_split_irt_data(
        verified_path=verified_path, pro_path=pro_path, terminal_bench_path=terminal_bench_path
    )

    # Quick breakdown
    by_bench = obs.agent_split_df.groupby("benchmark")["agent"].nunique().to_dict()
    print(
        "Included subjects:",
        len(obs.agent_split_df),
        f"(by benchmark: {by_bench})  models={len(obs.model_ids)} scaffolds={len(obs.scaffold_ids)} items={len(obs.item_ids)}",
    )
    print(f"Observations: {obs.y.numel():,}")

    pyro.clear_param_store()

    if args.model == "1pl":
        model_obj = ModelScaffold1PL(len(obs.model_ids), len(obs.scaffold_ids), len(obs.item_ids))
        subdir = "1d_1pl"
    else:
        model_obj = ModelScaffold2PL(len(obs.model_ids), len(obs.scaffold_ids), len(obs.item_ids))
        subdir = "1d_2pl"

    out_dir = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training {args.model.upper()} model... output -> {out_dir}")

    _ = train_svi(model_obj.model, model_obj.guide, obs, epochs=args.epochs, lr=args.lr)
    save_outputs(out_dir=out_dir, obs=obs, model_type=args.model)
    print("Done.")


if __name__ == "__main__":
    main()

