"""Per-cell multi-benchmark 1PL IRT training for subset extrapolation.

For each (target_dataset, observed_task_subset) cell, this module:

  1. Loads the FULL response matrices for the three non-target benchmarks
     (cached at module scope — these don't change across cells).
  2. Combines them with the target benchmark's responses restricted to
     `observed_tasks`.
  3. Trains a shared model+scaffold 1PL IRT via the existing infrastructure
     (`train_irt_model_scaffold_1pl` from experiment_agent_features).
  4. Caches `(theta_by_model, theta_by_scaffold, diff_by_item)` and the
     per-benchmark training item list under the provided cache directory so
     re-runs skip Pyro entirely.

Reuses (does not duplicate):
  - `train_irt_model_scaffold_1pl` and `build_multibench_obs_from_tagged_responses`
    from experiment_agent_features/predict_question_difficulty_multi_benchmark.py
  - `load_multibench_split_irt_data` from swebench_irt/train_model_scaffold_shared.py
  - Per-benchmark response loaders from experiment_agent_features.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Map subset-extrapolation dataset name -> multi-benchmark canonical name.
DATASET_TO_BENCH: Dict[str, str] = {
    "swebench_verified": "verified",
    "swebench_pro": "pro",
    "terminalbench": "terminal_bench",
    "gso": "gso",
}
ALL_BENCHES: Tuple[str, ...] = ("verified", "pro", "terminal_bench", "gso")

ROOT = Path(__file__).resolve().parents[1]

# Default responses.jsonl per benchmark. We keep this here (rather than
# rebuilding through ExperimentAConfig) so that loading is independent of the
# target dataset and can be cached once per process.
RESPONSES_PATHS: Dict[str, Path] = {
    "verified": ROOT / "data" / "swebench_verified" / "responses.jsonl",
    "pro": ROOT / "data" / "swebench_pro" / "responses.jsonl",
    "terminal_bench": ROOT / "data" / "terminalbench" / "responses.jsonl",
    "gso": ROOT / "data" / "gso" / "responses.jsonl",
}


@dataclass
class MultiBenchIRT:
    """Result of one fold's multi-benchmark IRT training.

    Field semantics:
      - theta_by_model: {model_name: theta_model}, one entry per unique model
        across all 4 benchmarks. Models the IRT didn't see (because all agents
        carrying that model were filtered out) will be absent.
      - theta_by_scaffold: same structure for scaffolds.
      - diff_by_item: {task_id: b}, one entry per item in the IRT training
        set. Crucially this does NOT include held-out target tasks — those
        difficulties are predicted separately by the held-out-difficulty
        Ridge.
      - training_item_ids_by_bench: which item IDs appeared in IRT training
        per benchmark. Used by the held-out Ridge to assemble the
        cross-benchmark training pool with the right per-benchmark routing
        for embeddings + judge features.
      - agent_to_ms: {(benchmark, agent_id): (model, scaffold)} for every
        agent the IRT learned a position for. Used by the scoring step to
        look up θ_agent = θ_model + θ_scaffold for target-benchmark agents.
    """

    theta_by_model: Dict[str, float]
    theta_by_scaffold: Dict[str, float]
    diff_by_item: Dict[str, float]
    training_item_ids_by_bench: Dict[str, List[str]]
    agent_to_ms: Dict[Tuple[str, str], Tuple[str, str]]


# ---------------------------------------------------------------------------
# Process-scoped lazy caches.
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_ALL_RESPONSES_TAGGED: Optional[List[Tuple[str, str, Dict[str, int]]]] = None
_AGENT_TO_MS_PAIR_FLAT: Optional[Dict[str, Tuple[str, str]]] = None  # keyed "bench::agent"
_AGENT_TO_MS_PAIR_NESTED: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None
_ITEM_IDS_BY_BENCH: Optional[Dict[str, Set[str]]] = None
_AGENT_SPLIT_DF = None  # cached pd.DataFrame from load_multibench_split_irt_data


def _load_all_benches_once() -> None:
    """Populate process-scope caches by reading all 4 response files once."""
    global _ALL_RESPONSES_TAGGED, _AGENT_TO_MS_PAIR_FLAT, _AGENT_TO_MS_PAIR_NESTED
    global _ITEM_IDS_BY_BENCH, _AGENT_SPLIT_DF
    with _LOCK:
        if _ALL_RESPONSES_TAGGED is not None:
            return

        # Lazy imports — these pull in torch/pyro, which we want to defer to
        # worker processes (not the parent CLI).
        from experiment_agent_features import predict_question_difficulty as base
        from experiment_agent_features.predict_question_difficulty_multi_benchmark import (
            load_all_responses_terminal,
            load_all_responses_generic,
        )

        import sys
        sys.path.insert(0, str(ROOT))
        from swebench_irt.train_model_scaffold_shared import load_multibench_split_irt_data

        # 1. Load per-benchmark responses tagged with their benchmark.
        all_tagged: List[Tuple[str, str, Dict[str, int]]] = []
        item_ids_by_bench: Dict[str, Set[str]] = {}

        for bench in ALL_BENCHES:
            pth = RESPONSES_PATHS[bench]
            if not pth.exists():
                raise FileNotFoundError(
                    f"Multi-benchmark IRT needs {bench} responses at {pth}, but file is missing."
                )
            if bench == "terminal_bench":
                rows = load_all_responses_terminal(str(pth))
            elif bench == "gso":
                rows = load_all_responses_generic(path=str(pth), normalize_item_ids=True)
            else:
                rows = base.load_all_responses(str(pth))
            items: Set[str] = set()
            for sid, resp in rows:
                all_tagged.append((bench, str(sid), resp))
                items.update(str(t) for t in resp.keys())
            item_ids_by_bench[bench] = items

        # 2. Load shared agent_split_df via the canonical IRT loader; this
        # parses every agent into (model, scaffold) using the same logic as
        # the rest of the codebase (split_agent_name + per-benchmark
        # overrides for Pro / GSO / Terminal-Bench).
        obs_full = load_multibench_split_irt_data(
            verified_path=RESPONSES_PATHS["verified"],
            pro_path=RESPONSES_PATHS["pro"],
            terminal_bench_path=RESPONSES_PATHS["terminal_bench"],
            gso_path=RESPONSES_PATHS["gso"],
        )

        flat: Dict[str, Tuple[str, str]] = {}
        nested: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for row in obs_full.agent_split_df.to_dict(orient="records"):
            bench = str(row.get("benchmark", "") or "").strip()
            agent = str(row.get("agent", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            scaffold = str(row.get("scaffold", "") or "").strip()
            if bench and agent and model and scaffold:
                flat[f"{bench}::{agent}"] = (model, scaffold)
                nested[(bench, agent)] = (model, scaffold)

        _ALL_RESPONSES_TAGGED = all_tagged
        _AGENT_TO_MS_PAIR_FLAT = flat
        _AGENT_TO_MS_PAIR_NESTED = nested
        _ITEM_IDS_BY_BENCH = item_ids_by_bench
        _AGENT_SPLIT_DF = obs_full.agent_split_df


def get_full_item_ids(bench: str) -> Set[str]:
    """Return every item ID seen on `bench` in the canonical responses file."""
    _load_all_benches_once()
    if bench not in (_ITEM_IDS_BY_BENCH or {}):
        raise ValueError(f"Unknown benchmark: {bench!r}")
    return set(_ITEM_IDS_BY_BENCH[bench])  # type: ignore[index]


def get_agent_to_ms() -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Return the canonical {(bench, agent): (model, scaffold)} map."""
    _load_all_benches_once()
    return dict(_AGENT_TO_MS_PAIR_NESTED or {})


def parseable_agents_for(target_dataset: str) -> set:
    """Return the set of target-benchmark agent IDs that decompose into
    (model, scaffold) via `split_agents_model_scaffold` (with per-benchmark
    overrides for Pro / GSO / Terminal-Bench).

    The subset extrapolation pipeline restricts every method (empirical,
    combined_calibrated, oracle) to this set so the comparison is over a
    single well-defined agent population per dataset.
    """
    if target_dataset not in DATASET_TO_BENCH:
        raise ValueError(f"Unknown target_dataset {target_dataset!r}")
    bench = DATASET_TO_BENCH[target_dataset]
    ms = get_agent_to_ms()
    return {a for (b, a) in ms.keys() if b == bench}


def _cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "model_abilities": cache_dir / "model_abilities.csv",
        "scaffold_abilities": cache_dir / "scaffold_abilities.csv",
        "items": cache_dir / "items.csv",
        "training_items": cache_dir / "training_items_by_bench.json",
    }


def _try_load_cached(cache_dir: Path) -> Optional[MultiBenchIRT]:
    import csv

    paths = _cache_paths(cache_dir)
    if not all(p.exists() for p in paths.values()):
        return None

    theta_by_model: Dict[str, float] = {}
    with paths["model_abilities"].open() as f:
        for row in csv.DictReader(f):
            theta_by_model[str(row["model_id"])] = float(row["theta"])

    theta_by_scaffold: Dict[str, float] = {}
    with paths["scaffold_abilities"].open() as f:
        for row in csv.DictReader(f):
            theta_by_scaffold[str(row["scaffold_id"])] = float(row["theta"])

    diff_by_item: Dict[str, float] = {}
    with paths["items"].open() as f:
        for row in csv.DictReader(f):
            diff_by_item[str(row["item_id"])] = float(row["b"])

    with paths["training_items"].open() as f:
        training_item_ids_by_bench = {k: list(v) for k, v in json.load(f).items()}

    return MultiBenchIRT(
        theta_by_model=theta_by_model,
        theta_by_scaffold=theta_by_scaffold,
        diff_by_item=diff_by_item,
        training_item_ids_by_bench=training_item_ids_by_bench,
        agent_to_ms=get_agent_to_ms(),
    )


def train_fold(
    *,
    target_dataset: str,
    observed_target_tasks: List[str],
    seed: int,
    cache_dir: Path,
    epochs: int = 5000,
    lr: float = 0.01,
    device: str = "cpu",
) -> MultiBenchIRT:
    """Train (or load cached) multi-benchmark IRT for one (dataset, size, seed) cell.

    The training pool is the union of:
      * full responses on the 3 non-target benchmarks, and
      * responses on `observed_target_tasks` from the target benchmark.
    """
    if target_dataset not in DATASET_TO_BENCH:
        raise ValueError(
            f"Unknown target_dataset {target_dataset!r}; valid: {list(DATASET_TO_BENCH)}"
        )

    cache_dir = Path(cache_dir)
    cached = _try_load_cached(cache_dir)
    if cached is not None:
        return cached

    _load_all_benches_once()
    assert _ALL_RESPONSES_TAGGED is not None
    assert _AGENT_TO_MS_PAIR_FLAT is not None
    assert _AGENT_SPLIT_DF is not None

    target_bench = DATASET_TO_BENCH[target_dataset]
    non_target_benches = [b for b in ALL_BENCHES if b != target_bench]

    # keep_item_ids: full items from non-target benchmarks + observed subset on
    # target benchmark. build_multibench_obs_from_tagged_responses filters per
    # observation, so the IRT only sees obs on these items.
    keep_items: Set[str] = set()
    for b in non_target_benches:
        keep_items.update(get_full_item_ids(b))
    obs_target_set = {str(t) for t in observed_target_tasks}
    if not obs_target_set:
        raise ValueError("observed_target_tasks was empty")
    keep_items.update(obs_target_set)

    # Lazy imports so workers (not the parent CLI) pay the torch/pyro cost.
    from experiment_agent_features.predict_question_difficulty_multi_benchmark import (
        build_multibench_obs_from_tagged_responses,
        train_irt_model_scaffold_1pl,
    )

    obs = build_multibench_obs_from_tagged_responses(
        all_responses_tagged=_ALL_RESPONSES_TAGGED,
        agent_to_ms_pair=_AGENT_TO_MS_PAIR_FLAT,
        obs_full_agent_split_df=_AGENT_SPLIT_DF,
        keep_item_ids=keep_items,
    )

    # Sanity: every observed target task must end up in the IRT's item set.
    # If not, the target subset isn't in the per-benchmark response file (data
    # integrity bug). Fail loudly per repo convention.
    irt_items = set(obs.item_ids)
    missing = obs_target_set - irt_items
    if missing:
        raise RuntimeError(
            f"{len(missing)} observed target tasks for {target_dataset} have zero "
            f"responses in {RESPONSES_PATHS[target_bench]} (first few: "
            f"{sorted(missing)[:5]}). Did you pass tasks that aren't in the response matrix?"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
        obs_train=obs,
        irt_model="1d_1pl",
        epochs=int(epochs),
        device=str(device),
        seed=int(seed),
        lr=float(lr),
        out_dir=str(cache_dir),
    )

    # Persist the per-benchmark item list so the held-out Ridge can route
    # each training item back to the right embedding / judge source.
    training_items_by_bench = {
        "verified": sorted(obs.verified_item_ids),
        "pro": sorted(obs.pro_item_ids),
        "terminal_bench": sorted(obs.terminal_bench_item_ids),
        "gso": sorted(getattr(obs, "gso_item_ids", set())),
    }
    with _cache_paths(cache_dir)["training_items"].open("w") as f:
        json.dump(training_items_by_bench, f)

    return MultiBenchIRT(
        theta_by_model=theta_by_model,
        theta_by_scaffold=theta_by_scaffold,
        diff_by_item=diff_by_item,
        training_item_ids_by_bench=training_items_by_bench,
        agent_to_ms=get_agent_to_ms(),
    )
