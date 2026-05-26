"""Fold-specific IRT training for the new-responses experiment."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import pandas as pd

from experiment_new_tasks.train_irt_split import set_torch_determinism
from swebench_irt.model_scaffold_combine import (
    attach_theta_combine_weights_from_path,
    normalize_theta_combine,
    theta_combine_cache_suffix,
)
from utils.irt_training import (
    as_frame as _as_frame,
    cache_meta_matches as _cache_meta_matches,
    responses_signature as _responses_signature,
)


def _make_observation_key(benchmark: str, subject_id: str, task_id: str) -> str:
    return f"{benchmark}::{subject_id}::{task_id}"


def get_split_cache_dir(
    output_base: Path,
    split_seed: int,
    fold_idx: int,
    k_folds: int,
    theta_combine: str = "sum",
) -> Path:
    split_name = (
        f"seed{split_seed}_fold{fold_idx}of{k_folds}_1d_1pl"
        f"{theta_combine_cache_suffix(theta_combine)}"
    )
    return output_base / split_name


def _observations_signature(observation_keys: Set[str]) -> str:
    payload = json.dumps(sorted(str(key) for key in observation_keys), separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cached_standard_irt(
    output_dir: Path,
    expected_meta: Dict[str, object],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    meta_path = output_dir / "cache_meta.json"
    abilities_path = output_dir / "abilities.csv"
    items_path = output_dir / "items.csv"
    if not (abilities_path.exists() and items_path.exists() and meta_path.exists()):
        return None

    try:
        with open(meta_path, "r") as f:
            cached_meta = json.load(f)
    except Exception:
        return None
    if cached_meta != expected_meta:
        return None

    abilities = pd.read_csv(abilities_path)
    if "subject_id" not in abilities.columns or "theta" not in abilities.columns:
        return None
    abilities = abilities.rename(columns={"theta": "ability"}).set_index("subject_id")

    items = pd.read_csv(items_path)
    if "item_id" not in items.columns or "b" not in items.columns:
        return None
    items = items.set_index("item_id")

    print(f"Found cached standard IRT model at {output_dir}")
    return abilities.sort_index(), items.sort_index()


def _model_scaffold_cache_meta(
    obs_train,
    *,
    irt_model: str,
    theta_combine: str,
    epochs: int,
    seed: int,
    lr: float,
    train_observations: Set[str],
) -> Dict[str, object]:
    obs_signature = hashlib.md5()
    for tensor in [obs_train.model_idx, obs_train.scaffold_idx, obs_train.item_idx, obs_train.y]:
        obs_signature.update(tensor.detach().cpu().numpy().tobytes())
    combine_norm = normalize_theta_combine(theta_combine)
    meta = {
        "cache_kind": "observation_split_model_scaffold_irt",
        "irt_model": str(irt_model),
        "epochs": int(epochs),
        "seed": int(seed),
        "lr": float(lr),
        "model_ids": list(obs_train.model_ids),
        "scaffold_ids": list(obs_train.scaffold_ids),
        "item_ids": list(obs_train.item_ids),
        "n_obs": int(obs_train.y.numel()),
        "obs_signature": obs_signature.hexdigest(),
        "train_observations_signature": _observations_signature(train_observations),
        "n_train_observations": int(len(train_observations)),
    }
    if combine_norm != "sum":
        meta["combine_theta"] = combine_norm
    return meta


def _load_cached_model_scaffold_irt(
    cache_dir: Path,
    expected_meta: Dict[str, object],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    meta_path = cache_dir / "cache_meta.json"
    model_path = cache_dir / "model_abilities.csv"
    scaffold_path = cache_dir / "scaffold_abilities.csv"
    items_path = cache_dir / "items.csv"
    if not (model_path.exists() and scaffold_path.exists() and items_path.exists() and meta_path.exists()):
        return None

    try:
        with open(meta_path, "r") as f:
            cached_meta = json.load(f)
    except Exception:
        return None
    if not _cache_meta_matches(cached_meta, expected_meta):
        return None

    model_df = pd.read_csv(model_path)
    scaffold_df = pd.read_csv(scaffold_path)
    item_df = pd.read_csv(items_path)
    if "model_id" not in model_df.columns or "theta" not in model_df.columns:
        return None
    if "scaffold_id" not in scaffold_df.columns or "theta" not in scaffold_df.columns:
        return None
    if "item_id" not in item_df.columns or "b" not in item_df.columns:
        return None

    model_df = model_df.rename(columns={"model_id": "model"}).set_index("model").sort_index()
    scaffold_df = scaffold_df.rename(columns={"scaffold_id": "scaffold"}).set_index("scaffold").sort_index()
    item_df = item_df.set_index("item_id").sort_index()
    attach_theta_combine_weights_from_path(
        model_df,
        scaffold_df,
        cache_dir,
        combine=cached_meta.get("combine_theta", "sum"),
    )

    print(f"Found cached model+scaffold IRT model at {cache_dir}")
    return model_df, scaffold_df, item_df


def get_or_train_model_scaffold_observation_split_irt(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    agent_to_ms_pair: Dict[str, Tuple[str, str]],
    train_observations: Set[str],
    all_item_ids: Set[str],
    output_base: Path,
    split_seed: int,
    fold_idx: int,
    k_folds: int,
    irt_model: str,
    theta_combine: str,
    epochs: int,
    device: str,
    lr: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/load model+scaffold IRT on non-held-out observations."""

    from utils.irt_training import (
        build_multibench_obs_from_tagged_responses,
        train_irt_model_scaffold_1pl,
    )

    if not train_observations:
        raise ValueError("train_observations was empty")

    theta_combine_norm = normalize_theta_combine(theta_combine)
    cache_dir = get_split_cache_dir(
        output_base,
        split_seed,
        fold_idx,
        k_folds,
        theta_combine_norm,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_obs_set = set(str(key) for key in train_observations)

    def keep_obs(benchmark: str, subject_id: str, item_id: str) -> bool:
        return _make_observation_key(benchmark, subject_id, item_id) in train_obs_set

    obs_train = build_multibench_obs_from_tagged_responses(
        all_responses_tagged=all_responses_tagged,
        agent_to_ms_pair=agent_to_ms_pair,
        obs_full_agent_split_df=None,
        keep_item_ids=set(all_item_ids),
        keep_obs_fn=keep_obs,
    )
    expected_meta = _model_scaffold_cache_meta(
        obs_train,
        irt_model=irt_model,
        theta_combine=theta_combine_norm,
        epochs=epochs,
        seed=split_seed,
        lr=lr,
        train_observations=train_observations,
    )
    cached = _load_cached_model_scaffold_irt(cache_dir, expected_meta)
    if cached is not None:
        return cached

    set_torch_determinism(False)
    try:
        theta_by_model, theta_by_scaffold, diff_by_item = train_irt_model_scaffold_1pl(
            obs_train=obs_train,
            irt_model=str(irt_model),
            theta_combine=theta_combine_norm,
            epochs=int(epochs),
            device=str(device),
            seed=int(split_seed),
            lr=float(lr),
            out_dir=str(cache_dir),
        )
    finally:
        set_torch_determinism(True)

    model_df = _as_frame(theta_by_model, "model", "theta")
    scaffold_df = _as_frame(theta_by_scaffold, "scaffold", "theta")
    item_df = _as_frame(diff_by_item, "item_id", "b")
    attach_theta_combine_weights_from_path(
        model_df,
        scaffold_df,
        cache_dir,
        combine=theta_combine_norm,
    )
    with open(cache_dir / "cache_meta.json", "w") as f:
        json.dump(expected_meta, f, indent=2, sort_keys=True)
    return model_df, scaffold_df, item_df


def get_or_train_observation_split_irt(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    train_observations: Set[str],
    all_item_ids: Set[str],
    output_base: Path,
    split_seed: int,
    fold_idx: int,
    k_folds: int,
    epochs: int,
    device: str,
    max_train_attempts: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/load standard agent/item IRT on non-held-out observations."""

    from utils.irt_training import (
        train_standard_irt_1pl_agents,
    )

    if not train_observations:
        raise ValueError("train_observations was empty")
    if max_train_attempts < 1:
        raise ValueError(f"max_train_attempts must be >= 1, got {max_train_attempts}")

    cache_dir = get_split_cache_dir(output_base, split_seed, fold_idx, k_folds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_meta = {
        "cache_kind": "observation_split_standard_agent_1pl",
        "epochs": int(epochs),
        "seed": int(split_seed),
        "item_ids": sorted(str(item_id) for item_id in all_item_ids),
        "responses_signature": _responses_signature(all_responses_tagged, all_item_ids),
        "train_observations_signature": _observations_signature(train_observations),
        "n_train_observations": int(len(train_observations)),
    }
    cached = _load_cached_standard_irt(cache_dir, expected_meta)
    if cached is not None:
        return cached

    train_obs_set = set(str(key) for key in train_observations)

    def keep_obs(benchmark: str, subject_id: str, item_id: str) -> bool:
        return _make_observation_key(benchmark, subject_id, item_id) in train_obs_set

    last_error: Optional[Exception] = None
    theta_by_agent: Dict[str, float] = {}
    diff_by_item: Dict[str, float] = {}
    for attempt_idx in range(max_train_attempts):
        attempt_number = attempt_idx + 1
        attempt_seed = int(split_seed) + attempt_idx
        if max_train_attempts > 1:
            print(
                f"Training standard split IRT attempt "
                f"{attempt_number}/{max_train_attempts} (seed={attempt_seed})"
            )
        set_torch_determinism(False)
        try:
            theta_by_agent, diff_by_item = train_standard_irt_1pl_agents(
                all_responses_tagged=all_responses_tagged,
                keep_item_ids=set(all_item_ids),
                epochs=int(epochs),
                device=str(device),
                seed=attempt_seed,
                out_dir=str(cache_dir),
                keep_obs_fn=keep_obs,
            )
            last_error = None
        except Exception as exc:
            last_error = exc
            if attempt_number == max_train_attempts:
                break
            print(
                f"Standard split IRT attempt {attempt_number} failed: {exc}. "
                "Retrying from a fresh initialization..."
            )
            continue
        finally:
            set_torch_determinism(True)
        break

    if last_error is not None:
        raise RuntimeError(
            f"Standard split IRT failed after {max_train_attempts} attempts"
        ) from last_error

    abilities = _as_frame(theta_by_agent, "agent", "ability")
    items = _as_frame(diff_by_item, "item_id", "b")
    with open(cache_dir / "cache_meta.json", "w") as f:
        json.dump(expected_meta, f, indent=2, sort_keys=True)
    return abilities, items


def get_or_train_oracle_irt(
    *,
    all_responses_tagged: Sequence[Tuple[str, str, Dict[str, int]]],
    all_item_ids: Set[str],
    output_dir: Path,
    epochs: int,
    device: str,
    seed: int,
    max_train_attempts: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/load full standard agent IRT used only by the oracle predictor."""

    from utils.irt_training import (
        train_standard_irt_1pl_agents,
    )

    if max_train_attempts < 1:
        raise ValueError(f"max_train_attempts must be >= 1, got {max_train_attempts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_meta = {
        "cache_kind": "standard_agent_1pl_oracle",
        "epochs": int(epochs),
        "seed": int(seed),
        "item_ids": sorted(str(item_id) for item_id in all_item_ids),
        "responses_signature": _responses_signature(all_responses_tagged, all_item_ids),
    }
    cached = _load_cached_standard_irt(output_dir, expected_meta)
    if cached is not None:
        return cached

    last_error: Optional[Exception] = None
    theta_by_agent: Dict[str, float] = {}
    diff_by_item: Dict[str, float] = {}
    for attempt_idx in range(max_train_attempts):
        attempt_number = attempt_idx + 1
        attempt_seed = int(seed) + attempt_idx
        attempt_dir = output_dir.parent / f"{output_dir.name}.attempt{attempt_number}"
        if max_train_attempts > 1:
            print(
                f"Training standard oracle IRT attempt "
                f"{attempt_number}/{max_train_attempts} (seed={attempt_seed})"
            )
        set_torch_determinism(False)
        try:
            theta_by_agent, diff_by_item = train_standard_irt_1pl_agents(
                all_responses_tagged=all_responses_tagged,
                keep_item_ids=set(all_item_ids),
                epochs=int(epochs),
                device=str(device),
                seed=attempt_seed,
                out_dir=str(attempt_dir),
            )
            last_error = None
        except Exception as exc:
            last_error = exc
            if attempt_number == max_train_attempts:
                break
            print(
                f"Standard oracle IRT attempt {attempt_number} failed: {exc}. "
                "Retrying from a fresh initialization..."
            )
            continue
        finally:
            set_torch_determinism(True)

        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        shutil.move(str(attempt_dir), str(output_dir))
        break

    if last_error is not None:
        raise RuntimeError(
            f"Standard oracle IRT failed after {max_train_attempts} attempts"
        ) from last_error

    abilities = _as_frame(theta_by_agent, "agent", "ability")
    items = _as_frame(diff_by_item, "item_id", "b")
    with open(output_dir / "cache_meta.json", "w") as f:
        json.dump(expected_meta, f, indent=2, sort_keys=True)
    return abilities, items
