"""Baseline IRT caching utilities for Experiment B.

Handles training and caching of IRT models on pre-frontier agents only.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def compute_baseline_irt_cache_key(
    responses_path: Path,
    pre_frontier_agents: List[str],
    cutoff_date: str,
) -> str:
    """Compute a cache key for baseline IRT based on training data.

    The cache key captures:
    - Path to responses file (file name, not full path for portability)
    - List of pre-frontier agents (sorted for determinism)
    - Cutoff date

    If any of these change, the cache should be invalidated.

    Args:
        responses_path: Path to response matrix JSONL
        pre_frontier_agents: List of pre-frontier agent IDs
        cutoff_date: Frontier cutoff date string (YYYYMMDD)

    Returns:
        Cache key string (first 12 chars of SHA256 hash)
    """
    cache_components = {
        "responses_file": responses_path.name,
        "pre_frontier_agents": sorted(pre_frontier_agents),
        "cutoff_date": cutoff_date,
    }
    cache_str = json.dumps(cache_components, sort_keys=True)
    cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()[:12]
    return cache_hash


def get_or_train_baseline_irt(
    responses_path: Path,
    pre_frontier_agents: List[str],
    cutoff_date: str,
    output_dir: Path,
    force_retrain: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get cached baseline IRT or train a new one.

    Baseline IRT is trained on pre-frontier agents only to provide
    uncontaminated difficulty estimates. The cache key includes:
    - Response matrix file name
    - Sorted list of pre-frontier agents
    - Cutoff date

    If any of these change, the cache is invalidated and IRT is retrained.

    Args:
        responses_path: Path to response matrix JSONL
        pre_frontier_agents: List of pre-frontier agent IDs
        cutoff_date: Frontier cutoff date string (YYYYMMDD)
        output_dir: Base output directory
        force_retrain: If True, retrain even if cache exists

    Returns:
        Tuple of (items_df, abilities_df):
            - items_df: DataFrame with 'b' column containing task difficulties
            - abilities_df: DataFrame with 'theta' column containing agent abilities
    """
    # Compute cache key and paths
    cache_key = compute_baseline_irt_cache_key(
        responses_path, pre_frontier_agents, cutoff_date
    )
    cache_dir = output_dir / "baseline_irt" / f"cache_{cache_key}"
    items_path = cache_dir / "items.csv"
    abilities_path = cache_dir / "abilities.csv"
    cache_info_path = cache_dir / "cache_info.json"

    # Check for valid cache (both items and abilities must exist)
    if (
        not force_retrain
        and items_path.exists()
        and abilities_path.exists()
        and cache_info_path.exists()
    ):
        # Verify cache info matches
        with open(cache_info_path) as f:
            cached_info = json.load(f)

        if cached_info.get("cache_key") == cache_key:
            baseline_items = pd.read_csv(items_path, index_col=0)
            baseline_abilities = pd.read_csv(abilities_path, index_col=0)
            logger.info(f"Loaded cached baseline IRT from {cache_dir}")
            logger.info(f"  Cache key: {cache_key}")
            logger.info(f"  Pre-frontier agents: {cached_info.get('n_pre_frontier_agents')}")
            logger.info(f"  Cutoff date: {cached_info.get('cutoff_date')}")
            return baseline_items, baseline_abilities

    # Train new baseline IRT
    logger.info(f"Training baseline IRT on pre-frontier agents...")
    logger.info(f"  Cache key: {cache_key}")
    logger.info(f"  Pre-frontier agents: {len(pre_frontier_agents)}")
    logger.info(f"  Cutoff date: {cutoff_date}")

    from experiment_sad_irt.train_evaluate import train_baseline_irt_on_prefrontier

    cache_dir.mkdir(parents=True, exist_ok=True)
    train_baseline_irt_on_prefrontier(
        responses_path=responses_path,
        pre_frontier_agents=pre_frontier_agents,
        output_dir=cache_dir,
    )

    # Save cache info for validation
    cache_info = {
        "cache_key": cache_key,
        "responses_file": responses_path.name,
        "responses_path": str(responses_path),
        "n_pre_frontier_agents": len(pre_frontier_agents),
        "pre_frontier_agents": sorted(pre_frontier_agents),
        "cutoff_date": cutoff_date,
    }
    with open(cache_info_path, "w") as f:
        json.dump(cache_info, f, indent=2)

    baseline_items = pd.read_csv(items_path, index_col=0)
    baseline_abilities = pd.read_csv(abilities_path, index_col=0)
    logger.info(f"Saved baseline IRT cache to {cache_dir}")
    return baseline_items, baseline_abilities
