"""Data preparation utilities for Experiment B.

This module consolidates:
- Agent/task splitting by date
- Frontier task identification (pass-rate and IRT-based)
- Baseline IRT training and caching
- ExperimentData loading and preparation

Main entry point: load_and_prepare_data()
"""

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from experiment_b.shared.config_base import DatasetConfig

logger = logging.getLogger(__name__)


# =============================================================================
# ExperimentData dataclass
# =============================================================================


@dataclass
class ExperimentData:
    """Runtime-computed values specific to this experiment run.

    Contains values that depend on experiment parameters (cutoff date,
    frontier definitions, etc.). Dataset-inherent values come from config.
    """

    config: DatasetConfig  # Reference to dataset config

    # Loaded IRT models (depend on paths, potentially CLI overrides)
    oracle_items: pd.DataFrame
    oracle_abilities: pd.DataFrame
    baseline_items: pd.DataFrame
    baseline_abilities: Optional[pd.DataFrame]

    # Agent splits (depend on cutoff_date)
    pre_frontier_agents: List[str]
    post_frontier_agents: List[str]

    # Task sets (depend on frontier definitions and cutoff)
    train_task_ids: List[str]  # All tasks in baseline IRT
    frontier_tasks_by_def: Dict[str, List[str]]  # frontier_def -> task list
    anchor_task_ids: List[str]  # For scale alignment

    # Cutoff date used (may be overridden from CLI)
    cutoff_date: str

    # Pre-filtered responses for training (no post-frontier agents)
    train_responses: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Ground truth difficulties from baseline IRT for training predictors
    baseline_ground_truth_b: Optional[np.ndarray] = field(default=None, repr=False)


# =============================================================================
# Agent splitting functions
# =============================================================================


def split_agents_by_dates(
    agents: List[str],
    agent_dates: Dict[str, str],
    cutoff_date: str,
) -> Tuple[List[str], List[str]]:
    """Split agents into pre-frontier and post-frontier by date cutoff.

    This is a generic function that works with any source of agent dates
    (e.g., from agent name prefix for SWE-bench, or from metadata for TerminalBench).

    Args:
        agents: List of agent names
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD format)
        cutoff_date: Date string in YYYYMMDD format. Agents with date >= cutoff
                     are post-frontier, agents with date < cutoff are pre-frontier.

    Returns:
        Tuple of (pre_frontier_agents, post_frontier_agents)

    Raises:
        ValueError: If any agent is missing a date in agent_dates
    """
    pre_frontier = []
    post_frontier = []
    missing_dates = []

    for agent in agents:
        date = agent_dates.get(agent)
        if not date:
            missing_dates.append(agent)
            continue

        if date >= cutoff_date:
            post_frontier.append(agent)
        else:
            pre_frontier.append(agent)

    if missing_dates:
        raise ValueError(
            f"{len(missing_dates)} agents missing dates. "
            f"First 5: {missing_dates[:5]}"
        )

    return pre_frontier, post_frontier


# =============================================================================
# Pass rate computation
# =============================================================================


def compute_pass_rates(
    responses_path: Path,
    agents: List[str],
) -> Dict[str, float]:
    """Compute empirical pass rate for each task among specified agents.

    Pass rate = fraction of agents that solved the task (at least once).
    Supports both binary (0/1) and binomial ({successes, trials}) data.

    Args:
        responses_path: Path to JSONL response matrix
        agents: List of agent names to include

    Returns:
        Dict mapping task_id -> pass_rate (0-1), where pass_rate is the
        fraction of agents that achieved at least one success on the task
    """
    agent_set = set(agents)
    task_successes: Dict[str, List[int]] = defaultdict(list)

    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            if data["subject_id"] not in agent_set:
                continue
            for task_id, response in data["responses"].items():
                # Handle both binary and binomial responses
                # For pass rate, we care about whether agent solved at least once
                if isinstance(response, dict) and "successes" in response:
                    # Binomial: 1 if any success, 0 otherwise
                    task_successes[task_id].append(1 if response["successes"] > 0 else 0)
                else:
                    # Binary: 0 or 1
                    task_successes[task_id].append(int(response))

    return {
        task_id: sum(successes) / len(successes) if successes else 0.0
        for task_id, successes in task_successes.items()
    }


# =============================================================================
# Frontier task identification
# =============================================================================


def identify_frontier_tasks_passrate(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
    pre_threshold: float = 0.1,
    post_threshold: float = 0.1,
) -> List[str]:
    """Identify frontier tasks using pass rate thresholds.

    Frontier tasks are those where:
    - Pass rate among pre-frontier agents <= pre_threshold (e.g., 10%)
    - Pass rate among post-frontier agents > post_threshold (e.g., 10%)

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names
        pre_threshold: Maximum pass rate for pre-frontier (default 0.1 = 10%)
        post_threshold: Minimum pass rate for post-frontier (default 0.1 = 10%)

    Returns:
        List of task_ids that are frontier tasks
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    frontier_tasks = []

    # Check for tasks in pre but not in post (and vice versa)
    pre_only = set(pre_pass_rates.keys()) - set(post_pass_rates.keys())
    post_only = set(post_pass_rates.keys()) - set(pre_pass_rates.keys())

    if pre_only:
        raise ValueError(
            f"{len(pre_only)} tasks have pre-frontier data but no post-frontier data. "
            f"First 5: {list(pre_only)[:5]}"
        )
    if post_only:
        raise ValueError(
            f"{len(post_only)} tasks have post-frontier data but no pre-frontier data. "
            f"First 5: {list(post_only)[:5]}"
        )

    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates[task_id]
        post_rate = post_pass_rates[task_id]

        if pre_rate <= pre_threshold and post_rate > post_threshold:
            frontier_tasks.append(task_id)

    return frontier_tasks


def identify_frontier_tasks_irt(
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    agent_dates: Dict[str, str],
    cutoff_date: str,
    solve_probability: float = 0.3,
) -> List[str]:
    """Identify frontier tasks using IRT probability threshold.

    A task is frontier if NO agent before the cutoff date has theta >= beta + logit(p)
    (i.e., no pre-frontier agent can solve it with at least p probability).

    This differs from identify_frontier_tasks_passrate() which uses empirical pass rates.
    The IRT-based definition is more principled since P(success) = sigmoid(theta - beta).

    Args:
        oracle_items: DataFrame with 'b' column (oracle task difficulties)
        oracle_abilities: DataFrame with 'theta' column (oracle agent abilities)
        agent_dates: Dict mapping agent_id -> date string (YYYYMMDD)
        cutoff_date: Cutoff date string (YYYYMMDD). Tasks where the first capable
                     agent appears ON or AFTER this date are frontier tasks.
        solve_probability: Probability threshold for considering an agent "capable"
            of solving a task (default 0.3, i.e., 30% solve rate)

    Returns:
        List of task_ids that are frontier tasks (excludes tasks with no capable agent)
    """
    from experiment_b.shared.date_forecasting import (
        compute_first_capable_dates,
        split_tasks_by_first_capable_date,
        parse_date,
    )

    # Compute first capable date for each task
    result = compute_first_capable_dates(
        oracle_items, oracle_abilities, agent_dates, solve_probability
    )

    # Split by cutoff: tasks where first capable agent is on/after cutoff are "frontier"
    cutoff_datetime = parse_date(cutoff_date)
    pre_cutoff_tasks, post_cutoff_tasks = split_tasks_by_first_capable_date(
        result.first_capable_dates, cutoff_datetime
    )

    # Exclude tasks with no capable agent (no ground truth for evaluation)
    # These are logged for visibility
    if result.tasks_without_capable_agent:
        logger.info(
            f"Excluding {len(result.tasks_without_capable_agent)} tasks with no capable agent "
            f"(no agent has >={solve_probability:.0%} solve probability)"
        )

    return post_cutoff_tasks


def identify_frontier_tasks_zero_pre(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
) -> List[str]:
    """Identify frontier tasks: zero pre-frontier solves, nonzero post-frontier solves.

    Frontier tasks are those where:
    - Pass rate among pre-frontier agents == 0 (no pre-frontier agent solves it)
    - Pass rate among post-frontier agents > 0 (at least one post-frontier agent solves it)

    This is a stricter criterion than identify_frontier_tasks_passrate() which allows up to
    10% pre-frontier pass rate.

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names

    Returns:
        List of task_ids that are frontier tasks
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    frontier_tasks = []
    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates[task_id]
        post_rate = post_pass_rates[task_id]

        if pre_rate == 0.0 and post_rate > 0.0:
            frontier_tasks.append(task_id)

    return frontier_tasks


def identify_nontrivial_tasks(
    responses_path: Path,
    pre_frontier_agents: List[str],
    post_frontier_agents: List[str],
    min_pass_rate: float = 0.10,
    max_pass_rate: float = 0.90,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """Identify tasks with non-trivial pass rates in BOTH agent groups.

    Non-trivial tasks have meaningful variation - neither too easy nor too hard
    for both pre-frontier and post-frontier agents. These are useful as anchor
    tasks for aligning IRT scales.

    Args:
        responses_path: Path to JSONL response matrix
        pre_frontier_agents: List of pre-frontier agent names
        post_frontier_agents: List of post-frontier agent names
        min_pass_rate: Minimum pass rate threshold (default 0.10 = 10%)
        max_pass_rate: Maximum pass rate threshold (default 0.90 = 90%)

    Returns:
        Tuple of (nontrivial_task_ids, pre_pass_rates, post_pass_rates)
    """
    pre_pass_rates = compute_pass_rates(responses_path, pre_frontier_agents)
    post_pass_rates = compute_pass_rates(responses_path, post_frontier_agents)

    # Check for tasks in pre but not in post (and vice versa)
    pre_only = set(pre_pass_rates.keys()) - set(post_pass_rates.keys())
    post_only = set(post_pass_rates.keys()) - set(pre_pass_rates.keys())

    if pre_only:
        raise ValueError(
            f"{len(pre_only)} tasks have pre-frontier data but no post-frontier data. "
            f"First 5: {list(pre_only)[:5]}"
        )
    if post_only:
        raise ValueError(
            f"{len(post_only)} tasks have post-frontier data but no pre-frontier data. "
            f"First 5: {list(post_only)[:5]}"
        )

    nontrivial_tasks = []
    for task_id in pre_pass_rates:
        pre_rate = pre_pass_rates[task_id]
        post_rate = post_pass_rates[task_id]

        # Both groups must have meaningful variation
        pre_nontrivial = min_pass_rate <= pre_rate <= max_pass_rate
        post_nontrivial = min_pass_rate <= post_rate <= max_pass_rate

        if pre_nontrivial and post_nontrivial:
            nontrivial_tasks.append(task_id)

    return nontrivial_tasks, pre_pass_rates, post_pass_rates


# =============================================================================
# Agent/response utilities
# =============================================================================


def get_all_agents_from_responses(responses_path: Path) -> List[str]:
    """Get list of all agent IDs from response matrix.

    Args:
        responses_path: Path to JSONL response matrix

    Returns:
        List of agent IDs (subject_id values)
    """
    agents = []
    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            agents.append(data["subject_id"])
    return agents


def get_agents_with_trajectories(trajectories_dir: Path) -> Set[str]:
    """Get set of agents that have trajectory data.

    Args:
        trajectories_dir: Path to trajectory directory

    Returns:
        Set of agent names that have trajectory subdirectories
    """
    agents = set()
    for path in trajectories_dir.iterdir():
        if path.is_dir() and not path.name.startswith("_"):
            agents.add(path.name)
    return agents


def get_pre_frontier_agents(
    responses_path: Path,
    trajectories_dir: Path,
    cutoff_date: str = "20250807",
) -> Tuple[List[str], List[str]]:
    """Get pre-frontier and post-frontier agent lists for training/inference.

    This is the canonical function for determining agent lists. It ensures
    consistent ordering between training and inference by:
    1. Reading agents from response matrix (preserves JSONL line order)
    2. Filtering to agents with trajectories
    3. Splitting by cutoff date

    NOTE: This function is specific to SWE-bench agents that have date prefixes
    in their names. For other datasets, use split_agents_by_dates() directly
    with agent dates from the dataset config.

    Args:
        responses_path: Path to JSONL response matrix
        trajectories_dir: Path to trajectory directory
        cutoff_date: Date cutoff for pre/post frontier (YYYYMMDD format)

    Returns:
        Tuple of (pre_frontier_agents, post_frontier_agents)
    """
    # Import here to avoid circular imports
    from experiment_b.datasets.swebench import extract_date_prefix

    # Get all agents in response matrix order (this order is preserved!)
    all_agents = get_all_agents_from_responses(responses_path)

    # Get agents with trajectories
    traj_agents = get_agents_with_trajectories(trajectories_dir)

    # Filter to agents with both (preserving response matrix order)
    agents_with_both = [a for a in all_agents if a in traj_agents]

    # Build agent_dates dict from name prefixes
    agent_dates = {
        agent: extract_date_prefix(agent)
        for agent in agents_with_both
    }

    # Split by cutoff date
    pre_frontier, post_frontier = split_agents_by_dates(
        agents_with_both, agent_dates, cutoff_date=cutoff_date
    )

    return pre_frontier, post_frontier


# =============================================================================
# Baseline IRT caching
# =============================================================================


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


def _train_baseline_irt_on_agents(
    responses_path: Path,
    agent_subset: List[str],
    output_dir: Path,
    epochs: int = 2000,
) -> Dict[str, float]:
    """Train standard IRT on a subset of agents using py_irt.

    Supports both binary (SWE-bench) and binomial (TerminalBench) data formats.

    Args:
        responses_path: Path to response matrix JSONL
        agent_subset: List of agent IDs to include in training
        output_dir: Directory to save IRT outputs
        epochs: Number of training epochs for py_irt

    Returns:
        Dict mapping task_id -> difficulty (β)
    """
    import pyro
    import torch
    import tempfile

    from py_irt.dataset import Dataset
    from py_irt.models import OneParamLog
    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    # Load response matrix and filter to subset of agents
    agent_set = set(agent_subset)
    filtered_records = []
    is_binomial = False

    with open(responses_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record['subject_id'] in agent_set:
                filtered_records.append(record)
                # Check if first response is dict (binomial data)
                if not is_binomial and record['responses']:
                    first_response = next(iter(record['responses'].values()))
                    if isinstance(first_response, dict) and "successes" in first_response:
                        is_binomial = True

    logger.info(f"Loaded {len(filtered_records)} agent responses for baseline IRT")
    logger.info(f"Data format: {'binomial' if is_binomial else 'binary'}")

    if is_binomial:
        # Write filtered records to temp file and use from_jsonlines
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            for record in filtered_records:
                tmp.write(json.dumps(record) + '\n')
            tmp_path = tmp.name

        try:
            dataset = Dataset.from_jsonlines(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)
    else:
        # Binary data - use pandas approach
        data_list = []
        for record in filtered_records:
            row = {'subject_id': record['subject_id']}
            row.update(record['responses'])
            data_list.append(row)

        df = pd.DataFrame(data_list)
        item_columns = [col for col in df.columns if col != 'subject_id']
        dataset = Dataset.from_pandas(df, subject_column="subject_id", item_columns=item_columns)

    # Train 1PL IRT with fixed seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    config = IrtConfig(
        model_type=OneParamLog,
        priors="hierarchical",
        initializers=[
            {"name": "difficulty_from_accuracy", "eps": 1e-3},
        ],
        seed=seed,
    )

    # Clear pyro param store to avoid conflicts
    pyro.clear_param_store()

    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(epochs=epochs)

    # Extract difficulty and ability parameters
    difficulties = list(trainer.best_params["diff"])
    abilities = list(trainer.best_params["ability"])
    item_id_map = trainer.best_params["item_ids"]
    subject_id_map = trainer.best_params["subject_ids"]
    item_ids = [item_id_map[i] for i in range(len(difficulties))]
    subject_ids = [subject_id_map[i] for i in range(len(abilities))]

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    items_df = pd.DataFrame({
        "b": difficulties,
    }, index=item_ids)
    items_df.to_csv(output_dir / "items.csv")

    abilities_df = pd.DataFrame({
        "theta": abilities,
    }, index=subject_ids)
    abilities_df.to_csv(output_dir / "abilities.csv")

    logger.info(f"Baseline IRT saved to {output_dir}")
    logger.info(f"β stats: mean={np.mean(difficulties):.4f}, std={np.std(difficulties):.4f}")
    logger.info(f"θ stats: mean={np.mean(abilities):.4f}, std={np.std(abilities):.4f}")

    return {task_id: diff for task_id, diff in zip(item_ids, difficulties)}


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

    cache_dir.mkdir(parents=True, exist_ok=True)
    _train_baseline_irt_on_agents(
        responses_path=responses_path,
        agent_subset=pre_frontier_agents,
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


# =============================================================================
# Main entry point: load_and_prepare_data
# =============================================================================


def load_and_prepare_data(args: argparse.Namespace, config: DatasetConfig) -> ExperimentData:
    """Load IRT models and compute experiment-specific splits.

    Args:
        args: Parsed CLI arguments
        config: Dataset configuration (with lazy-loaded data)

    Returns:
        ExperimentData with all loaded models and computed splits
    """
    # Override paths from CLI args if provided
    oracle_irt_path = args.oracle_irt_path or config.oracle_irt_path
    oracle_abilities_path = args.oracle_abilities_path or config.oracle_abilities_path
    baseline_irt_path = args.baseline_irt_path or config.baseline_irt_path
    cutoff_date = args.cutoff_date or config.cutoff_date
    pre_threshold = args.pre_threshold if args.pre_threshold is not None else config.pre_threshold
    post_threshold = args.post_threshold if args.post_threshold is not None else config.post_threshold

    print(f"  Dataset: {config.name}")
    print(f"  Cutoff date: {cutoff_date}")

    # Validate required files exist
    required_files = [
        (config.responses_path, "Response matrix"),
        (oracle_irt_path, "Oracle IRT"),
        (oracle_abilities_path, "Oracle abilities"),
    ]
    for path, name in required_files:
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    # Load IRT models and abilities
    print("\nLoading IRT models...")
    oracle_items = pd.read_csv(oracle_irt_path, index_col=0)
    oracle_abilities = pd.read_csv(oracle_abilities_path, index_col=0)
    print(f"  Oracle IRT: {len(oracle_items)} tasks")
    print(f"  Oracle abilities: {len(oracle_abilities)} agents")

    # Load response matrix (use lazy-loaded from config)
    print("\nLoading response matrix...")
    print(f"  Loaded responses for {len(config.responses)} agents")

    # Get agent dates from config and split by cutoff
    print("\nIdentifying frontier tasks...")
    print(f"  Agents with dates: {len(config.agent_dates)} / {len(config.all_agents)}")

    pre_frontier, post_frontier = split_agents_by_dates(
        config.all_agents, config.agent_dates, cutoff_date
    )
    print(f"  Pre-frontier agents (< {cutoff_date}): {len(pre_frontier)}")
    print(f"  Post-frontier agents (>= {cutoff_date}): {len(post_frontier)}")

    # Load or train baseline IRT (pre-frontier agents only)
    baseline_items: Optional[pd.DataFrame] = None
    baseline_abilities: Optional[pd.DataFrame] = None

    if baseline_irt_path and baseline_irt_path.exists():
        baseline_abilities_path_file = baseline_irt_path.parent / "abilities.csv"
        if baseline_abilities_path_file.exists():
            baseline_items = pd.read_csv(baseline_irt_path, index_col=0)
            baseline_abilities = pd.read_csv(baseline_abilities_path_file, index_col=0)
            print(f"  Baseline IRT: {len(baseline_items)} tasks (loaded from {baseline_irt_path})")
        else:
            print(f"  Baseline IRT abilities not found at {baseline_abilities_path_file}, retraining...")

    if baseline_items is None:
        print("\nLoading/training baseline IRT...")
        baseline_items, baseline_abilities = get_or_train_baseline_irt(
            responses_path=config.responses_path,
            pre_frontier_agents=pre_frontier,
            cutoff_date=cutoff_date,
            output_dir=config.output_dir,
        )
        print(f"  Baseline IRT: {len(baseline_items)} tasks, {len(baseline_abilities)} agents")

    irt_solve_prob = config.irt_solve_probability

    # Identify frontier tasks for each definition
    frontier_tasks_by_def: Dict[str, List[str]] = {}
    for frontier_def in args.frontier_definitions:
        if frontier_def == "irt":
            frontier_task_ids = identify_frontier_tasks_irt(
                oracle_items=oracle_items,
                oracle_abilities=oracle_abilities,
                agent_dates=config.agent_dates,
                cutoff_date=cutoff_date,
                solve_probability=irt_solve_prob,
            )
            print(f"  Frontier tasks ({frontier_def}: no pre-frontier agent with >={irt_solve_prob:.0%} solve prob): {len(frontier_task_ids)}")
        elif frontier_def == "zero_pre":
            frontier_task_ids = identify_frontier_tasks_zero_pre(
                config.responses_path,
                pre_frontier,
                post_frontier,
            )
            print(f"  Frontier tasks ({frontier_def}: 0% pre, >0% post): {len(frontier_task_ids)}")
        else:  # passrate
            frontier_task_ids = identify_frontier_tasks_passrate(
                config.responses_path,
                pre_frontier,
                post_frontier,
                pre_threshold,
                post_threshold,
            )
            print(f"  Frontier tasks ({frontier_def}: <={pre_threshold*100:.0f}% pre, >{post_threshold*100:.0f}% post): {len(frontier_task_ids)}")
        frontier_tasks_by_def[frontier_def] = frontier_task_ids

    # Identify nontrivial anchor tasks for scale alignment
    print("\nIdentifying nontrivial anchor tasks...")
    anchor_task_ids, _, _ = identify_nontrivial_tasks(
        config.responses_path,
        pre_frontier,
        post_frontier,
        min_pass_rate=0.10,
        max_pass_rate=0.90,
    )
    print(f"  Anchor tasks (10-90% pass rate in both groups): {len(anchor_task_ids)}")

    train_task_ids = list(baseline_items.index)
    print(f"  Training tasks: {len(train_task_ids)}")

    # Pre-filter responses for training (no post-frontier agents)
    train_responses = {
        agent_id: agent_responses
        for agent_id, agent_responses in config.responses.items()
        if agent_id in pre_frontier
    }

    baseline_ground_truth_b = baseline_items.loc[train_task_ids, "b"].values

    return ExperimentData(
        config=config,
        oracle_items=oracle_items,
        oracle_abilities=oracle_abilities,
        baseline_items=baseline_items,
        baseline_abilities=baseline_abilities,
        pre_frontier_agents=pre_frontier,
        post_frontier_agents=post_frontier,
        train_task_ids=train_task_ids,
        frontier_tasks_by_def=frontier_tasks_by_def,
        anchor_task_ids=anchor_task_ids,
        cutoff_date=cutoff_date,
        train_responses=train_responses,
        baseline_ground_truth_b=baseline_ground_truth_b,
    )
