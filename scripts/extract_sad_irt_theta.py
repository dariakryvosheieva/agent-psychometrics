#!/usr/bin/env python3
"""Extract theta (ability) values from SAD-IRT checkpoints.

This script extracts learned ability (theta) values from SAD-IRT checkpoints
and saves them as simple CSV files that can be transferred and used for evaluation.

Unlike beta extraction, theta extraction requires reconstructing agent IDs because
checkpoints don't store them directly. Uses experiment_sad_irt.data_splits for
agent ordering logic (this module has no heavy dependencies like peft/transformers).

Usage:
    # Extract from a single checkpoint
    python scripts/extract_sad_irt_theta.py \
        --checkpoint chris_output/sad_irt_long/full_20260118_024625/checkpoint_best*.pt \
        --output_dir chris_output/sad_irt_theta_values

    # Extract from all checkpoints in a directory
    python scripts/extract_sad_irt_theta.py \
        --checkpoint_dir chris_output/sad_irt_long \
        --output_dir chris_output/sad_irt_theta_values

    # Override paths if checkpoint config doesn't have them
    python scripts/extract_sad_irt_theta.py \
        --checkpoint_dir chris_output/sad_irt_long \
        --response_matrix clean_data/swebench_verified/swebench_verified_20251120_full.jsonl \
        --trajectory_dir chris_output/trajectory_summaries_api \
        --cutoff_date 20250807

Output format (CSV):
    agent_id,theta
    20240615_sweagent_gpt4,0.234
    20240701_aider_claude,-0.567
    ...
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

# Import directly to avoid loading peft/transformers from __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "data_splits", PROJECT_ROOT / "experiment_sad_irt" / "data_splits.py"
)
data_splits_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_splits_module)
get_pre_frontier_agents = data_splits_module.get_pre_frontier_agents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths (can be overridden via CLI or checkpoint config)
DEFAULT_RESPONSE_MATRIX = "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl"
DEFAULT_TRAJECTORY_DIR = "chris_output/trajectory_summaries_api"
DEFAULT_CUTOFF_DATE = "20250807"


def get_agent_ids_from_checkpoint(
    checkpoint: Dict,
    response_matrix_path: Optional[Path] = None,
    trajectory_dir: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
) -> Tuple[List[str], str]:
    """Get agent IDs for a checkpoint.

    Reconstructs agent ordering using checkpoint config or provided paths.

    Args:
        checkpoint: Loaded checkpoint dictionary
        response_matrix_path: Override for response matrix path
        trajectory_dir: Override for trajectory directory
        cutoff_date: Override for cutoff date

    Returns:
        Tuple of (agent_ids, cutoff_date_used)
    """
    config = checkpoint.get("config", {})

    # Use overrides if provided, otherwise fall back to checkpoint config
    if response_matrix_path is None:
        response_matrix_path = Path(
            config.get("response_matrix_path", DEFAULT_RESPONSE_MATRIX)
        )
    if trajectory_dir is None:
        trajectory_dir = Path(config.get("trajectory_dir", DEFAULT_TRAJECTORY_DIR))
    if cutoff_date is None:
        cutoff_date = config.get("frontier_cutoff_date", DEFAULT_CUTOFF_DATE)

    logger.info(f"Using response matrix: {response_matrix_path}")
    logger.info(f"Using trajectory dir: {trajectory_dir}")
    logger.info(f"Using cutoff date: {cutoff_date}")

    agent_ids, _ = get_pre_frontier_agents(
        response_matrix_path, trajectory_dir, cutoff_date
    )

    return agent_ids, cutoff_date


def extract_theta_from_checkpoint(
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    response_matrix_path: Optional[Path] = None,
    trajectory_dir: Optional[Path] = None,
    cutoff_date: Optional[str] = None,
) -> pd.DataFrame:
    """Extract theta values from a SAD-IRT checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        output_path: Optional path to save CSV output
        response_matrix_path: Override for response matrix path
        trajectory_dir: Override for trajectory directory
        cutoff_date: Override for cutoff date

    Returns:
        DataFrame with columns [agent_id, theta]
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint (CPU-only to avoid GPU memory)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract theta values
    theta_weight = checkpoint["model_state_dict"]["theta.weight"]
    theta_values = theta_weight.squeeze(-1).numpy()
    logger.info(f"Extracted {len(theta_values)} theta values")

    # Get agent IDs
    agent_ids, cutoff_used = get_agent_ids_from_checkpoint(
        checkpoint, response_matrix_path, trajectory_dir, cutoff_date
    )
    logger.info(f"Found {len(agent_ids)} pre-frontier agents (cutoff: {cutoff_used})")

    if len(agent_ids) != len(theta_values):
        logger.warning(
            f"Agent count mismatch: {len(agent_ids)} agent_ids vs {len(theta_values)} theta values"
        )
        # Truncate to the shorter length
        min_len = min(len(agent_ids), len(theta_values))
        agent_ids = agent_ids[:min_len]
        theta_values = theta_values[:min_len]

    # Create DataFrame
    df = pd.DataFrame({
        "agent_id": agent_ids,
        "theta": theta_values,
    })
    df = df.set_index("agent_id")

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info(f"Saved theta values to: {output_path}")

    return df


def find_best_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """Find all 'best' checkpoints in a directory tree.

    Looks for files matching 'checkpoint_best*.pt' in subdirectories.

    Args:
        checkpoint_dir: Root directory to search

    Returns:
        List of checkpoint file paths
    """
    checkpoints = []

    # Look for checkpoint_best*.pt files
    for cp_file in checkpoint_dir.rglob("checkpoint_best*.pt"):
        checkpoints.append(cp_file)

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Extract theta (ability) values from SAD-IRT checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a specific checkpoint file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Directory to search for checkpoint_best*.pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/sad_irt_theta_values"),
        help="Output directory for theta CSV files",
    )
    parser.add_argument(
        "--response_matrix",
        type=Path,
        default=None,
        help="Override response matrix path (default: from checkpoint config)",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=Path,
        default=None,
        help="Override trajectory directory (default: from checkpoint config)",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        default=None,
        help="Override cutoff date YYYYMMDD (default: from checkpoint config)",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_dir is None:
        print("Error: Must specify either --checkpoint or --checkpoint_dir")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect checkpoints to process
    checkpoints = []
    if args.checkpoint is not None:
        if args.checkpoint.exists():
            checkpoints.append(args.checkpoint)
        else:
            # Try glob pattern
            for cp in Path(".").glob(str(args.checkpoint)):
                checkpoints.append(cp)

    if args.checkpoint_dir is not None:
        checkpoints.extend(find_best_checkpoints(args.checkpoint_dir))

    if not checkpoints:
        print("No checkpoints found")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s) to process")

    # Process each checkpoint
    for checkpoint_path in checkpoints:
        # Create output filename from checkpoint path
        # e.g., full_20260118_024625_psi_batchnorm_lora_r64/checkpoint_best_step472.pt
        #    -> full_20260118_024625_psi_batchnorm_lora_r64.csv
        parent_name = checkpoint_path.parent.name
        output_name = f"{parent_name}.csv"
        output_path = args.output_dir / output_name

        try:
            extract_theta_from_checkpoint(
                checkpoint_path,
                output_path,
                response_matrix_path=args.response_matrix,
                trajectory_dir=args.trajectory_dir,
                cutoff_date=args.cutoff_date,
            )
            print(f"  ✓ {parent_name}")
        except Exception as e:
            print(f"  ✗ {parent_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTheta values saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
