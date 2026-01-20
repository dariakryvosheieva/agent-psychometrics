#!/usr/bin/env python3
"""Extract beta values from SAD-IRT checkpoints.

This is a standalone script with minimal dependencies (torch, pandas, json).
It does NOT import from experiment_sad_irt to avoid peft/transformers dependencies.

This script extracts learned difficulty (beta) values from SAD-IRT checkpoints
and saves them as simple CSV files that can be transferred and used for evaluation.

Usage:
    # Extract from a single checkpoint
    python scripts/extract_sad_irt_beta.py \
        --checkpoint chris_output/sad_irt_long/full_20260118_024625/checkpoint_best*.pt \
        --output_dir chris_output/sad_irt_beta_values

    # Extract from all checkpoints in a directory
    python scripts/extract_sad_irt_beta.py \
        --checkpoint_dir chris_output/sad_irt_long \
        --output_dir chris_output/sad_irt_beta_values

Output format (CSV):
    task_id,beta
    astropy__astropy-12907,1.234
    django__django-11099,0.567
    ...
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_task_ids_from_checkpoint(checkpoint: Dict) -> List[str]:
    """Get task IDs from checkpoint.

    Newer checkpoints (after 2026-01-19) store task_ids directly.
    For older checkpoints, we fall back to reading from the response matrix.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        List of task IDs in the same order as the beta embedding
    """
    # Newer checkpoints store task_ids directly
    if "task_ids" in checkpoint and checkpoint["task_ids"] is not None:
        logger.info("Using task_ids from checkpoint")
        return checkpoint["task_ids"]

    # Fallback for older checkpoints: read from response matrix
    logger.info("Checkpoint missing task_ids, falling back to response matrix")
    config = checkpoint.get("config", {})
    response_matrix_path = Path(
        config.get(
            "response_matrix_path",
            "clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        )
    )

    # Read first line to get task IDs (all agents have the same tasks)
    with open(response_matrix_path) as f:
        first_line = json.loads(f.readline())
        task_ids = list(first_line["responses"].keys())

    return task_ids


def extract_beta_from_checkpoint(
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Extract beta values from a SAD-IRT checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        output_path: Optional path to save CSV output

    Returns:
        DataFrame with columns [task_id, beta]
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint (CPU-only to avoid GPU memory)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract beta values
    beta_weight = checkpoint["model_state_dict"]["beta.weight"]
    beta_values = beta_weight.squeeze(-1).numpy()
    logger.info(f"Extracted {len(beta_values)} beta values")

    # Get task IDs from checkpoint
    task_ids = get_task_ids_from_checkpoint(checkpoint)

    if len(task_ids) != len(beta_values):
        logger.warning(
            f"Task count mismatch: {len(task_ids)} task_ids vs {len(beta_values)} beta values"
        )
        # Truncate to the shorter length
        min_len = min(len(task_ids), len(beta_values))
        task_ids = task_ids[:min_len]
        beta_values = beta_values[:min_len]

    # Create DataFrame
    df = pd.DataFrame({
        "task_id": task_ids,
        "beta": beta_values,
    })
    df = df.set_index("task_id")

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logger.info(f"Saved beta values to: {output_path}")

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
        description="Extract beta values from SAD-IRT checkpoints"
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
        default=Path("chris_output/sad_irt_beta_values"),
        help="Output directory for beta CSV files",
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
            extract_beta_from_checkpoint(checkpoint_path, output_path)
            print(f"  ✓ {parent_name}")
        except Exception as e:
            print(f"  ✗ {parent_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nBeta values saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
