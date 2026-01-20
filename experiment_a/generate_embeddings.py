#!/usr/bin/env python3
"""Generate embeddings for SWE-bench Verified tasks.

Embeds task instruction + solution from the HuggingFace dataset
using a configurable backbone model for difficulty prediction.

This script reuses the embedding infrastructure from predict_question_difficulty.py
(developed by Daria with careful ablations) to ensure consistent methodology.

The format is:
    Task statement:
    {problem_statement from HuggingFace}

    Solution:
    {patch from HuggingFace}

    How difficult is the above task for a coding agent? ...

Example usage:
    python -m experiment_a.generate_embeddings \
        --out_dir chris_output/experiment_a/embeddings \
        --batch_size 1 \
        --device_map auto

For SLURM cluster with task sharding:
    sbatch slurm_scripts/swebench_embeddings.sh "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the embedding infrastructure from predict_question_difficulty.py
from predict_question_difficulty import (
    DIFFICULTY_INSTRUCTION,
    ItemRecord,
    embed_items,
    iter_swebench_items,
    normalize_swebench_item_id,
)


def load_swebench_items(
    dataset_name: str,
    split: str,
    start_idx: int = 0,
    n_inputs: int = 0,
    seed: int = 0,
    shuffle: bool = False,
) -> List[ItemRecord]:
    """Load SWE-bench tasks as ItemRecords for embedding.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (e.g., "test")
        start_idx: Starting index for task sharding
        n_inputs: Number of tasks to load (0 = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle before selecting

    Returns:
        List of ItemRecord objects compatible with embed_items()
    """
    # Load all items first, then slice for sharding
    all_items = list(iter_swebench_items(
        dataset_names=[dataset_name],
        split=split,
        dataset_paths=[],
        n_inputs=0,  # Load all first
        seed=seed,
        shuffle=shuffle,
    ))

    print(f"Loaded {len(all_items)} total tasks from {dataset_name}")

    # Apply sharding
    if start_idx > 0 or n_inputs > 0:
        end_idx = start_idx + n_inputs if n_inputs > 0 else len(all_items)
        all_items = all_items[start_idx:end_idx]
        print(f"Shard: start_idx={start_idx}, n_inputs={n_inputs}, actual={len(all_items)}")

    return all_items


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for SWE-bench Verified tasks"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    # Sharding support for parallel jobs
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for task sharding (for parallel jobs)",
    )
    parser.add_argument(
        "--n_inputs",
        type=int,
        default=0,
        help="Number of tasks to embed (0 = all remaining from start_idx)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle tasks before sharding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling",
    )
    # Model configuration (same defaults as experiment_a_terminalbench)
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model for embeddings",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map (e.g., auto, none)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        help="Attention implementation (e.g., auto, flash_attention_2)",
    )
    parser.add_argument(
        "--embedding_layer",
        type=int,
        default=-1,
        help="Which hidden layer to pool embeddings from (-1 = last)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for model loading",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="chris_output/experiment_a/embeddings",
        help="Output directory",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DIFFICULTY_INSTRUCTION,
        help="Difficulty instruction to append",
    )
    args = parser.parse_args()

    # Resolve paths
    out_dir = ROOT / args.out_dir

    print(f"Dataset: {args.dataset_name} (split={args.split})")
    print(f"Backbone: {args.backbone}")

    # Load tasks as ItemRecords
    items = load_swebench_items(
        dataset_name=args.dataset_name,
        split=args.split,
        start_idx=args.start_idx,
        n_inputs=args.n_inputs,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    print(f"Prepared {len(items)} items for embedding")

    if not items:
        print("No items to embed. Exiting.")
        return

    # Embed tasks using the exact same infrastructure as predict_question_difficulty.py
    print(f"Embedding with backbone: {args.backbone}")
    print(f"Max length: {args.max_length}")
    print(f"Embedding layer: {args.embedding_layer}")

    ids_sorted, embeddings_by_id, counts_by_id, embedding_dim = embed_items(
        items=items,
        backbone=args.backbone,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        instruction=args.instruction,
        embedding_layer=args.embedding_layer,
    )

    print(f"Embedded {len(ids_sorted)} tasks with dim={embedding_dim}")

    # Build embedding matrix
    X = np.stack([embeddings_by_id[tid] for tid in ids_sorted], axis=0).astype(np.float32)

    # Save embeddings
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with model info (similar format to predict_question_difficulty.py)
    safe_backbone = args.backbone.replace("/", "__")
    layer_flag = "" if args.embedding_layer == -1 else f"__layer{args.embedding_layer}"

    # Include shard info in filename if sharding
    if args.start_idx > 0 or args.n_inputs > 0:
        shard_flag = f"__shard{args.start_idx}-{args.start_idx + len(ids_sorted)}"
    else:
        shard_flag = ""

    out_path = out_dir / f"embeddings__{safe_backbone}__pool-lasttoken{layer_flag}__maxlen{args.max_length}{shard_flag}.npz"

    np.savez_compressed(
        out_path,
        task_ids=np.array(ids_sorted, dtype=object),
        X=X,
        backbone=np.array([args.backbone], dtype=object),
        max_length=np.array([args.max_length], dtype=np.int64),
        embedding_dim=np.array([embedding_dim], dtype=np.int64),
        embedding_layer=np.array([args.embedding_layer], dtype=np.int64),
        instruction=np.array([args.instruction], dtype=object),
        dataset_name=np.array([args.dataset_name], dtype=object),
        split=np.array([args.split], dtype=object),
        start_idx=np.array([args.start_idx], dtype=np.int64),
        n_inputs=np.array([len(ids_sorted)], dtype=np.int64),
    )

    print(f"Saved embeddings to: {out_path}")


if __name__ == "__main__":
    main()
