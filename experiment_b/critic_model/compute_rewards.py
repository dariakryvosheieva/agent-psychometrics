#!/usr/bin/env python3
"""
Compute per-step rewards for agent trajectories using OpenHands Critic Model.

This script runs on GPU (H200) and extracts V_t values for each step in
a trajectory, where V_t ≈ P(success | state_t).

Usage:
    python -m experiment_b.critic_model.compute_rewards \
        --trajectories_dir trajectory_data/unified_trajs \
        --output_dir chris_output/experiment_b/critic_rewards \
        --shard_id 0 --num_shards 2
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "all-hands/openhands-critic-32b-exp-20250417"
MAX_LENGTH = 32768  # Model's max position embeddings


@dataclass
class TrajectoryData:
    """Container for loaded trajectory data."""
    task_id: str
    agent_id: str
    resolved: bool
    messages: List[Dict[str, str]]


# ============================================================================
# Trajectory Loading
# ============================================================================

def load_trajectory(filepath: Path) -> Optional[TrajectoryData]:
    """Load a trajectory JSON file."""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
        return TrajectoryData(
            task_id=data.get("task_id", ""),
            agent_id=data.get("agent", ""),
            resolved=data.get("resolved", False),
            messages=data.get("messages", []),
        )
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def get_assistant_turn_boundaries(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int,
) -> Tuple[str, List[int]]:
    """
    Format trajectory as text and find token positions for each assistant turn.

    Returns:
        full_text: The formatted trajectory text
        assistant_end_positions: Token indices where each assistant turn ends
    """
    # Build the conversation text
    parts = []
    assistant_ends = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
            # Mark where this assistant turn ends
            assistant_ends.append(len("".join(parts)))
        else:
            parts.append(f"<|{role}|>\n{content}\n")

    full_text = "".join(parts)

    # Tokenize to find exact token positions
    # We need to map character positions to token positions
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    offsets = encoding["offset_mapping"][0].tolist()  # List of (start, end) char positions

    # Map character positions to token indices
    assistant_token_positions = []
    for char_pos in assistant_ends:
        # Find the token that contains this character position
        for token_idx, (start, end) in enumerate(offsets):
            if start <= char_pos <= end:
                assistant_token_positions.append(token_idx)
                break
        else:
            # If not found (due to truncation), use last token
            if offsets:
                assistant_token_positions.append(len(offsets) - 1)

    return full_text, assistant_token_positions, encoding


def format_trajectory_chat(messages: List[Dict[str, str]]) -> str:
    """
    Format trajectory messages for the critic model.

    Uses a simple format that mirrors OpenHands conversation structure.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"System: {content}\n\n")
        elif role == "user":
            parts.append(f"User: {content}\n\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n\n")
        else:
            parts.append(f"{role.capitalize()}: {content}\n\n")

    return "".join(parts)


# ============================================================================
# Critic Model Inference
# ============================================================================

class CriticModel:
    """Wrapper for OpenHands Critic Model inference."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.dtype = dtype

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Ensure padding is set up
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model from {model_name}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def get_per_step_rewards(
        self,
        messages: List[Dict[str, str]],
        max_length: int = MAX_LENGTH,
    ) -> np.ndarray:
        """
        Compute reward values for each assistant turn in the trajectory.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum sequence length

        Returns:
            Array of shape (num_assistant_turns,) with reward values
        """
        # Find assistant turns
        assistant_indices = [
            i for i, msg in enumerate(messages)
            if msg.get("role") == "assistant"
        ]

        if not assistant_indices:
            return np.array([], dtype=np.float32)

        # Format trajectory text
        text = format_trajectory_chat(messages)

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)
        offsets = encoding["offset_mapping"][0].tolist()

        # Find token positions for end of each assistant turn
        # We'll extract the reward at the last token of each assistant response
        assistant_end_chars = []
        char_pos = 0
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Update character position
            if role == "system":
                char_pos += len(f"System: {content}\n\n")
            elif role == "user":
                char_pos += len(f"User: {content}\n\n")
            elif role == "assistant":
                char_pos += len(f"Assistant: {content}\n\n")
                assistant_end_chars.append(char_pos - 1)  # End of this turn
            else:
                char_pos += len(f"{role.capitalize()}: {content}\n\n")

        # Map character positions to token indices
        assistant_token_positions = []
        for char_end in assistant_end_chars:
            token_idx = len(offsets) - 1  # Default to last token
            for idx, (start, end) in enumerate(offsets):
                if start <= char_end < end or (idx > 0 and end > char_end):
                    token_idx = idx
                    break
            assistant_token_positions.append(min(token_idx, len(offsets) - 1))

        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # outputs.logits shape: (batch_size, seq_len, num_labels)
            # num_labels = 1 for this model, so squeeze
            logits = outputs.logits.squeeze(-1)  # (batch_size, seq_len)

        # Extract rewards at assistant turn boundaries
        rewards = []
        logits_np = logits[0].float().cpu().numpy()

        for token_pos in assistant_token_positions:
            if 0 <= token_pos < len(logits_np):
                rewards.append(logits_np[token_pos])
            else:
                rewards.append(0.0)

        return np.array(rewards, dtype=np.float32)


# ============================================================================
# Main Processing Loop
# ============================================================================

def discover_trajectories(
    trajectories_dir: Path,
    agents: Optional[List[str]] = None,
) -> List[Tuple[str, str, Path]]:
    """
    Discover all (agent, task, filepath) tuples.

    Returns:
        List of (agent_id, task_id, filepath)
    """
    results = []

    if agents is None:
        agents = sorted([
            d.name for d in trajectories_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    for agent in agents:
        agent_dir = trajectories_dir / agent
        if not agent_dir.is_dir():
            continue

        for json_file in agent_dir.glob("*.json"):
            task_id = json_file.stem
            # Skip metadata files (not actual trajectories)
            if task_id.startswith("_"):
                continue
            results.append((agent, task_id, json_file))

    return results


def process_trajectories(
    trajectories_dir: Path,
    output_dir: Path,
    shard_id: int = 0,
    num_shards: int = 1,
    limit: Optional[int] = None,
    skip_existing: bool = True,
):
    """
    Process all trajectories and save per-step rewards.

    Args:
        trajectories_dir: Directory containing agent trajectory folders
        output_dir: Where to save reward files
        shard_id: Which shard to process (for parallelization)
        num_shards: Total number of shards
        limit: Maximum number of trajectories to process
        skip_existing: Skip trajectories that already have output files
    """
    # Discover all trajectories
    print(f"Discovering trajectories from {trajectories_dir}...")
    all_trajs = discover_trajectories(trajectories_dir)
    print(f"Found {len(all_trajs)} total trajectories")

    # Shard the work
    if num_shards > 1:
        all_trajs = sorted(all_trajs, key=lambda x: (x[0], x[1]))  # Sort for determinism
        all_trajs = [t for i, t in enumerate(all_trajs) if i % num_shards == shard_id]
        print(f"Processing shard {shard_id}/{num_shards}: {len(all_trajs)} trajectories")

    if limit:
        all_trajs = all_trajs[:limit]
        print(f"Limited to {limit} trajectories")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    critic = CriticModel()

    # Process each trajectory
    processed = 0
    skipped = 0
    errors = 0

    for agent, task_id, filepath in tqdm(all_trajs, desc="Computing rewards"):
        # Check if output already exists
        agent_output_dir = output_dir / agent
        output_file = agent_output_dir / f"{task_id}.npz"

        if skip_existing and output_file.exists():
            skipped += 1
            continue

        # Load trajectory
        traj = load_trajectory(filepath)
        if traj is None:
            errors += 1
            continue

        # Compute rewards
        try:
            rewards = critic.get_per_step_rewards(traj.messages)

            # Save results
            agent_output_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                output_file,
                rewards=rewards,
                task_id=task_id,
                agent_id=agent,
                resolved=traj.resolved,
                num_steps=len(rewards),
                model_name=critic.model_name,
            )
            processed += 1

        except Exception as e:
            print(f"Error processing {agent}/{task_id}: {e}")
            errors += 1
            continue

    print(f"\nCompleted:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-step rewards using OpenHands Critic Model"
    )

    parser.add_argument(
        "--trajectories_dir",
        type=Path,
        default=Path("trajectory_data/unified_trajs"),
        help="Directory containing agent trajectory folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/experiment_b/critic_rewards"),
        help="Directory to save reward files",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Which shard to process (0-indexed)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for parallelization",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of trajectories to process",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recompute even if output file exists",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without running",
    )

    args = parser.parse_args()

    if args.dry_run:
        all_trajs = discover_trajectories(args.trajectories_dir)
        print(f"Would process {len(all_trajs)} trajectories")
        print(f"Shard {args.shard_id}/{args.num_shards}")
        if args.num_shards > 1:
            shard_trajs = [t for i, t in enumerate(sorted(all_trajs))
                          if i % args.num_shards == args.shard_id]
            print(f"This shard: {len(shard_trajs)} trajectories")
        return

    process_trajectories(
        trajectories_dir=args.trajectories_dir,
        output_dir=args.output_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
