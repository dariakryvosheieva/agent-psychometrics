#!/usr/bin/env python3
"""
Compute embeddings for agent trajectories on SWE-bench tasks.

Adapts Daria's prior embedding framework to work with trajectory data for
Experiment B posterior difficulty prediction.

Usage:
    python -m experiment_b.compute_trajectory_embeddings \
        --trajectories_dir trajectory_data/unified_trajs \
        --output_dir chris_output/experiment_b/trajectory_embeddings \
        --backbone "Qwen/Qwen3-VL-8B-Instruct" \
        --content_type full \
        --instruction_type difficulty
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import Ridge
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================================
# Constants and Configuration
# ============================================================================

INSTRUCTION_TEMPLATES = {
    "difficulty": (
        "How difficult is the above task for a coding agent? "
        "Please output one floating-point number from 0 (very easy) to 1 (very hard). "
        "Your difficulty score:\n"
    ),
    "residual": (
        "Based on the agent's trajectory above, how much harder or easier was this task "
        "than initially expected? Output a number from -1 (much easier) to 1 (much harder). "
        "Your residual difficulty score:\n"
    ),
    "progress": (
        "Based on the trajectory above, rate the agent's progress toward solving the task. "
        "Output a number from 0 (completely stuck, no progress) to 1 (nearly solved). "
        "Your progress score:\n"
    ),
    "closeness": (
        "How close was the agent to solving this task based on its trajectory? "
        "Output a number from 0 (very far from solution) to 1 (almost solved). "
        "Your closeness score:\n"
    ),
}

ContentType = Literal["full", "condensed", "failure_focused", "no_solution"]
InstructionType = Literal["difficulty", "residual", "progress", "closeness"]


# ============================================================================
# Text Processing Utilities
# ============================================================================

def _sanitize_text(s: str) -> str:
    """Replace ASCII control characters with spaces."""
    return "".join(
        (" " if (ord(ch) < 32 and ch not in ("\n", "\t")) else ch)
        for ch in (s or "")
    )


def normalize_swebench_item_id(raw_item_id: str) -> str:
    """Normalize SWE-bench item IDs to consistent format."""
    s = str(raw_item_id or "").strip()
    if s.startswith("instance_"):
        s = s[len("instance_"):]
    # Remove -v... suffixes
    s = re.sub(r"-v.*$", "", s)
    return s.strip()


# ============================================================================
# Trajectory Processing
# ============================================================================

@dataclass
class TrajectoryRecord:
    """Container for trajectory data."""
    task_id: str
    agent_id: str
    resolved: bool
    messages: List[Dict]
    problem_statement: str
    solution: str


def load_trajectory(trajectories_dir: Path, agent: str, task_id: str) -> Optional[Dict]:
    """Load a single trajectory JSON file."""
    traj_path = trajectories_dir / agent / f"{task_id}.json"
    if not traj_path.exists():
        return None
    try:
        with open(traj_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _normalize_content(content) -> str:
    """Normalize message content to string.

    Some agents (OpenHands, codesweep) store content as a list of content blocks
    like [{'type': 'text', 'text': '...'}] instead of a plain string.
    """
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", str(item)))
            else:
                text_parts.append(str(item))
        return " ".join(text_parts)
    return str(content) if content else ""


def extract_trajectory_text(messages: List[Dict], max_chars: int = 50000) -> str:
    """Extract text content from trajectory messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_chars: Maximum characters to include (truncates from start)

    Returns:
        Combined trajectory text
    """
    parts = []
    total_chars = 0

    # Skip system prompt (usually first message)
    msgs = messages[1:] if messages and messages[0].get("role") == "system" else messages

    for msg in msgs:
        role = msg.get("role", "unknown")
        content = _normalize_content(msg.get("content", ""))

        # Add role prefix for clarity
        if role == "user":
            prefix = "[Environment Output]\n"
        elif role == "assistant":
            prefix = "[Agent]\n"
        else:
            prefix = f"[{role}]\n"

        part = prefix + content + "\n\n"
        parts.append(part)
        total_chars += len(part)

    full_text = "".join(parts)

    # Truncate from the beginning if too long (keep the end of trajectory)
    if len(full_text) > max_chars:
        full_text = "... [trajectory truncated] ...\n" + full_text[-max_chars:]

    return full_text


def extract_errors_only(messages: List[Dict], max_chars: int = 20000) -> str:
    """Extract only error-related content from trajectory.

    Looks for tracebacks, error messages, and failed assertions.
    """
    error_patterns = [
        r"Traceback \(most recent call last\):.*?(?=\n[^\s]|\Z)",
        r"(?:Error|Exception|AssertionError|ValueError|TypeError|KeyError|AttributeError|IndexError):[^\n]+",
        r"FAILED[^\n]+",
        r"FAIL:[^\n]+",
    ]

    errors = []
    for msg in messages:
        content = _normalize_content(msg.get("content", ""))
        for pattern in error_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            errors.extend(matches)

    error_text = "\n---\n".join(errors)
    if len(error_text) > max_chars:
        error_text = error_text[-max_chars:]

    return error_text if error_text else "[No errors found in trajectory]"


def summarize_trajectory(messages: List[Dict], max_chars: int = 15000) -> str:
    """Create a condensed summary of the trajectory.

    Includes: first few messages, last few messages, and any errors.
    """
    # Skip system prompt
    msgs = messages[1:] if messages and messages[0].get("role") == "system" else messages

    if not msgs:
        return "[Empty trajectory]"

    parts = []

    # First 3 exchanges (roughly task description and initial attempts)
    first_n = min(6, len(msgs))
    parts.append("=== Initial Messages ===\n")
    for msg in msgs[:first_n]:
        role = msg.get("role", "?")
        content = _normalize_content(msg.get("content", ""))[:2000]
        parts.append(f"[{role}] {content}\n")

    # Last 3 exchanges (final state)
    if len(msgs) > first_n:
        parts.append("\n=== Final Messages ===\n")
        last_n = min(6, len(msgs) - first_n)
        for msg in msgs[-last_n:]:
            role = msg.get("role", "?")
            content = _normalize_content(msg.get("content", ""))[:2000]
            parts.append(f"[{role}] {content}\n")

    summary = "".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n... [summary truncated]"

    return summary


def format_embedding_input(
    *,
    problem_statement: str,
    solution: str,
    trajectory_text: str,
    instruction: str,
    content_type: ContentType,
) -> str:
    """Format the full embedding input based on content type.

    Args:
        problem_statement: Task description
        solution: Gold solution/patch
        trajectory_text: Processed trajectory content
        instruction: Instruction suffix
        content_type: How much content to include

    Returns:
        Formatted text for embedding
    """
    problem = _sanitize_text(problem_statement).strip()
    sol = _sanitize_text(solution).strip()
    traj = _sanitize_text(trajectory_text).strip()
    instr = _sanitize_text(instruction).strip()

    if content_type == "no_solution":
        # Generalization test: no solution provided
        return f"Task:\n{problem}\n\nAgent Trajectory:\n{traj}\n\n{instr}".strip()
    elif content_type == "full":
        return f"Task:\n{problem}\n\nSolution:\n{sol}\n\nAgent Trajectory:\n{traj}\n\n{instr}".strip()
    elif content_type == "condensed":
        return f"Task:\n{problem}\n\nSolution:\n{sol}\n\nTrajectory Summary:\n{traj}\n\n{instr}".strip()
    elif content_type == "failure_focused":
        return f"Task:\n{problem}\n\nSolution:\n{sol}\n\nErrors from Trajectory:\n{traj}\n\n{instr}".strip()
    else:
        raise ValueError(f"Unknown content_type: {content_type}")


# ============================================================================
# Model Loading (adapted from predict_question_difficulty.py)
# ============================================================================

def _try_load_model_class(backbone: str, *, trust_remote_code: bool, model_kwargs: dict):
    """Load a HF model supporting both LMs and VLMs."""
    # Compatibility shim for transformers version differences
    # Some remote-code checkpoints expect symbols that may have been renamed
    try:
        import transformers.activations as _act
        if not hasattr(_act, "PytorchGELUTanh") and hasattr(_act, "GELUTanh"):
            _act.PytorchGELUTanh = _act.GELUTanh  # type: ignore[attr-defined]
    except Exception:
        pass

    errors = []

    # Try different auto classes in order of preference
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            backbone, trust_remote_code=trust_remote_code, **model_kwargs
        )
    except Exception as e:
        errors.append(("AutoModelForImageTextToText", e))

    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(
            backbone, trust_remote_code=trust_remote_code, **model_kwargs
        )
    except Exception as e:
        errors.append(("AutoModelForVision2Seq", e))

    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            backbone, trust_remote_code=trust_remote_code, **model_kwargs
        )
    except Exception as e:
        errors.append(("AutoModelForCausalLM", e))

    try:
        return AutoModel.from_pretrained(
            backbone, trust_remote_code=trust_remote_code, **model_kwargs
        )
    except Exception as e:
        errors.append(("AutoModel", e))

    msg = "Failed to load model:\n" + "\n".join(
        [f"- {name}: {err}" for name, err in errors]
    )
    raise RuntimeError(msg)


def _select_text_submodel(model: torch.nn.Module) -> torch.nn.Module:
    """For VLM wrappers, prefer the language/text tower."""
    for attr in ("language_model", "text_model"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Module):
            return m
    m = getattr(model, "model", None)
    if isinstance(m, torch.nn.Module) and hasattr(m, "get_input_embeddings"):
        return m
    return model


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool embeddings using last token position."""
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    return last_hidden_state.gather(dim=1, index=idx).squeeze(1)


def _get_hidden_states_tuple(outputs):
    """Best-effort extraction of hidden states tuple from HF outputs."""
    for attr in ("hidden_states", "encoder_hidden_states", "decoder_hidden_states"):
        if hasattr(outputs, attr):
            hs = getattr(outputs, attr)
            if hs is not None:
                return hs
    return None


def _extract_hidden_state(outputs, *, embedding_layer: int) -> torch.Tensor:
    """Extract hidden state from model outputs.

    If embedding_layer == -1, uses the last layer by default.
    """
    layer = int(embedding_layer)

    # Fast path: if caller wants "last" and last_hidden_state is provided, use it
    if layer == -1 and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if layer == -1 and hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state

    # Try hidden_states tuple
    hs = _get_hidden_states_tuple(outputs)
    if hs is not None:
        try:
            return hs[layer]
        except Exception as e:
            raise RuntimeError(
                f"Requested embedding_layer={layer}, but model returned {len(hs)} hidden_states entries. "
                f"Try a value in [-{len(hs)}, {len(hs)-1}] or use --embedding_layer -1 for last."
            ) from e

    # Fallback for last layer if hidden_states weren't present
    if layer == -1 and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if layer == -1 and hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state

    raise RuntimeError(
        "Model outputs did not expose hidden_states needed for selecting a layer. "
        "Try using --embedding_layer -1, and ensure the model supports output_hidden_states=True."
    )


# ============================================================================
# Embedding Computation
# ============================================================================

class TrajectoryEmbedder:
    """Compute embeddings for trajectory data."""

    def __init__(
        self,
        backbone: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_length: int = 8192,
        batch_size: int = 1,
        embedding_layer: int = -1,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.backbone = backbone
        self.max_length = max_length
        self.batch_size = batch_size
        self.embedding_layer = embedding_layer

        # Initialize model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Determine dtype
        if torch_dtype == "auto":
            dtype_arg = "auto"
        elif torch_dtype in ("float16", "fp16"):
            dtype_arg = torch.float16
        elif torch_dtype in ("bfloat16", "bf16"):
            dtype_arg = torch.bfloat16
        else:
            dtype_arg = torch.float32

        model_kwargs = {"torch_dtype": dtype_arg}
        if device_map and device_map not in ("", "none"):
            model_kwargs["device_map"] = device_map
        if attn_implementation and attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = _try_load_model_class(
            backbone, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs
        )
        self.model.eval()

        # Explicit device placement if device_map not used
        if device_map in ("", "none", None):
            self.model.to(self.device)

        self.text_model = _select_text_submodel(self.model)

        # Disable KV caching
        for m in (self.model, self.text_model):
            cfg = getattr(m, "config", None)
            if cfg is not None and hasattr(cfg, "use_cache"):
                try:
                    cfg.use_cache = False
                except Exception:
                    pass

        try:
            self.embed_device = self.text_model.get_input_embeddings().weight.device
        except Exception:
            self.embed_device = self.device

        self.embedding_dim: Optional[int] = None

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        enc = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.embed_device)
        attention_mask = enc["attention_mask"].to(self.embed_device)

        with torch.no_grad():
            fwd_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

            try:
                sig = inspect.signature(self.text_model.forward)
                if "use_cache" in sig.parameters:
                    fwd_kwargs["use_cache"] = False
            except Exception:
                fwd_kwargs["use_cache"] = False

            out = self.text_model(**fwd_kwargs)
            h = _extract_hidden_state(out, embedding_layer=self.embedding_layer)
            pooled = last_token_pool(h, attention_mask)
            embedding = pooled.detach().float().cpu().numpy()[0]

        self.embedding_dim = embedding.shape[0]
        return embedding.astype(np.float32)


def load_swebench_tasks() -> Dict[str, Dict]:
    """Load SWE-bench Verified task data."""
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {
        normalize_swebench_item_id(ex["instance_id"]): {
            "problem_statement": ex["problem_statement"],
            "patch": ex["patch"],
        }
        for ex in ds
    }


def compute_trajectory_embeddings(
    trajectories_dir: Path,
    output_dir: Path,
    agents: List[str],
    task_ids: List[str],
    content_type: ContentType,
    instruction_type: InstructionType,
    embedder: TrajectoryEmbedder,
    shard_id: int = 0,
    num_shards: int = 1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute embeddings for all (task, agent) pairs.

    Args:
        trajectories_dir: Directory containing agent trajectory folders
        output_dir: Where to save embeddings
        agents: List of agent IDs to process
        task_ids: List of task IDs to process
        content_type: How to format trajectory content
        instruction_type: Which instruction suffix to use
        embedder: TrajectoryEmbedder instance
        shard_id: Which shard to process (for parallelization)
        num_shards: Total number of shards

    Returns:
        Dict mapping task_id -> agent_id -> embedding
    """
    # Load task data
    print("Loading SWE-bench task data...")
    tasks = load_swebench_tasks()

    instruction = INSTRUCTION_TEMPLATES[instruction_type]

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shard agents for parallel processing
    if num_shards > 1:
        agents = sorted(agents)
        agents = [a for i, a in enumerate(agents) if i % num_shards == shard_id]
        print(f"Processing shard {shard_id}/{num_shards}: {len(agents)} agents")

    results: Dict[str, Dict[str, np.ndarray]] = {}

    total_pairs = len(agents) * len(task_ids)
    processed = 0
    skipped = 0

    with tqdm(total=total_pairs, desc="Computing embeddings") as pbar:
        for agent in agents:
            agent_dir = output_dir / agent
            agent_dir.mkdir(exist_ok=True)

            for task_id in task_ids:
                pbar.update(1)

                # Check if already computed
                output_file = agent_dir / f"{task_id}.npz"
                if output_file.exists():
                    skipped += 1
                    continue

                # Load trajectory
                traj = load_trajectory(trajectories_dir, agent, task_id)
                if traj is None:
                    continue

                # Get task data
                task_data = tasks.get(task_id)
                if task_data is None:
                    continue

                # Process trajectory based on content type
                messages = traj.get("messages", [])
                if content_type == "condensed":
                    trajectory_text = summarize_trajectory(messages)
                elif content_type == "failure_focused":
                    trajectory_text = extract_errors_only(messages)
                else:
                    trajectory_text = extract_trajectory_text(messages)

                # Format embedding input
                text = format_embedding_input(
                    problem_statement=task_data["problem_statement"],
                    solution=task_data["patch"],
                    trajectory_text=trajectory_text,
                    instruction=instruction,
                    content_type=content_type,
                )

                # Compute embedding
                try:
                    embedding = embedder.embed_text(text)

                    # Save immediately with full metadata (matching Daria's format)
                    np.savez_compressed(
                        output_file,
                        embedding=embedding,
                        task_id=task_id,
                        agent_id=agent,
                        content_type=content_type,
                        instruction_type=instruction_type,
                        instruction=instruction,
                        resolved=traj.get("resolved", False),
                        backbone=embedder.backbone,
                        max_length=embedder.max_length,
                        embedding_dim=embedder.embedding_dim,
                        embedding_layer=embedder.embedding_layer,
                    )

                    # Track results
                    if task_id not in results:
                        results[task_id] = {}
                    results[task_id][agent] = embedding
                    processed += 1

                except Exception as e:
                    print(f"Error processing {agent}/{task_id}: {e}")
                    continue

    print(f"\nCompleted: {processed} embeddings computed, {skipped} skipped (already exist)")
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory embeddings for Experiment B"
    )

    # Paths
    parser.add_argument(
        "--trajectories_dir",
        type=Path,
        default=Path("trajectory_data/unified_trajs"),
        help="Directory containing agent trajectory folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("chris_output/experiment_b/trajectory_embeddings"),
        help="Directory to save embeddings",
    )

    # Model configuration
    parser.add_argument(
        "--backbone",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model to use for embeddings",
    )
    parser.add_argument(
        "--embedding_layer",
        type=int,
        default=-1,
        help="Which layer to extract embeddings from (-1 for last)",
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
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        help="Attention implementation (e.g., auto, flash_attention_2)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map (e.g. auto). Use 'none' to force single-device .to(device).",
    )

    # Content configuration
    parser.add_argument(
        "--content_type",
        type=str,
        default="full",
        choices=["full", "condensed", "failure_focused", "no_solution"],
        help="How to format trajectory content",
    )
    parser.add_argument(
        "--instruction_type",
        type=str,
        default="difficulty",
        choices=["difficulty", "residual", "progress", "closeness"],
        help="Which instruction suffix to use",
    )

    # Parallelization
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Which shard to process (for parallel execution)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards",
    )

    # Filtering
    parser.add_argument(
        "--agents",
        type=str,
        nargs="*",
        default=None,
        help="Specific agents to process (default: all)",
    )
    parser.add_argument(
        "--limit_agents",
        type=int,
        default=None,
        help="Limit number of agents to process",
    )

    args = parser.parse_args()

    # Discover agents and tasks
    print(f"Discovering agents from {args.trajectories_dir}...")
    if args.agents:
        agents = args.agents
    else:
        agents = sorted([
            d.name for d in args.trajectories_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    if args.limit_agents:
        agents = agents[:args.limit_agents]

    print(f"Found {len(agents)} agents")

    # Get task IDs from SWE-bench
    print("Loading task IDs from SWE-bench...")
    tasks = load_swebench_tasks()
    task_ids = list(tasks.keys())
    print(f"Found {len(task_ids)} tasks")

    # Initialize embedder
    print(f"Loading model: {args.backbone}")
    embedder = TrajectoryEmbedder(
        backbone=args.backbone,
        max_length=args.max_length,
        batch_size=args.batch_size,
        embedding_layer=args.embedding_layer,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

    # Compute embeddings
    output_subdir = args.output_dir / f"{args.content_type}_{args.instruction_type}"

    compute_trajectory_embeddings(
        trajectories_dir=args.trajectories_dir,
        output_dir=output_subdir,
        agents=agents,
        task_ids=task_ids,
        content_type=args.content_type,
        instruction_type=args.instruction_type,
        embedder=embedder,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    print("Done!")


if __name__ == "__main__":
    main()
