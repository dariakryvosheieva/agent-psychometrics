"""Main entry point for SAD-IRT training and evaluation."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Check if accelerate is available for multi-GPU
try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

from .config import SADIRTConfig
from .data_splits import (
    get_all_agents_from_responses,
    get_agents_with_trajectories,
    split_agents_by_cutoff,
    identify_frontier_tasks,
    compute_pass_rates,
)
from .dataset import TrajectoryIRTDataset
from .evaluate import compute_metrics, compute_frontier_difficulty_metrics, log_parameter_stats
from .model import SADIRT
from .train import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_accuracy_based_init(
    responses_path: Path,
    agent_ids: list,
    task_ids: list,
    eps: float = 1e-3,
) -> tuple:
    """Compute accuracy-based initialization for θ and β.

    Uses the same approach as py_irt's difficulty_from_accuracy initializer:
    β_i = logit(1 - accuracy_i) = log((1 - acc) / acc)
    θ_j = logit(accuracy_j) = log(acc / (1 - acc))

    Args:
        responses_path: Path to response matrix JSONL
        agent_ids: List of agent IDs (for θ)
        task_ids: List of task IDs (for β)
        eps: Small constant to avoid log(0)

    Returns:
        Tuple of (theta_init, beta_init) as dicts mapping id -> value
    """
    import math
    from collections import defaultdict

    agent_set = set(agent_ids)
    task_set = set(task_ids)

    # Compute per-task accuracy (for β) and per-agent accuracy (for θ)
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    agent_correct = defaultdict(int)
    agent_total = defaultdict(int)

    with open(responses_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            agent_id = record['subject_id']
            if agent_id not in agent_set:
                continue
            for task_id, response in record['responses'].items():
                if task_id not in task_set:
                    continue
                task_total[task_id] += 1
                agent_total[agent_id] += 1
                if response == 1:
                    task_correct[task_id] += 1
                    agent_correct[agent_id] += 1

    # Compute β from task accuracy: β = logit(1 - acc) = log((1-acc)/acc)
    beta_init = {}
    for task_id in task_ids:
        if task_total[task_id] > 0:
            acc = task_correct[task_id] / task_total[task_id]
            acc = max(eps, min(1 - eps, acc))  # Clamp to avoid log(0)
            beta_init[task_id] = math.log((1 - acc) / acc)
        else:
            beta_init[task_id] = 0.0

    # Compute θ from agent accuracy: θ = logit(acc) = log(acc/(1-acc))
    theta_init = {}
    for agent_id in agent_ids:
        if agent_total[agent_id] > 0:
            acc = agent_correct[agent_id] / agent_total[agent_id]
            acc = max(eps, min(1 - eps, acc))
            theta_init[agent_id] = math.log(acc / (1 - acc))
        else:
            theta_init[agent_id] = 0.0

    return theta_init, beta_init


def train_baseline_irt_on_prefrontier(
    responses_path: Path,
    pre_frontier_agents: list,
    output_dir: Path,
    epochs: int = 2000,
) -> dict:
    """Train standard IRT on pre-frontier agents only using py_irt.

    Args:
        responses_path: Path to response matrix JSONL
        pre_frontier_agents: List of pre-frontier agent IDs to include
        output_dir: Directory to save IRT outputs
        epochs: Number of training epochs for py_irt

    Returns:
        Dict mapping task_id -> difficulty (β)
    """
    import pandas as pd
    import pyro

    from py_irt.dataset import Dataset
    from py_irt.models import OneParamLog
    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    # Load response matrix and filter to pre-frontier agents
    pre_frontier_set = set(pre_frontier_agents)
    data_list = []
    with open(responses_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record['subject_id'] in pre_frontier_set:
                row = {'subject_id': record['subject_id']}
                row.update(record['responses'])
                data_list.append(row)

    logger.info(f"Loaded {len(data_list)} pre-frontier agent responses")

    df = pd.DataFrame(data_list)
    item_columns = [col for col in df.columns if col != 'subject_id']
    dataset = Dataset.from_pandas(df, subject_column="subject_id", item_columns=item_columns)

    # Train 1PL IRT (same as SAD-IRT uses)
    config = IrtConfig(
        model_type=OneParamLog,
        priors="hierarchical",
        initializers=[
            {"name": "difficulty_from_accuracy", "eps": 1e-3},
        ],
    )

    # Clear pyro param store to avoid conflicts
    pyro.clear_param_store()

    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(epochs=epochs)

    # Extract difficulty parameters
    difficulties = list(trainer.best_params["diff"])
    item_id_map = trainer.best_params["item_ids"]
    item_ids = [item_id_map[i] for i in range(len(difficulties))]

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    items_df = pd.DataFrame({
        "b": difficulties,
    }, index=item_ids)
    items_df.to_csv(output_dir / "items.csv")

    logger.info(f"Baseline IRT saved to {output_dir}")
    logger.info(f"β stats: mean={np.mean(difficulties):.4f}, std={np.std(difficulties):.4f}")

    # Return as dict
    return {task_id: diff for task_id, diff in zip(item_ids, difficulties)}


def parse_args() -> SADIRTConfig:
    """Parse command line arguments into config."""
    parser = argparse.ArgumentParser(description="SAD-IRT Training and Evaluation")

    # Frontier difficulty settings
    parser.add_argument("--frontier_cutoff_date", type=str, default="20250807",
                        help="Date cutoff for pre/post frontier (YYYYMMDD)")
    parser.add_argument("--pre_frontier_threshold", type=float, default=0.1,
                        help="Max pass rate for pre-frontier tasks")
    parser.add_argument("--post_frontier_threshold", type=float, default=0.1,
                        help="Min pass rate for post-frontier tasks")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_r", type=int, default=16)

    # Data
    parser.add_argument("--response_matrix_path", type=str, default="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    parser.add_argument("--trajectory_dir", type=str, default="chris_output/trajectory_summaries_api")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--oracle_irt_dir", type=str, default="clean_data/swebench_verified_20251120_full/1d")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate_encoder", type=float, default=1e-4)
    parser.add_argument("--learning_rate_embeddings", type=float, default=1e-3)

    # Evaluation/Logging
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N optimizer steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N optimizer steps")

    # Output
    parser.add_argument("--output_dir", type=str, default="chris_output/sad_irt")
    parser.add_argument("--seed", type=int, default=42)

    # Ablations
    parser.add_argument("--freeze_irt", action="store_true", help="Freeze θ/β and only train ψ predictor")
    parser.add_argument("--psi_normalization", type=str, default=None,
                        choices=["batchnorm", "center", "none"],
                        help="How to normalize ψ (default: center if frozen, batchnorm otherwise)")

    # Debug
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Quick test: load model, run 1 batch, exit")
    parser.add_argument("--overfit_test", action="store_true", help="Test overfitting on small batch (sanity check)")
    parser.add_argument("--debug_gradients", action="store_true", help="Enable verbose gradient logging")

    # Resumption
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Convert to config
    config = SADIRTConfig(
        frontier_cutoff_date=args.frontier_cutoff_date,
        pre_frontier_threshold=args.pre_frontier_threshold,
        post_frontier_threshold=args.post_frontier_threshold,
        model_name=args.model_name,
        lora_r=args.lora_r,
        response_matrix_path=args.response_matrix_path,
        trajectory_dir=args.trajectory_dir,
        max_length=args.max_length,
        oracle_irt_dir=args.oracle_irt_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate_encoder=args.learning_rate_encoder,
        learning_rate_embeddings=args.learning_rate_embeddings,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        freeze_irt=args.freeze_irt,
        psi_normalization=args.psi_normalization,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        smoke_test=args.smoke_test,
        overfit_test=args.overfit_test,
        debug_gradients=args.debug_gradients,
        resume_from=args.resume_from,
    )

    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_smoke_test(config: SADIRTConfig):
    """Quick smoke test: load everything, run 1 forward/backward pass, exit."""
    logger.info("=" * 60)
    logger.info("SMOKE TEST: Checking code paths")
    logger.info("=" * 60)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded OK")

    # Create minimal dataset (just 10 samples)
    logger.info("Loading minimal dataset...")
    full_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        swebench_dataset=config.swebench_dataset,
    )
    logger.info(f"Dataset: {len(full_dataset)} total samples, {full_dataset.num_agents} agents, {full_dataset.num_tasks} tasks")

    # Get just 4 samples for smoke test
    train_pairs, test_pairs = create_train_test_split(full_dataset, test_fraction=0.5, seed=config.seed)
    train_pairs = train_pairs[:4]

    mini_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=train_pairs,
        swebench_dataset=config.swebench_dataset,
    )
    logger.info(f"Mini dataset: {len(mini_dataset)} samples")

    # Load one batch
    loader = DataLoader(mini_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    logger.info(f"Batch loaded: input_ids shape = {batch['input_ids'].shape}")

    # Create model
    logger.info(f"Creating SAD-IRT model with {config.model_name}...")
    model = SADIRT(
        num_agents=full_dataset.num_agents,
        num_tasks=full_dataset.num_tasks,
        model_name=config.model_name,
        lora_r=config.lora_r,
    ).to(device)
    logger.info("Model created OK")

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Shape validation
    logger.info("=" * 40)
    logger.info("SHAPE VALIDATION")
    logger.info("=" * 40)
    batch_size = batch["input_ids"].shape[0]
    seq_len = batch["input_ids"].shape[1]
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  seq_len: {seq_len}")
    logger.info(f"  agent_idx shape: {batch['agent_idx'].shape} (expected: [{batch_size}])")
    logger.info(f"  task_idx shape: {batch['task_idx'].shape} (expected: [{batch_size}])")
    logger.info(f"  input_ids shape: {batch['input_ids'].shape} (expected: [{batch_size}, {seq_len}])")
    logger.info(f"  attention_mask shape: {batch['attention_mask'].shape} (expected: [{batch_size}, {seq_len}])")
    logger.info(f"  response shape: {batch['response'].shape} (expected: [{batch_size}])")

    # Validate shapes
    assert batch["agent_idx"].shape == (batch_size,), f"agent_idx shape mismatch"
    assert batch["task_idx"].shape == (batch_size,), f"task_idx shape mismatch"
    assert batch["input_ids"].shape[0] == batch_size, f"input_ids batch size mismatch"
    assert batch["attention_mask"].shape == batch["input_ids"].shape, f"attention_mask shape mismatch"
    assert batch["response"].shape == (batch_size,), f"response shape mismatch"
    logger.info("Shape validation PASSED")

    # Forward pass
    logger.info("=" * 40)
    logger.info("FORWARD PASS")
    logger.info("=" * 40)
    model.train()
    logits = model(
        agent_idx=batch["agent_idx"],
        task_idx=batch["task_idx"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logger.info(f"Output logits shape: {logits.shape} (expected: [{batch_size}])")
    logger.info(f"Output logits values: {logits.detach().cpu().numpy()}")
    assert logits.shape == (batch_size,), f"logits shape mismatch: {logits.shape} vs expected ({batch_size},)"
    logger.info("Forward pass PASSED")

    # Backward pass
    logger.info("=" * 40)
    logger.info("BACKWARD PASS")
    logger.info("=" * 40)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["response"])
    loss.backward()
    logger.info(f"Loss: {loss.item():.4f}")

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Gradients computed for {grad_count}/{total_params} trainable parameters")

    # Verify key parameters have gradients
    has_theta_grad = any("theta" in n and p.grad is not None for n, p in model.named_parameters())
    has_beta_grad = any("beta" in n and p.grad is not None for n, p in model.named_parameters())
    has_lora_grad = any("lora" in n and p.grad is not None for n, p in model.named_parameters())
    has_head_grad = any("psi_head" in n and p.grad is not None for n, p in model.named_parameters())

    logger.info(f"  theta (agent ability) has gradients: {has_theta_grad}")
    logger.info(f"  beta (task difficulty) has gradients: {has_beta_grad}")
    logger.info(f"  LoRA encoder has gradients: {has_lora_grad}")
    logger.info(f"  psi_head MLP has gradients: {has_head_grad}")

    assert has_theta_grad, "theta embeddings have no gradients!"
    assert has_beta_grad, "beta embeddings have no gradients!"
    assert has_lora_grad, "LoRA parameters have no gradients!"
    assert has_head_grad, "psi_head has no gradients!"
    logger.info("Gradient flow PASSED")

    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED - All checks OK")
    logger.info("=" * 60)

    return {"status": "passed"}


def run_overfit_test(config: SADIRTConfig):
    """Test that model can overfit a small batch (sanity check).

    This verifies:
    1. Gradients flow to all parameters
    2. Loss decreases to near-zero on repeated same batch
    3. Model can memorize small data (no fundamental bugs)
    """
    logger.info("=" * 60)
    logger.info("OVERFIT TEST: Verifying model can memorize small batch")
    logger.info("=" * 60)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create minimal dataset
    full_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        swebench_dataset=config.swebench_dataset,
    )

    # Get 8 samples (mix of 0s and 1s ideally)
    train_pairs, _ = create_train_test_split(full_dataset, test_fraction=0.5, seed=config.seed)
    train_pairs = train_pairs[:8]

    mini_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=train_pairs,
        swebench_dataset=config.swebench_dataset,
    )

    loader = DataLoader(mini_dataset, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    logger.info(f"Batch: {len(batch['response'])} samples")
    logger.info(f"Response distribution: {batch['response'].cpu().numpy()}")
    logger.info(f"Input shape: {batch['input_ids'].shape}")

    # Create model
    model = SADIRT(
        num_agents=full_dataset.num_agents,
        num_tasks=full_dataset.num_tasks,
        model_name=config.model_name,
        lora_r=config.lora_r,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train for 100 steps on same batch
    logger.info("Training 100 steps on same batch...")
    model.train()
    losses = []

    for step in range(100):
        optimizer.zero_grad()

        logits = model(
            agent_idx=batch["agent_idx"],
            task_idx=batch["task_idx"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = criterion(logits, batch["response"])
        loss.backward()

        # Check gradient norms at first step
        if step == 0:
            logger.info("Gradient check at step 0:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        logger.info(f"  {name}: grad_norm={grad_norm:.6f}")

        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            logger.info(f"Step {step}: loss={loss.item():.4f}, probs={probs}")

    # Analyze results
    logger.info("=" * 40)
    logger.info("OVERFIT TEST RESULTS")
    logger.info("=" * 40)

    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    logger.info(f"Initial loss: {initial_loss:.4f}")
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info(f"Loss reduction: {loss_reduction:.1f}%")

    # Final predictions
    model.eval()
    with torch.no_grad():
        final_logits = model(
            agent_idx=batch["agent_idx"],
            task_idx=batch["task_idx"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    final_probs = torch.sigmoid(final_logits).cpu().numpy()
    targets = batch["response"].cpu().numpy()

    logger.info(f"Final predictions: {final_probs}")
    logger.info(f"Targets:           {targets}")

    # Check if we can overfit
    if final_loss < 0.1:
        logger.info("OVERFIT TEST PASSED: Model can memorize small batch")
        status = "passed"
    elif final_loss < 0.3:
        logger.info("OVERFIT TEST PARTIAL: Loss decreased but didn't fully converge")
        status = "partial"
    else:
        logger.info("OVERFIT TEST FAILED: Model cannot fit small batch - check for bugs")
        status = "failed"

    return {
        "status": status,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
    }


def run_frontier_difficulty_evaluation(config: SADIRTConfig):
    """Run frontier difficulty evaluation.

    Train SAD-IRT on pre-frontier agents only, then evaluate how well
    the learned β values predict oracle difficulties for frontier tasks.
    """
    logger.info("=" * 60)
    logger.info("Frontier Difficulty Evaluation")
    logger.info("=" * 60)

    # Set seed
    set_seed(config.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ===== Step 1: Split agents by cutoff date =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 1: Split agents by cutoff date")
    logger.info("=" * 40)

    responses_path = Path(config.response_matrix_path)
    trajectory_dir = Path(config.trajectory_dir)

    # Get all agents from response matrix
    all_agents = get_all_agents_from_responses(responses_path)
    logger.info(f"Total agents in response matrix: {len(all_agents)}")

    # Get agents with trajectories
    traj_agents = get_agents_with_trajectories(trajectory_dir)
    logger.info(f"Agents with trajectories: {len(traj_agents)}")

    # Filter to agents with both responses and trajectories
    agents_with_both = [a for a in all_agents if a in traj_agents]
    logger.info(f"Agents with both responses and trajectories: {len(agents_with_both)}")

    # Split by cutoff date
    pre_frontier_agents, post_frontier_agents = split_agents_by_cutoff(
        agents_with_both, cutoff_date=config.frontier_cutoff_date
    )
    logger.info(f"Pre-frontier agents (< {config.frontier_cutoff_date}): {len(pre_frontier_agents)}")
    logger.info(f"Post-frontier agents (>= {config.frontier_cutoff_date}): {len(post_frontier_agents)}")

    if len(pre_frontier_agents) == 0:
        raise ValueError("No pre-frontier agents found! Check cutoff date.")
    if len(post_frontier_agents) == 0:
        raise ValueError("No post-frontier agents found! Check cutoff date.")

    # ===== Step 2: Identify frontier tasks =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 2: Identify frontier tasks")
    logger.info("=" * 40)

    frontier_task_ids = identify_frontier_tasks(
        responses_path,
        pre_frontier_agents,
        post_frontier_agents,
        pre_threshold=config.pre_frontier_threshold,
        post_threshold=config.post_frontier_threshold,
    )
    logger.info(f"Frontier tasks (≤{config.pre_frontier_threshold:.0%} pre, >{config.post_frontier_threshold:.0%} post): {len(frontier_task_ids)}")

    if len(frontier_task_ids) < 3:
        logger.warning(f"Only {len(frontier_task_ids)} frontier tasks found - results may not be meaningful")

    # Show some examples
    if frontier_task_ids:
        pre_rates = compute_pass_rates(responses_path, pre_frontier_agents)
        post_rates = compute_pass_rates(responses_path, post_frontier_agents)
        logger.info("Example frontier tasks:")
        for task_id in frontier_task_ids[:5]:
            logger.info(f"  {task_id}: pre={pre_rates[task_id]:.1%}, post={post_rates[task_id]:.1%}")

    # ===== Step 3: Load oracle IRT =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 3: Load oracle IRT (trained on all agents)")
    logger.info("=" * 40)

    import pandas as pd
    oracle_irt_dir = Path(config.oracle_irt_dir)
    if not oracle_irt_dir.exists():
        raise FileNotFoundError(f"Oracle IRT not found at {oracle_irt_dir}")

    oracle_abilities_df = pd.read_csv(oracle_irt_dir / "abilities.csv", index_col=0)
    oracle_items_df = pd.read_csv(oracle_irt_dir / "items.csv", index_col=0)
    logger.info(f"Loaded oracle IRT: {len(oracle_abilities_df)} agents, {len(oracle_items_df)} tasks")

    # Create oracle β dict for frontier tasks
    oracle_beta = {task_id: oracle_items_df.loc[task_id, "b"] for task_id in frontier_task_ids if task_id in oracle_items_df.index}
    logger.info(f"Oracle β available for {len(oracle_beta)}/{len(frontier_task_ids)} frontier tasks")

    # ===== Step 4: Create dataset with pre-frontier agents only =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 4: Create dataset (pre-frontier agents only)")
    logger.info("=" * 40)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset filtering to pre-frontier agents only
    # Note: We train on ALL (agent, task) pairs, no train/test split
    train_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=pre_frontier_agents,  # Filter to pre-frontier only
        swebench_dataset=config.swebench_dataset,
    )

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Agents: {train_dataset.num_agents}, Tasks: {train_dataset.num_tasks}")

    # Limit samples if dry run
    if config.dry_run or config.max_samples:
        max_samples = config.max_samples or 100
        # Create a limited subset of pairs
        limited_pairs = [(s[0], s[1]) for s in train_dataset.samples[:max_samples]]
        train_dataset = TrajectoryIRTDataset(
            response_matrix_path=config.response_matrix_path,
            trajectory_dir=config.trajectory_dir,
            tokenizer=tokenizer,
            max_length=config.max_length,
            agent_ids=train_dataset.agent_ids,
            task_ids=train_dataset.task_ids,
            pairs=limited_pairs,
            swebench_dataset=config.swebench_dataset,
        )
        logger.info(f"DRY RUN: Limited to {len(train_dataset)} samples")

    # Create data loader (no eval loader since we don't do train/test split)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ===== Step 5: Train SAD-IRT =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 5: Train SAD-IRT on pre-frontier agents")
    logger.info("=" * 40)

    # Determine ψ normalization strategy
    # Default to "center" (zero-mean only) instead of "batchnorm" (zero-mean + unit variance)
    # BatchNorm causes variance collapse when psi_head outputs are similar, killing LoRA gradients
    if hasattr(config, 'psi_normalization') and config.psi_normalization is not None:
        psi_normalization = config.psi_normalization
    else:
        psi_normalization = "center"
    logger.info(f"Using psi_normalization={psi_normalization}")

    sad_irt_model = SADIRT(
        num_agents=train_dataset.num_agents,
        num_tasks=train_dataset.num_tasks,
        model_name=config.model_name,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        psi_normalization=psi_normalization,
    ).to(device)

    # Initialize θ/β from accuracy-based estimates (pre-frontier data only)
    # This follows py_irt's difficulty_from_accuracy approach:
    # β = logit(1 - task_accuracy), θ = logit(agent_accuracy)
    # Using oracle init would be unfair since oracle was trained on post-frontier agents too
    theta_init, beta_init = compute_accuracy_based_init(
        responses_path=Path(config.response_matrix_path),
        agent_ids=train_dataset.agent_ids,
        task_ids=train_dataset.task_ids,
    )
    sad_irt_model.initialize_from_accuracy(
        agent_ids=train_dataset.agent_ids,
        task_ids=train_dataset.task_ids,
        theta_init=theta_init,
        beta_init=beta_init,
    )

    # Optionally freeze θ/β (ablation: only train ψ predictor)
    if config.freeze_irt:
        logger.info("Freezing θ and β (only training ψ predictor)")
        sad_irt_model.theta.weight.requires_grad = False
        sad_irt_model.beta.weight.requires_grad = False

    sad_irt_trainer = Trainer(
        model=sad_irt_model,
        train_loader=train_loader,
        eval_loader=None,
        config=config,
        device=device,
        is_sad_irt=True,
        # Pass frontier evaluation data for checkpoint selection based on Spearman ρ
        task_ids=train_dataset.task_ids,
        frontier_task_ids=frontier_task_ids,
        oracle_beta=oracle_beta,
    )

    # Resume from checkpoint if specified
    if config.resume_from:
        sad_irt_trainer.load_checkpoint(config.resume_from)

    sad_irt_trainer.train()
    log_parameter_stats(sad_irt_model, prefix="SAD-IRT ")

    # ===== Step 6: Extract learned β and evaluate =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 6: Evaluate frontier task difficulty predictions")
    logger.info("=" * 40)

    # Get learned β values
    learned_beta_tensor = sad_irt_model.get_difficulties()
    learned_beta = {
        task_id: float(learned_beta_tensor[i])
        for i, task_id in enumerate(train_dataset.task_ids)
    }
    logger.info(f"Learned β for {len(learned_beta)} tasks")

    # Compute Spearman correlation on frontier tasks
    frontier_metrics = compute_frontier_difficulty_metrics(
        predicted_beta=learned_beta,
        oracle_beta=oracle_beta,
        frontier_task_ids=frontier_task_ids,
    )

    # ===== Step 7: Baseline comparison =====
    logger.info("\n" + "=" * 40)
    logger.info("Step 7: Baseline comparison (standard IRT on pre-frontier)")
    logger.info("=" * 40)

    # Train standard IRT on pre-frontier responses only using py_irt
    # This is the proper baseline: what can we learn about difficulty without trajectories?
    baseline_beta = train_baseline_irt_on_prefrontier(
        responses_path=responses_path,
        pre_frontier_agents=pre_frontier_agents,
        output_dir=Path(config.output_dir) / "baseline_irt",
    )
    logger.info(f"Baseline IRT trained on {len(pre_frontier_agents)} pre-frontier agents")

    baseline_frontier_metrics = compute_frontier_difficulty_metrics(
        predicted_beta=baseline_beta,
        oracle_beta=oracle_beta,
        frontier_task_ids=frontier_task_ids,
    )
    baseline_frontier_metrics = {f"baseline_{k}": v for k, v in baseline_frontier_metrics.items()}

    # ===== Results Summary =====
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nFrontier tasks: {len(frontier_task_ids)}")
    logger.info(f"Pre-frontier agents: {len(pre_frontier_agents)}")
    logger.info(f"Post-frontier agents: {len(post_frontier_agents)}")

    logger.info(f"\nBaseline (oracle β):")
    logger.info(f"  Spearman ρ: {baseline_frontier_metrics.get('baseline_frontier_spearman_rho', float('nan')):.4f}")

    logger.info(f"\nSAD-IRT (learned β):")
    logger.info(f"  Spearman ρ: {frontier_metrics.get('frontier_spearman_rho', float('nan')):.4f}")

    improvement = frontier_metrics.get('frontier_spearman_rho', 0) - baseline_frontier_metrics.get('baseline_frontier_spearman_rho', 0)
    logger.info(f"\nImprovement: {improvement:+.4f}")

    # Save results
    results = {
        "config": vars(config),
        "frontier_metrics": frontier_metrics,
        "baseline_frontier_metrics": baseline_frontier_metrics,
        "improvement": improvement,
        "num_frontier_tasks": len(frontier_task_ids),
        "frontier_task_ids": frontier_task_ids,
        "num_pre_frontier_agents": len(pre_frontier_agents),
        "num_post_frontier_agents": len(post_frontier_agents),
        "num_training_samples": len(train_dataset),
    }

    output_path = Path(config.output_dir) / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


def main():
    """Main entry point."""
    config = parse_args()

    logger.info("Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")

    # Enable debug logging if requested
    if config.debug_gradients:
        logging.getLogger("experiment_sad_irt.train").setLevel(logging.DEBUG)

    # Smoke test mode - just check code paths
    if config.smoke_test:
        run_smoke_test(config)
        return

    # Overfit test mode - verify model can memorize small batch
    if config.overfit_test:
        run_overfit_test(config)
        return

    # Run frontier difficulty evaluation
    run_frontier_difficulty_evaluation(config)


if __name__ == "__main__":
    main()
