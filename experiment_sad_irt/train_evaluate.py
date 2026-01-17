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
from .dataset import TrajectoryIRTDataset, create_train_test_split
from .evaluate import compute_metrics, log_parameter_stats
from .model import SADIRT, StandardIRT
from .train import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> SADIRTConfig:
    """Parse command line arguments into config."""
    parser = argparse.ArgumentParser(description="SAD-IRT Training and Evaluation")

    # Mode
    parser.add_argument("--mode", type=str, default="full_auc", choices=["full_auc", "calibration"])

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora_r", type=int, default=16)

    # Data
    parser.add_argument("--response_matrix_path", type=str, default="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl")
    parser.add_argument("--trajectory_dir", type=str, default="trajectory_data/unified_trajs")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--hard_threshold", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate_encoder", type=float, default=1e-4)
    parser.add_argument("--learning_rate_embeddings", type=float, default=1e-3)

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
        mode=args.mode,
        model_name=args.model_name,
        lora_r=args.lora_r,
        response_matrix_path=args.response_matrix_path,
        trajectory_dir=args.trajectory_dir,
        max_length=args.max_length,
        test_fraction=args.test_fraction,
        hard_threshold=args.hard_threshold,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate_encoder=args.learning_rate_encoder,
        learning_rate_embeddings=args.learning_rate_embeddings,
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


def run_full_auc_evaluation(config: SADIRTConfig):
    """Run Part 2: Full AUC evaluation comparing SAD-IRT to baseline IRT."""
    logger.info("=" * 60)
    logger.info("Part 2: Full AUC Evaluation")
    logger.info("=" * 60)

    # Set seed
    set_seed(config.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create full dataset (to get all agents/tasks)
    logger.info("Loading dataset...")
    full_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        swebench_dataset=config.swebench_dataset,
    )

    logger.info(f"Full dataset: {len(full_dataset)} samples")
    logger.info(f"Agents: {full_dataset.num_agents}, Tasks: {full_dataset.num_tasks}")

    # Create train/test split by (agent, task) pairs
    train_pairs, test_pairs = create_train_test_split(
        full_dataset, test_fraction=config.test_fraction, seed=config.seed
    )

    # Limit samples if dry run
    if config.dry_run or config.max_samples:
        max_samples = config.max_samples or 100
        train_pairs = train_pairs[:max_samples]
        test_pairs = test_pairs[:min(max_samples // 4, len(test_pairs))]
        logger.info(f"DRY RUN: Limited to {len(train_pairs)} train, {len(test_pairs)} test pairs")

    # Create train/test datasets with the specific pairs
    train_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=train_pairs,
        swebench_dataset=config.swebench_dataset,
    )

    test_dataset = TrajectoryIRTDataset(
        response_matrix_path=config.response_matrix_path,
        trajectory_dir=config.trajectory_dir,
        tokenizer=tokenizer,
        max_length=config.max_length,
        agent_ids=full_dataset.agent_ids,
        task_ids=full_dataset.task_ids,
        pairs=test_pairs,
        swebench_dataset=config.swebench_dataset,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ===== Load Pre-trained Baseline IRT =====
    # We use pre-computed IRT values instead of training from scratch
    # This saves significant time since baseline IRT is deterministic given the data
    logger.info("\n" + "=" * 40)
    logger.info("Loading Pre-trained Baseline IRT")
    logger.info("=" * 40)

    baseline_irt_dir = Path("clean_data/swebench_verified_20251120_full/1d")
    baseline_metrics = {}

    if baseline_irt_dir.exists():
        import pandas as pd
        abilities_df = pd.read_csv(baseline_irt_dir / "abilities.csv", index_col=0)
        items_df = pd.read_csv(baseline_irt_dir / "items.csv", index_col=0)
        logger.info(f"Loaded pre-trained IRT: {len(abilities_df)} agents, {len(items_df)} tasks")

        # Create baseline model and initialize with pre-trained values
        baseline_model = StandardIRT(
            num_agents=full_dataset.num_agents,
            num_tasks=full_dataset.num_tasks,
        ).to(device)

        # Initialize with pre-trained values
        with torch.no_grad():
            for i, agent_id in enumerate(full_dataset.agent_ids):
                if agent_id in abilities_df.index:
                    baseline_model.theta.weight[i] = abilities_df.loc[agent_id, "theta"]
            for i, task_id in enumerate(full_dataset.task_ids):
                if task_id in items_df.index:
                    baseline_model.beta.weight[i] = items_df.loc[task_id, "b"]

        # Evaluate on test set
        baseline_model.eval()
        all_logits = []
        all_responses = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = baseline_model(
                    agent_idx=batch["agent_idx"],
                    task_idx=batch["task_idx"],
                )
                all_logits.append(logits.cpu())
                all_responses.append(batch["response"].cpu())

        all_logits = torch.cat(all_logits)
        all_responses = torch.cat(all_responses)
        baseline_metrics = compute_metrics(all_logits, all_responses)
        logger.info(f"Baseline IRT metrics (pre-trained): {baseline_metrics}")
    else:
        logger.warning(f"Pre-trained IRT not found at {baseline_irt_dir}, training from scratch")
        baseline_model = StandardIRT(
            num_agents=full_dataset.num_agents,
            num_tasks=full_dataset.num_tasks,
        ).to(device)

        baseline_trainer = Trainer(
            model=baseline_model,
            train_loader=train_loader,
            eval_loader=test_loader,
            config=config,
            device=device,
            is_sad_irt=False,
        )

        baseline_metrics = baseline_trainer.train()
        logger.info(f"Baseline IRT final metrics: {baseline_metrics}")

    log_parameter_stats(baseline_model, prefix="Baseline ")

    # ===== Train SAD-IRT =====
    logger.info("\n" + "=" * 40)
    logger.info("Training SAD-IRT (with trajectory encoder)")
    logger.info("=" * 40)

    # Determine ψ normalization strategy
    # - If explicitly specified via CLI, use that
    # - Otherwise: use centering for frozen IRT (to avoid BatchNorm instability),
    #   and batchnorm for trainable IRT (following SAD-IRT paper)
    if hasattr(config, 'psi_normalization') and config.psi_normalization is not None:
        psi_normalization = config.psi_normalization
    else:
        psi_normalization = "center" if config.freeze_irt else "batchnorm"
    logger.info(f"Using psi_normalization={psi_normalization}")

    sad_irt_model = SADIRT(
        num_agents=full_dataset.num_agents,
        num_tasks=full_dataset.num_tasks,
        model_name=config.model_name,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        psi_normalization=psi_normalization,
    ).to(device)

    # Initialize θ/β from pre-trained IRT (much better than random init)
    if baseline_irt_dir.exists():
        sad_irt_model.initialize_from_pretrained_irt(
            agent_ids=full_dataset.agent_ids,
            task_ids=full_dataset.task_ids,
            abilities_df=abilities_df,
            items_df=items_df,
        )

    # Optionally freeze θ/β (ablation: only train ψ predictor)
    if config.freeze_irt:
        logger.info("Freezing θ and β (only training ψ predictor)")
        sad_irt_model.theta.weight.requires_grad = False
        sad_irt_model.beta.weight.requires_grad = False

    sad_irt_trainer = Trainer(
        model=sad_irt_model,
        train_loader=train_loader,
        eval_loader=test_loader,
        config=config,
        device=device,
        is_sad_irt=True,
    )

    # Resume from checkpoint if specified (overrides initialization)
    if config.resume_from:
        sad_irt_trainer.load_checkpoint(config.resume_from)

    sad_irt_metrics = sad_irt_trainer.train()
    logger.info(f"SAD-IRT final metrics: {sad_irt_metrics}")
    log_parameter_stats(sad_irt_model, prefix="SAD-IRT ")

    # ===== Compare Results =====
    logger.info("\n" + "=" * 40)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 40)

    baseline_auc = baseline_metrics.get("auc", 0)
    sad_irt_auc = sad_irt_metrics.get("auc", 0)
    improvement = sad_irt_auc - baseline_auc

    logger.info(f"Baseline IRT AUC: {baseline_auc:.4f}")
    logger.info(f"SAD-IRT AUC:      {sad_irt_auc:.4f}")
    logger.info(f"Improvement:      {improvement:+.4f} ({improvement / baseline_auc * 100:+.2f}%)")

    # Save results
    results = {
        "config": vars(config),
        "baseline_metrics": baseline_metrics,
        "sad_irt_metrics": sad_irt_metrics,
        "improvement": improvement,
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "num_agents": full_dataset.num_agents,
        "num_tasks": full_dataset.num_tasks,
    }

    output_path = Path(config.output_dir) / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    return results


def run_calibration_evaluation(config: SADIRTConfig):
    """Run Part 1: Calibration evaluation on hard tasks."""
    logger.info("=" * 60)
    logger.info("Part 1: Calibration Evaluation (Hard Tasks)")
    logger.info("=" * 60)

    # TODO: Implement calibration evaluation
    # This requires:
    # 1. Train SAD-IRT on M1+M2 agents only
    # 2. Train oracle IRT on M1+M2+M3 agents
    # 3. Compare β estimates on hard tasks

    raise NotImplementedError(
        "Calibration evaluation not yet implemented. "
        "Use --mode full_auc for Part 2 evaluation first."
    )


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

    if config.mode == "full_auc":
        run_full_auc_evaluation(config)
    elif config.mode == "calibration":
        run_calibration_evaluation(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
