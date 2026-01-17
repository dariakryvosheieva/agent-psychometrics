"""Training loop for SAD-IRT."""

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import SADIRTConfig
from .evaluate import compute_metrics

logger = logging.getLogger(__name__)


def compute_gradient_norms(model: nn.Module, detailed: bool = False) -> Dict[str, float]:
    """Compute gradient norms for each parameter group.

    Returns dict with:
    - total_norm: Overall gradient norm
    - embedding_norm: Gradient norm for theta/beta embeddings
    - encoder_norm: Gradient norm for encoder (LoRA) params
    - head_norm: Gradient norm for psi_head MLP

    If detailed=True, also includes per-layer norms for debugging.
    """
    norms = {"embedding": 0.0, "encoder": 0.0, "head": 0.0, "other": 0.0}
    counts = {"embedding": 0, "encoder": 0, "head": 0, "other": 0}

    # For detailed logging
    detailed_norms = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.data.norm(2).item() ** 2

        if detailed:
            detailed_norms[name] = param.grad.data.norm(2).item()

        if "theta" in name or "beta" in name:
            norms["embedding"] += grad_norm
            counts["embedding"] += 1
        elif "lora" in name:
            norms["encoder"] += grad_norm
            counts["encoder"] += 1
        elif "psi_head" in name:
            norms["head"] += grad_norm
            counts["head"] += 1
        else:
            norms["other"] += grad_norm
            counts["other"] += 1

    # Convert to actual norms
    result = {}
    total = 0.0
    for key in norms:
        if counts[key] > 0:
            result[f"{key}_norm"] = norms[key] ** 0.5
            total += norms[key]
    result["total_norm"] = total ** 0.5

    if detailed:
        result["detailed"] = detailed_norms

    return result


class Trainer:
    """Trainer for SAD-IRT model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: SADIRTConfig,
        device: torch.device,
        is_sad_irt: bool = True,
        # Frontier evaluation data (for checkpoint selection based on Spearman ρ)
        task_ids: Optional[list] = None,
        frontier_task_ids: Optional[list] = None,
        oracle_beta: Optional[Dict[str, float]] = None,
    ):
        """Initialize trainer.

        Args:
            model: SAD-IRT or StandardIRT model
            train_loader: Training data loader
            eval_loader: Evaluation data loader (optional)
            config: Training configuration
            device: Device to train on
            is_sad_irt: Whether model is SAD-IRT (affects optimizer setup)
            task_ids: List of task IDs in order matching model indices (for frontier eval)
            frontier_task_ids: List of frontier task IDs (for checkpoint selection)
            oracle_beta: Dict mapping task_id -> oracle difficulty (for checkpoint selection)
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = device
        self.is_sad_irt = is_sad_irt

        # Frontier evaluation data
        self.task_ids = task_ids
        self.frontier_task_ids = frontier_task_ids
        self.oracle_beta = oracle_beta

        # Setup optimizer with different learning rates
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Training state
        self.global_step = 0
        self.best_auc = 0.0
        self.best_spearman = -1.0  # For frontier evaluation

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer with different learning rates for different parameter groups."""
        if self.is_sad_irt:
            # Different learning rates for encoder vs embeddings
            encoder_params = []
            embedding_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "theta" in name or "beta" in name:
                    embedding_params.append(param)
                else:
                    encoder_params.append(param)

            param_groups = [
                {"params": encoder_params, "lr": self.config.learning_rate_encoder},
                {"params": embedding_params, "lr": self.config.learning_rate_embeddings},
            ]
        else:
            # Standard IRT - all params same learning rate
            param_groups = [
                {"params": self.model.parameters(), "lr": self.config.learning_rate_embeddings}
            ]

        return AdamW(param_groups, weight_decay=self.config.weight_decay)

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup.

        Note: Counts are in optimizer steps (after gradient accumulation),
        not raw batches.
        """
        # Calculate optimizer steps, not raw batches
        batches_per_epoch = len(self.train_loader)
        optimizer_steps_per_epoch = batches_per_epoch // self.config.gradient_accumulation_steps
        num_optimizer_steps = optimizer_steps_per_epoch * self.config.epochs
        num_warmup_steps = int(num_optimizer_steps * self.config.warmup_ratio)

        logger.info(f"Scheduler setup: {num_optimizer_steps} optimizer steps, {num_warmup_steps} warmup steps")

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_optimizer_steps - num_warmup_steps,
            eta_min=1e-6,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # For tracking ψ statistics across epoch
        psi_values_epoch = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with ψ tracking if debug mode
            if self.config.debug_gradients and self.is_sad_irt:
                logits, psi_raw = self._forward_with_psi_tracking(batch)
                psi_values_epoch.append(psi_raw.detach())
            else:
                logits = self.model(
                    agent_idx=batch["agent_idx"],
                    task_idx=batch["task_idx"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            # Compute loss
            loss = self.criterion(logits, batch["response"])
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Log gradients BEFORE clipping for debugging
                if self.config.debug_gradients and self.global_step % self.config.logging_steps == 0:
                    self._log_detailed_gradients()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    current_loss = loss.item() * self.config.gradient_accumulation_steps
                    pbar.set_postfix({"loss": current_loss, "lr": f"{lr:.2e}"})

                    # Log gradient norms (after clipping now, for comparison)
                    grad_norms = compute_gradient_norms(self.model)
                    logger.debug(f"Step {self.global_step} gradients (post-clip): {grad_norms}")

                    # Log ψ stats if SAD-IRT
                    if self.is_sad_irt and hasattr(self.model, "get_psi_stats"):
                        psi_stats = self.model.get_psi_stats()
                        logger.debug(f"Step {self.global_step} ψ stats: {psi_stats}")

                # Evaluation
                if self.eval_loader is not None and self.global_step % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    logger.info(f"Step {self.global_step}: {metrics}")

                    # Save best model (with metrics for easy identification)
                    if metrics["auc"] > self.best_auc:
                        self.best_auc = metrics["auc"]
                        self.save_checkpoint("best", metrics=metrics)

                    self.model.train()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        # End of epoch: log ψ statistics summary
        if self.config.debug_gradients and self.is_sad_irt and psi_values_epoch:
            all_psi = torch.cat(psi_values_epoch)
            logger.info(
                f"Epoch {epoch + 1} ψ summary: "
                f"mean={all_psi.mean().item():.6f}, "
                f"std={all_psi.std().item():.6f}, "
                f"min={all_psi.min().item():.6f}, "
                f"max={all_psi.max().item():.6f}"
            )

        return {"train_loss": total_loss / num_batches}

    def _forward_with_psi_tracking(self, batch: Dict[str, torch.Tensor]):
        """Forward pass that also returns raw ψ values for debugging."""
        # We need to manually extract ψ values during forward
        # This duplicates some model code but allows tracking
        model = self.model

        # Get IRT parameters
        theta = model.theta(batch["agent_idx"])
        beta = model.beta(batch["task_idx"])

        # Encode trajectory
        outputs = model.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.last_hidden_state
        seq_lengths = batch["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]

        # Predict ψ (raw, before normalization)
        psi_raw = model.psi_head(last_token_hidden.float())

        # Apply normalization
        if model.psi_normalization == "batchnorm" and model.psi_bn is not None:
            if model.training and psi_raw.size(0) > 1:
                psi = model.psi_bn(psi_raw)
            else:
                model.psi_bn.eval()
                psi = model.psi_bn(psi_raw)
                if model.training:
                    model.psi_bn.train()
        elif model.psi_normalization == "center":
            psi = psi_raw - psi_raw.mean()
        else:
            psi = psi_raw

        # IRT formula
        logits = theta - (beta + psi)

        return logits.squeeze(-1), psi_raw.squeeze(-1)

    def _log_detailed_gradients(self):
        """Log detailed gradient information for debugging."""
        grad_norms = compute_gradient_norms(self.model, detailed=True)

        # Summary
        logger.info(
            f"Step {self.global_step} gradients (pre-clip): "
            f"total={grad_norms['total_norm']:.6f}, "
            f"embedding={grad_norms.get('embedding_norm', 0):.6f}, "
            f"encoder={grad_norms.get('encoder_norm', 0):.6f}, "
            f"head={grad_norms.get('head_norm', 0):.6f}"
        )

        # Key parameter gradients
        detailed = grad_norms.get("detailed", {})
        key_params = ["psi_head.weight", "psi_head.bias", "theta.weight", "beta.weight"]
        for param_name in key_params:
            for full_name, norm in detailed.items():
                if param_name in full_name:
                    logger.info(f"  {param_name}: grad_norm={norm:.8f}")
                    break

        # Sample of LoRA gradients (just first few)
        lora_grads = [(k, v) for k, v in detailed.items() if "lora" in k]
        if lora_grads:
            lora_grads.sort(key=lambda x: -x[1])  # Sort by magnitude
            logger.info(f"  Top 3 LoRA grads: {lora_grads[:3]}")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval set."""
        if self.eval_loader is None:
            return {}

        self.model.eval()

        all_logits = []
        all_responses = []

        for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            logits = self.model(
                agent_idx=batch["agent_idx"],
                task_idx=batch["task_idx"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            all_logits.append(logits.cpu())
            all_responses.append(batch["response"].cpu())

        all_logits = torch.cat(all_logits)
        all_responses = torch.cat(all_responses)

        return compute_metrics(all_logits, all_responses)

    @torch.no_grad()
    def evaluate_frontier(self) -> Dict[str, float]:
        """Evaluate Spearman ρ on frontier tasks against oracle β.

        This is the primary metric for checkpoint selection in frontier difficulty mode.
        """
        if self.frontier_task_ids is None or self.oracle_beta is None or self.task_ids is None:
            return {}

        from scipy import stats

        self.model.eval()

        # Get current learned β values
        learned_beta_tensor = self.model.get_difficulties()
        learned_beta = {
            task_id: float(learned_beta_tensor[i])
            for i, task_id in enumerate(self.task_ids)
        }

        # Compute Spearman ρ on frontier tasks
        predicted_values = []
        oracle_values = []

        for task_id in self.frontier_task_ids:
            if task_id in learned_beta and task_id in self.oracle_beta:
                predicted_values.append(learned_beta[task_id])
                oracle_values.append(self.oracle_beta[task_id])

        if len(predicted_values) < 3:
            return {"frontier_spearman_rho": float("nan"), "num_frontier_tasks": len(predicted_values)}

        spearman_rho, spearman_p = stats.spearmanr(predicted_values, oracle_values)

        return {
            "frontier_spearman_rho": float(spearman_rho),
            "frontier_spearman_p": float(spearman_p),
            "num_frontier_tasks": len(predicted_values),
        }

    def train(self) -> Dict[str, float]:
        """Full training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Effective batch size: {self.config.effective_batch_size}")

        # Support resumption from checkpoint
        start_epoch = getattr(self, "_resume_epoch", 0)
        if start_epoch > 0:
            logger.info(f"Resuming from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, self.config.epochs):
            self._current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} train metrics: {train_metrics}")

            # Frontier evaluation (Spearman ρ-based) - primary metric for SAD-IRT
            frontier_metrics = self.evaluate_frontier()
            if frontier_metrics:
                logger.info(f"Epoch {epoch + 1} frontier metrics: {frontier_metrics}")

                spearman = frontier_metrics.get("frontier_spearman_rho", -1.0)
                if not math.isnan(spearman) and spearman > self.best_spearman:
                    self.best_spearman = spearman
                    self.save_checkpoint("best", metrics=frontier_metrics)
                    logger.info(f"New best Spearman ρ: {spearman:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}", metrics=frontier_metrics if frontier_metrics else None)

        # Final evaluation
        final_metrics = self.evaluate_frontier()
        if final_metrics:
            logger.info(f"Final frontier metrics: {final_metrics}")

        return final_metrics

    def save_checkpoint(self, name: str, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint with versioned filename.

        Checkpoints are saved with timestamp to avoid overwriting.
        Format: checkpoint_{name}_step{step}_{timestamp}.pt

        Args:
            name: Checkpoint type (e.g., "best", "epoch_1")
            metrics: Optional metrics dict to include (e.g., {"auc": 0.85})
        """
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.output_dir / f"checkpoint_{name}_step{self.global_step}_{timestamp}.pt"

        # For SAD-IRT, only save the trainable parts
        if self.is_sad_irt:
            # Save LoRA weights, embeddings, and MLP head
            state_dict = {}
            for key, value in self.model.state_dict().items():
                # Save embeddings, MLP head, BatchNorm, and LoRA weights
                if any(x in key for x in ["theta", "beta", "psi_head", "psi_bn", "lora"]):
                    state_dict[key] = value
        else:
            state_dict = self.model.state_dict()

        # Convert config to dict for serialization (avoids PyTorch 2.6 weights_only issues)
        from dataclasses import asdict
        config_dict = asdict(self.config)

        checkpoint_data = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_auc": self.best_auc,
            "config": config_dict,  # Save as dict, not object
            "epoch": getattr(self, "_current_epoch", 0),
            "timestamp": timestamp,
        }
        if metrics:
            checkpoint_data["metrics"] = metrics

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint for resumption.

        Args:
            checkpoint_path: Path to checkpoint file. The checkpoint contains
                model weights, optimizer state, scheduler state, and training
                progress (global_step, epoch, best_auc).
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # Use weights_only=False since we save config as dict (safe for our own checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_auc = checkpoint.get("best_auc", 0.0)
        self._resume_epoch = checkpoint.get("epoch", 0)

        # Log checkpoint info
        timestamp = checkpoint.get("timestamp", "unknown")
        metrics = checkpoint.get("metrics", {})
        logger.info(f"Resumed from step {self.global_step}, epoch {self._resume_epoch}, best_auc={self.best_auc:.4f}")
        logger.info(f"Checkpoint timestamp: {timestamp}")
        if metrics:
            logger.info(f"Checkpoint metrics: {metrics}")
