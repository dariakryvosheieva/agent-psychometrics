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


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Compute gradient norms for each parameter group.

    Returns dict with:
    - total_norm: Overall gradient norm
    - embedding_norm: Gradient norm for theta/beta embeddings
    - encoder_norm: Gradient norm for encoder (LoRA) params
    - head_norm: Gradient norm for psi_head MLP
    """
    norms = {"embedding": 0.0, "encoder": 0.0, "head": 0.0, "other": 0.0}
    counts = {"embedding": 0, "encoder": 0, "head": 0, "other": 0}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.data.norm(2).item() ** 2

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
    ):
        """Initialize trainer.

        Args:
            model: SAD-IRT or StandardIRT model
            train_loader: Training data loader
            eval_loader: Evaluation data loader (optional)
            config: Training configuration
            device: Device to train on
            is_sad_irt: Whether model is SAD-IRT (affects optimizer setup)
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = device
        self.is_sad_irt = is_sad_irt

        # Setup optimizer with different learning rates
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Training state
        self.global_step = 0
        self.best_auc = 0.0

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

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
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

                    # Log gradient norms (before clipping, so compute before step)
                    # Note: gradients are already accumulated at this point
                    grad_norms = compute_gradient_norms(self.model)
                    logger.debug(f"Step {self.global_step} gradients: {grad_norms}")

                    # Log ψ stats if SAD-IRT
                    if self.is_sad_irt and hasattr(self.model, "get_psi_stats"):
                        psi_stats = self.model.get_psi_stats()
                        logger.debug(f"Step {self.global_step} ψ stats: {psi_stats}")

                # Evaluation
                if self.eval_loader is not None and self.global_step % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    logger.info(f"Step {self.global_step}: {metrics}")

                    # Save best model
                    if metrics["auc"] > self.best_auc:
                        self.best_auc = metrics["auc"]
                        self.save_checkpoint("best")

                    self.model.train()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        return {"train_loss": total_loss / num_batches}

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

            # End of epoch evaluation
            if self.eval_loader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1} eval metrics: {eval_metrics}")

                if eval_metrics["auc"] > self.best_auc:
                    self.best_auc = eval_metrics["auc"]
                    self.save_checkpoint("best")

            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        # Final evaluation
        final_metrics = {}
        if self.eval_loader is not None:
            final_metrics = self.evaluate()
            logger.info(f"Final eval metrics: {final_metrics}")

        return final_metrics

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{name}.pt"

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

        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_auc": self.best_auc,
                "config": self.config,
                "epoch": getattr(self, "_current_epoch", 0),
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint for resumption."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Load optimizer and scheduler state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.global_step = checkpoint.get("global_step", 0)
        self.best_auc = checkpoint.get("best_auc", 0.0)
        self._resume_epoch = checkpoint.get("epoch", 0)

        logger.info(f"Resumed from step {self.global_step}, epoch {self._resume_epoch}, best_auc={self.best_auc:.4f}")
