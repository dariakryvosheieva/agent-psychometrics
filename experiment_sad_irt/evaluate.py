"""Evaluation metrics for SAD-IRT.

This module contains SAD-IRT specific evaluation functions (torch-based).
For general frontier evaluation utilities, see experiment_b.shared.evaluation.

Re-exports from experiment_b.shared.evaluation for backwards compatibility:
- analyze_scale_alignment
- compute_scale_offset
- shift_to_oracle_scale
- load_responses_dict
- compute_frontier_auc
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score, brier_score_loss

# Re-export general utilities from experiment_b for backwards compatibility
from experiment_b.shared.evaluation import (
    analyze_scale_alignment,
    compute_scale_offset,
    shift_to_oracle_scale,
    load_responses_dict,
    compute_frontier_auc,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    logits: torch.Tensor,
    responses: torch.Tensor,
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        logits: Model predictions (unnormalized log-odds)
        responses: Ground truth binary responses

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    logits_np = logits.numpy()
    responses_np = responses.numpy()

    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits_np))

    # AUC-ROC
    try:
        auc = roc_auc_score(responses_np, probs)
    except ValueError:
        # All responses are the same class
        auc = 0.5

    # Brier score (lower is better)
    brier = brier_score_loss(responses_np, probs)

    # Accuracy at threshold 0.5
    preds = (probs >= 0.5).astype(int)
    accuracy = (preds == responses_np).mean()

    # Log loss (cross-entropy)
    eps = 1e-7
    probs_clipped = np.clip(probs, eps, 1 - eps)
    log_loss = -np.mean(
        responses_np * np.log(probs_clipped) + (1 - responses_np) * np.log(1 - probs_clipped)
    )

    return {
        "auc": float(auc),
        "brier": float(brier),
        "accuracy": float(accuracy),
        "log_loss": float(log_loss),
    }


def compute_difficulty_correlation(
    estimated_beta: torch.Tensor,
    oracle_beta: torch.Tensor,
    task_pass_rates: Optional[torch.Tensor] = None,
    hard_threshold: float = 0.2,
) -> Dict[str, float]:
    """Compute correlation between estimated and oracle difficulties.

    Args:
        estimated_beta: Estimated task difficulties from SAD-IRT
        oracle_beta: Oracle task difficulties (from full IRT)
        task_pass_rates: Pass rates per task (to identify hard tasks)
        hard_threshold: Threshold for "hard" tasks (pass rate <=)

    Returns:
        Dictionary of correlation metrics
    """
    estimated = estimated_beta.numpy()
    oracle = oracle_beta.numpy()

    # Overall correlations
    pearson_r, pearson_p = stats.pearsonr(estimated, oracle)
    spearman_r, spearman_p = stats.spearmanr(estimated, oracle)

    results = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

    # Hard task correlations (if pass rates provided)
    if task_pass_rates is not None:
        pass_rates = task_pass_rates.numpy()
        hard_mask = pass_rates <= hard_threshold

        if hard_mask.sum() > 2:  # Need at least 3 points for correlation
            hard_estimated = estimated[hard_mask]
            hard_oracle = oracle[hard_mask]

            hard_pearson_r, hard_pearson_p = stats.pearsonr(hard_estimated, hard_oracle)
            hard_spearman_r, hard_spearman_p = stats.spearmanr(hard_estimated, hard_oracle)

            results.update({
                "hard_pearson_r": float(hard_pearson_r),
                "hard_pearson_p": float(hard_pearson_p),
                "hard_spearman_r": float(hard_spearman_r),
                "hard_spearman_p": float(hard_spearman_p),
                "num_hard_tasks": int(hard_mask.sum()),
            })
        else:
            logger.warning(f"Only {hard_mask.sum()} hard tasks, skipping hard task correlations")

    return results


def compute_calibration(
    probs: np.ndarray,
    responses: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute calibration metrics.

    Args:
        probs: Predicted probabilities
        responses: Ground truth binary responses
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with ECE (Expected Calibration Error) and MCE (Max Calibration Error)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue

        bin_probs = probs[mask]
        bin_responses = responses[mask]

        avg_confidence = bin_probs.mean()
        avg_accuracy = bin_responses.mean()

        gap = abs(avg_confidence - avg_accuracy)
        ece += mask.sum() / len(probs) * gap
        mce = max(mce, gap)

    return {
        "ece": float(ece),
        "mce": float(mce),
    }


def log_parameter_stats(model, prefix: str = ""):
    """Log statistics about model parameters."""
    if hasattr(model, "get_abilities"):
        abilities = model.get_abilities()
        logger.info(
            f"{prefix}θ (abilities): mean={abilities.mean():.4f}, "
            f"std={abilities.std():.4f}, min={abilities.min():.4f}, max={abilities.max():.4f}"
        )

    if hasattr(model, "get_difficulties"):
        difficulties = model.get_difficulties()
        logger.info(
            f"{prefix}β (difficulties): mean={difficulties.mean():.4f}, "
            f"std={difficulties.std():.4f}, min={difficulties.min():.4f}, max={difficulties.max():.4f}"
        )

    if hasattr(model, "get_psi_stats"):
        psi_stats = model.get_psi_stats()
        logger.info(f"{prefix}ψ BatchNorm stats: {psi_stats}")
