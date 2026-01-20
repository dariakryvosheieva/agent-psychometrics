"""Evaluation metrics for frontier task difficulty prediction.

This module provides functions for:
- Computing Spearman correlation on frontier tasks
- Scale alignment between predicted and oracle difficulties
- ROC-AUC computation using IRT probability model
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def compute_frontier_difficulty_metrics(
    predicted_beta: Dict[str, float],
    oracle_beta: Dict[str, float],
    frontier_task_ids: List[str],
) -> Dict[str, float]:
    """Compute Spearman correlation for frontier task difficulties.

    Args:
        predicted_beta: Dict mapping task_id -> predicted difficulty
        oracle_beta: Dict mapping task_id -> oracle difficulty (from full IRT)
        frontier_task_ids: List of task IDs that are frontier tasks

    Returns:
        Dictionary with Spearman rho and p-value for frontier tasks
    """
    # Get matched pairs
    predicted_values = []
    oracle_values = []

    for task_id in frontier_task_ids:
        if task_id in predicted_beta and task_id in oracle_beta:
            predicted_values.append(predicted_beta[task_id])
            oracle_values.append(oracle_beta[task_id])

    if len(predicted_values) < 3:
        logger.warning(f"Only {len(predicted_values)} frontier tasks with both predictions, skipping correlation")
        return {
            "frontier_spearman_rho": float("nan"),
            "frontier_spearman_p": float("nan"),
            "num_frontier_tasks": len(predicted_values),
        }

    predicted_arr = np.array(predicted_values)
    oracle_arr = np.array(oracle_values)

    spearman_rho, spearman_p = stats.spearmanr(predicted_arr, oracle_arr)
    pearson_r, pearson_p = stats.pearsonr(predicted_arr, oracle_arr)

    logger.info(f"Frontier tasks ({len(predicted_values)}): Spearman ρ = {spearman_rho:.4f} (p={spearman_p:.4f})")

    return {
        "frontier_spearman_rho": float(spearman_rho),
        "frontier_spearman_p": float(spearman_p),
        "frontier_pearson_r": float(pearson_r),
        "frontier_pearson_p": float(pearson_p),
        "num_frontier_tasks": len(predicted_values),
    }


# =============================================================================
# Scale Alignment Functions
# =============================================================================


def analyze_scale_alignment(
    predicted_beta: Dict[str, float],
    oracle_beta: Dict[str, float],
    anchor_task_ids: List[str],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze the relationship between predicted and oracle difficulties.

    Creates diagnostic plots to understand whether a constant shift, affine
    transformation, or more complex function is needed for alignment.

    Args:
        predicted_beta: Dict mapping task_id -> predicted difficulty
        oracle_beta: Dict mapping task_id -> oracle difficulty
        anchor_task_ids: List of task IDs to use as anchors (nontrivial tasks)
        output_path: Optional path to save diagnostic plot

    Returns:
        Dict with analysis results including fit parameters and residual stats
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    # Collect matched pairs
    predicted_vals = []
    oracle_vals = []
    task_ids = []

    for task_id in anchor_task_ids:
        if task_id in predicted_beta and task_id in oracle_beta:
            predicted_vals.append(predicted_beta[task_id])
            oracle_vals.append(oracle_beta[task_id])
            task_ids.append(task_id)

    if len(predicted_vals) < 3:
        logger.warning(f"Only {len(predicted_vals)} anchor tasks, cannot analyze alignment")
        return {"n_anchors": len(predicted_vals), "error": "insufficient data"}

    predicted_arr = np.array(predicted_vals)
    oracle_arr = np.array(oracle_vals)

    # Compute various alignment statistics
    results = {
        "n_anchors": len(predicted_vals),
    }

    # 1. Constant shift analysis
    offsets = oracle_arr - predicted_arr
    constant_offset = float(np.mean(offsets))
    constant_residuals = offsets - constant_offset

    results["constant_shift"] = {
        "offset": constant_offset,
        "residual_mean": float(np.mean(constant_residuals)),
        "residual_std": float(np.std(constant_residuals)),
        "residual_min": float(np.min(constant_residuals)),
        "residual_max": float(np.max(constant_residuals)),
    }

    # 2. Affine transformation analysis (oracle = a * predicted + b)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        predicted_arr, oracle_arr
    )
    affine_predicted = slope * predicted_arr + intercept
    affine_residuals = oracle_arr - affine_predicted

    results["affine_transform"] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "residual_mean": float(np.mean(affine_residuals)),
        "residual_std": float(np.std(affine_residuals)),
    }

    # 3. Correlation metrics
    spearman_r, spearman_p = scipy_stats.spearmanr(predicted_arr, oracle_arr)
    pearson_r, pearson_p = scipy_stats.pearsonr(predicted_arr, oracle_arr)

    results["correlation"] = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
    }

    # Create diagnostic plot
    if output_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Predicted vs Oracle scatter with both fits
        ax1 = axes[0, 0]
        ax1.scatter(predicted_arr, oracle_arr, alpha=0.5, s=20, label="Tasks")

        # Add y=x line
        lims = [
            min(predicted_arr.min(), oracle_arr.min()) - 0.5,
            max(predicted_arr.max(), oracle_arr.max()) + 0.5,
        ]
        ax1.plot(lims, lims, "k--", alpha=0.5, label="y=x")

        # Add constant shift line
        ax1.plot(
            lims,
            [l + constant_offset for l in lims],
            "r-",
            alpha=0.7,
            label=f"Constant: y=x+{constant_offset:.3f}",
        )

        # Add affine fit line
        ax1.plot(
            lims,
            [slope * l + intercept for l in lims],
            "g-",
            alpha=0.7,
            label=f"Affine: y={slope:.3f}x+{intercept:.3f}",
        )

        ax1.set_xlabel("Predicted β")
        ax1.set_ylabel("Oracle β")
        ax1.set_title(f"Scale Alignment ({len(predicted_vals)} anchor tasks)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)

        # Plot 2: Offset distribution (oracle - predicted)
        ax2 = axes[0, 1]
        ax2.hist(offsets, bins=30, edgecolor="black", alpha=0.7)
        ax2.axvline(constant_offset, color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {constant_offset:.3f}")
        ax2.axvline(0, color="black", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Offset (Oracle β - Predicted β)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Offset Distribution (std={np.std(offsets):.3f})")
        ax2.legend()

        # Plot 3: Residuals after constant shift vs predicted
        ax3 = axes[1, 0]
        ax3.scatter(predicted_arr, constant_residuals, alpha=0.5, s=20)
        ax3.axhline(0, color="red", linestyle="--", alpha=0.7)

        # Add trend line to residuals
        res_slope, res_intercept, _, _, _ = scipy_stats.linregress(
            predicted_arr, constant_residuals
        )
        ax3.plot(
            lims,
            [res_slope * l + res_intercept for l in lims],
            "g-",
            alpha=0.7,
            label=f"Trend: slope={res_slope:.3f}",
        )

        ax3.set_xlabel("Predicted β")
        ax3.set_ylabel("Residual after constant shift")
        ax3.set_title("Residuals vs Predicted (constant shift)")
        ax3.legend()

        # Plot 4: Residuals after affine transform vs predicted
        ax4 = axes[1, 1]
        ax4.scatter(predicted_arr, affine_residuals, alpha=0.5, s=20)
        ax4.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax4.set_xlabel("Predicted β")
        ax4.set_ylabel("Residual after affine transform")
        ax4.set_title(f"Residuals vs Predicted (affine, R²={r_value**2:.3f})")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Saved scale alignment diagnostic plot to {output_path}")

    return results


def compute_scale_offset(
    predicted_beta: Dict[str, float],
    oracle_beta: Dict[str, float],
    anchor_task_ids: List[str],
    method: str = "constant",
) -> Dict[str, float]:
    """Compute alignment parameters between predicted and oracle difficulties.

    Args:
        predicted_beta: Dict mapping task_id -> predicted difficulty
        oracle_beta: Dict mapping task_id -> oracle difficulty
        anchor_task_ids: List of task IDs to use as anchors (nontrivial tasks)
        method: Alignment method - "constant" or "affine"

    Returns:
        Dict with alignment parameters:
        - For "constant": {"offset": float}
        - For "affine": {"slope": float, "intercept": float}
    """
    # Collect matched pairs
    predicted_vals = []
    oracle_vals = []

    for task_id in anchor_task_ids:
        if task_id in predicted_beta and task_id in oracle_beta:
            predicted_vals.append(predicted_beta[task_id])
            oracle_vals.append(oracle_beta[task_id])

    if not predicted_vals:
        logger.warning("No anchor tasks found in both predicted and oracle")
        if method == "constant":
            return {"offset": 0.0}
        else:
            return {"slope": 1.0, "intercept": 0.0}

    predicted_arr = np.array(predicted_vals)
    oracle_arr = np.array(oracle_vals)

    if method == "constant":
        offset = float(np.mean(oracle_arr - predicted_arr))
        logger.info(f"Constant offset from {len(predicted_vals)} anchors: {offset:.4f}")
        return {"offset": offset}

    elif method == "affine":
        from scipy import stats as scipy_stats
        slope, intercept, r_value, p_value, _ = scipy_stats.linregress(
            predicted_arr, oracle_arr
        )
        r_squared = r_value ** 2
        logger.info(
            f"Affine transform from {len(predicted_vals)} anchors: "
            f"slope={slope:.4f}, intercept={intercept:.4f}, R²={r_squared:.4f}"
        )
        return {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared)}

    else:
        raise ValueError(f"Unknown alignment method: {method}")


def shift_to_oracle_scale(
    predicted_beta: Dict[str, float],
    alignment_params: Dict[str, float],
) -> Dict[str, float]:
    """Shift predicted difficulties to align with oracle scale.

    Args:
        predicted_beta: Dict mapping task_id -> predicted difficulty
        alignment_params: Dict with alignment parameters:
            - For constant shift: {"offset": float}
            - For affine: {"slope": float, "intercept": float}

    Returns:
        Dict mapping task_id -> shifted difficulty
    """
    if "slope" in alignment_params:
        # Affine transformation: oracle = slope * predicted + intercept
        slope = alignment_params["slope"]
        intercept = alignment_params["intercept"]
        return {
            task_id: slope * beta + intercept
            for task_id, beta in predicted_beta.items()
        }
    else:
        # Constant shift: oracle = predicted + offset
        offset = alignment_params["offset"]
        return {
            task_id: beta + offset
            for task_id, beta in predicted_beta.items()
        }


# =============================================================================
# Frontier AUC Computation
# =============================================================================


def load_responses_dict(responses_path: Path) -> Dict[str, Dict[str, int]]:
    """Load response matrix as nested dict: agent_id -> task_id -> 0|1.

    Args:
        responses_path: Path to JSONL response matrix

    Returns:
        Nested dict of responses
    """
    responses = {}
    with open(responses_path) as f:
        for line in f:
            data = json.loads(line)
            agent_id = data["subject_id"]
            responses[agent_id] = data["responses"]
    return responses


def compute_frontier_auc(
    oracle_abilities: pd.DataFrame,
    shifted_beta: Dict[str, float],
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    eval_agents: List[str],
) -> Dict[str, Any]:
    """Compute ROC-AUC on frontier tasks using oracle abilities and shifted difficulties.

    For each (agent, task) pair:
        - Compute P(success) = sigmoid(theta_oracle - beta_shifted)
        - Compare to actual response

    Args:
        oracle_abilities: DataFrame with 'theta' column, indexed by agent_id
        shifted_beta: Dict of difficulties aligned to oracle scale
        responses: Response matrix as nested dict (agent_id -> task_id -> 0|1)
        frontier_task_ids: Tasks to evaluate on
        eval_agents: Agents to evaluate on (typically post-frontier only)

    Returns:
        Dict with 'auc', 'n_pairs', 'n_positive', 'n_negative'
    """
    y_true = []
    y_scores = []

    for agent_id in eval_agents:
        if agent_id not in oracle_abilities.index:
            continue
        if agent_id not in responses:
            continue

        theta = oracle_abilities.loc[agent_id, "theta"]
        agent_responses = responses[agent_id]

        for task_id in frontier_task_ids:
            if task_id not in shifted_beta:
                continue
            if task_id not in agent_responses:
                continue

            beta = shifted_beta[task_id]
            prob = float(sigmoid(theta - beta))
            response = agent_responses[task_id]

            y_scores.append(prob)
            y_true.append(response)

    n_pairs = len(y_true)
    n_positive = sum(y_true)
    n_negative = n_pairs - n_positive

    if n_pairs == 0:
        logger.warning("No (agent, task) pairs found for AUC computation")
        return {
            "auc": None,
            "n_pairs": 0,
            "n_positive": 0,
            "n_negative": 0,
        }

    if n_positive == 0 or n_negative == 0:
        logger.warning(f"All responses are the same class (pos={n_positive}, neg={n_negative})")
        return {
            "auc": None,
            "n_pairs": n_pairs,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    auc = roc_auc_score(y_true, y_scores)

    logger.info(
        f"Frontier AUC: {auc:.4f} (n_pairs={n_pairs}, pos={n_positive}, neg={n_negative})"
    )

    return {
        "auc": float(auc),
        "n_pairs": n_pairs,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
