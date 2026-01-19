"""Unified evaluation pipeline for Experiment A.

This module provides:
- compute_auc: Generic AUC computation using IRT formula
- PredictorConfig: Configuration for a single predictor
- run_evaluation_pipeline: Run all predictors and compute AUC
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

from experiment_a_common.dataset import ExperimentData


def compute_irt_probability(theta: float, beta: float) -> float:
    """Compute IRT probability using 1PL formula.

    P(success) = sigmoid(theta - beta)

    Args:
        theta: Agent ability
        beta: Task difficulty

    Returns:
        Probability of success
    """
    return float(sigmoid(theta - beta))


def compute_auc(
    data: ExperimentData,
    predicted_difficulties: Dict[str, float],
    use_full_abilities: bool = False,
) -> Dict[str, Any]:
    """Compute AUC for predicted difficulties using IRT formula.

    This is the unified AUC computation that works for both binary and
    binomial data by delegating the expansion to the dataset.

    Args:
        data: The experiment dataset (handles binary/binomial expansion)
        predicted_difficulties: Mapping of task_id -> predicted difficulty
        use_full_abilities: If True, use full IRT abilities (oracle only).
                           If False, use train-only abilities (all methods).

    Returns:
        Dict with 'auc', 'n_predictions', and 'n_observations'
    """
    abilities = data.full_abilities if use_full_abilities else data.train_abilities

    y_true: List[int] = []
    y_scores: List[float] = []
    n_pairs = 0

    for task_id in data.test_tasks:
        beta_pred = predicted_difficulties.get(task_id)
        if beta_pred is None:
            continue

        for agent_id in abilities.index:
            if task_id not in data.responses.get(agent_id, {}):
                continue

            theta = abilities.loc[agent_id, "ability"]
            prob = compute_irt_probability(theta, beta_pred)

            # Delegate expansion to dataset (handles binary vs binomial)
            yt, ys = data.expand_for_auc(agent_id, task_id, prob)
            y_true.extend(yt)
            y_scores.extend(ys)
            n_pairs += 1

    if len(y_true) == 0 or len(set(y_true)) < 2:
        return {"auc": None, "n_predictions": n_pairs, "n_observations": len(y_true)}

    auc = roc_auc_score(y_true, y_scores)
    return {
        "auc": float(auc),
        "n_predictions": n_pairs,
        "n_observations": len(y_true),
    }


@dataclass
class PredictorConfig:
    """Configuration for a single difficulty predictor."""

    # Predictor class or factory function
    predictor_class: Type
    # Name for results dict
    name: str
    # Display name for printing
    display_name: str
    # Whether this predictor is enabled
    enabled: bool = True
    # Keyword arguments to pass to predictor constructor
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Whether to use full IRT abilities (only for oracle)
    use_full_abilities: bool = False
    # Whether to store predictions in results
    store_predictions: bool = True


@dataclass
class PredictorResult:
    """Result from running a single predictor."""

    name: str
    auc: Optional[float]
    auc_result: Dict[str, Any]
    predictions: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def evaluate_single_predictor(
    data: ExperimentData,
    config: PredictorConfig,
    train_b: np.ndarray,
    verbose: bool = True,
) -> PredictorResult:
    """Evaluate a single predictor.

    Args:
        data: The experiment dataset
        config: Predictor configuration
        train_b: Ground truth difficulties for training tasks
        verbose: Whether to print progress

    Returns:
        PredictorResult with AUC and metadata
    """
    try:
        # Instantiate predictor
        predictor = config.predictor_class(**config.kwargs)

        if verbose:
            # Print predictor-specific info
            if hasattr(predictor, "n_embeddings"):
                print(f"   Embeddings loaded: {predictor.n_embeddings} tasks")
            if hasattr(predictor, "embedding_dim"):
                print(f"   Embedding dim: {predictor.embedding_dim}")
            if hasattr(predictor, "n_tasks"):
                print(f"   Features loaded: {predictor.n_tasks} tasks")
            if hasattr(predictor, "n_features"):
                print(f"   Feature count: {predictor.n_features}")

        # Fit predictor
        predictor.fit(data.train_tasks, train_b)

        # Print selected features if applicable
        if verbose and hasattr(predictor, "print_selected_features"):
            predictor.print_selected_features()
        elif verbose and hasattr(predictor, "print_feature_coefficients"):
            predictor.print_feature_coefficients()

        # Predict
        predictions = predictor.predict(data.test_tasks)

        # Compute AUC
        auc_result = compute_auc(
            data, predictions, use_full_abilities=config.use_full_abilities
        )

        if verbose:
            auc_val = auc_result.get("auc")
            if auc_val is not None:
                print(f"   {config.display_name} AUC: {auc_val:.4f}")
            else:
                print(f"   {config.display_name} AUC: N/A")

        # Collect metadata from predictor
        metadata: Dict[str, Any] = {}
        for attr in [
            "n_embeddings", "embedding_dim", "n_tasks", "n_features",
            "selected_features", "feature_coefficients", "feature_names",
        ]:
            if hasattr(predictor, attr):
                val = getattr(predictor, attr)
                if val is not None:
                    metadata[attr] = val

        # Get additional metadata from get_metadata() if available
        if hasattr(predictor, "get_metadata"):
            metadata.update(predictor.get_metadata())

        return PredictorResult(
            name=config.name,
            auc=auc_result.get("auc"),
            auc_result=auc_result,
            predictions=predictions if config.store_predictions else None,
            metadata=metadata,
        )

    except Exception as e:
        import traceback
        if verbose:
            print(f"   Error with {config.display_name}: {e}")
            traceback.print_exc()
        return PredictorResult(
            name=config.name,
            auc=None,
            auc_result={},
            error=str(e),
        )


def run_evaluation_pipeline(
    data: ExperimentData,
    predictor_configs: List[PredictorConfig],
    verbose: bool = True,
    compute_binomial: bool = False,
) -> Dict[str, Any]:
    """Run all configured predictors and compute AUC for each.

    Args:
        data: The experiment dataset
        predictor_configs: List of predictor configurations
        verbose: Whether to print progress
        compute_binomial: If True and data is binomial, compute MAE/accuracy metrics

    Returns:
        Dict with results for each predictor
    """
    results: Dict[str, Any] = {}
    predictions: Dict[str, Dict[str, float]] = {}

    # Get training difficulties
    train_b = data.get_train_difficulties()

    for i, config in enumerate(predictor_configs, 1):
        if not config.enabled:
            results[config.name] = {"skipped": True}
            continue

        if verbose:
            print(f"\n{i}. Evaluating {config.display_name}...")

        result = evaluate_single_predictor(data, config, train_b, verbose)

        # Format result for storage
        if result.error:
            results[config.name] = {"error": result.error}
        else:
            result_dict: Dict[str, Any] = {
                "auc_result": result.auc_result,
            }
            result_dict.update(result.metadata)

            # Add kwargs for reproducibility
            for key, val in config.kwargs.items():
                if key not in result_dict:
                    result_dict[key] = str(val) if hasattr(val, "__fspath__") else val

            # Compute binomial metrics if requested
            if compute_binomial and result.predictions:
                from experiment_a_common.binomial_metrics import compute_binomial_metrics
                from experiment_a_common.dataset import BinomialExperimentData
                if isinstance(data, BinomialExperimentData):
                    binom_result = compute_binomial_metrics(
                        data, result.predictions, use_full_abilities=config.use_full_abilities
                    )
                    result_dict["binomial_metrics"] = binom_result.to_dict()
                    if verbose:
                        print(f"   {config.display_name} MAE: {binom_result.mae:.4f}, "
                              f"Accuracy: {binom_result.pass5_accuracy:.4f}")

            results[config.name] = result_dict

            if result.predictions:
                predictions[config.name] = result.predictions

    if predictions:
        results["difficulty_predictions"] = predictions

    return results


def convert_numpy(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj
