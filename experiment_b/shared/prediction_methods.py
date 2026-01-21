"""Prediction methods for frontier task difficulty prediction.

This module contains:
- FeatureIRTPredictor: Joint IRT + feature learning predictor
- Helper functions for collecting predictions from various methods
- Feature source builders

All prediction method code should live here. Adding a new method should only
require changes to this file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.feature_source import (
    TaskFeatureSource,
    EmbeddingFeatureSource,
    CSVFeatureSource,
)
from experiment_ab_shared.feature_predictor import FeatureBasedPredictor
from experiment_b.shared.config_base import DatasetConfig


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class FeatureIRTResults:
    """Results from running Feature-IRT methods."""

    predictions: Dict[str, Dict[str, float]]  # method_name -> beta predictions
    abilities: Dict[str, Dict[str, float]]  # method_name -> theta abilities
    diagnostics: Optional[List[Dict[str, Any]]] = None  # Grid search diagnostics
    best_predictors: Dict[str, Any] = field(default_factory=dict)  # source_name -> predictor


# =============================================================================
# Feature-IRT Predictor
# =============================================================================


class FeatureIRTPredictor:
    """IRT model with feature-based difficulties learned jointly with abilities.

    Instead of fitting Ridge regression on ground-truth IRT difficulties,
    this directly maximizes the IRT log-likelihood:

        minimize: -Σ_ij log P(y_ij | θ_j, b_i) + regularization

    where b_i = features[i] @ w + bias + residual[i] is a linear function
    of features plus an optional per-task residual.

    This approach trains the feature-to-difficulty mapping end-to-end,
    jointly learning abilities θ_j alongside feature weights w.

    Note: This predictor is designed for Experiment B where agents are
    held out (not tasks). All tasks should be seen during training.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        use_residuals: bool = False,
        l2_weight: float = 0.01,
        l2_residual: float = 10.0,
        l2_ability: float = 0.01,
        lr: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-5,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """Initialize Feature-IRT predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            use_residuals: Include per-task residuals. Default False because
                residuals tend to overfit; the main benefit comes from joint
                ability learning, not residuals.
            l2_weight: L2 regularization on feature weights.
            l2_residual: L2 regularization on residuals (high = encourage feature usage).
            l2_ability: L2 regularization on mean(abilities)² for identifiability.
            lr: Learning rate for L-BFGS optimizer.
            max_iter: Maximum optimization iterations.
            tol: Convergence tolerance.
            device: Device for PyTorch tensors ("cpu" or "cuda"). For typical
                   experiment sizes (~100 agents, ~500 tasks), CPU is fast enough
                   (a few seconds). Use "cuda" for very large datasets.
            verbose: Print training progress.
        """
        self.source = source
        self.use_residuals = use_residuals
        self.l2_weight = l2_weight
        self.l2_residual = l2_residual
        self.l2_ability = l2_ability
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.verbose = verbose

        # Model state (set after fit())
        self._weights: Optional[np.ndarray] = None  # (feature_dim,)
        self._bias: Optional[float] = None
        self._residuals: Optional[Dict[str, float]] = None  # task_id -> residual
        self._abilities: Optional[np.ndarray] = None  # (n_agents,)
        self._agent_ids: Optional[List[str]] = None
        self._train_task_ids: Optional[List[str]] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False
        self._training_loss_history: List[float] = []
        self._loss_components: List[Dict[str, float]] = []

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        suffix = "w/ residuals" if self.use_residuals else "no residuals"
        return f"Feature-IRT ({self.source.name}, {suffix})"

    def fit(
        self,
        task_ids: List[str],
        ground_truth_b: np.ndarray,  # Unused - API compatibility
        responses: Dict[str, Dict[str, int]],
    ) -> None:
        """Fit by maximizing IRT log-likelihood jointly with abilities.

        IMPORTANT: The responses dict should ONLY contain pre-frontier agents.
        This is enforced at the call site by filtering before passing.

        Args:
            task_ids: Training task IDs (should include ALL tasks for Experiment B).
            ground_truth_b: Ignored (kept for API compatibility with other predictors).
            responses: Pre-filtered response matrix {agent_id: {task_id: 0/1}}.
                       Must only contain agents that should be used for training.
        """
        try:
            import torch
            from torch.optim import LBFGS
            from torch.distributions import Bernoulli
        except ImportError:
            raise ImportError("PyTorch is required for FeatureIRTPredictor")

        # Determine device
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            if self.verbose:
                print("   Warning: CUDA not available, falling back to CPU")
            device = "cpu"

        # Get agent IDs from responses
        agent_ids = list(responses.keys())
        n_agents = len(agent_ids)
        n_tasks = len(task_ids)

        if n_agents == 0:
            raise ValueError("No agents in responses")
        if n_tasks == 0:
            raise ValueError("No tasks to train on")

        # Store training task IDs
        self._train_task_ids = list(task_ids)
        self._agent_ids = agent_ids

        # Get features from source
        features = self.source.get_features(task_ids)  # (n_tasks, feature_dim)
        feature_dim = features.shape[1]

        # Standardize features
        self._scaler = StandardScaler()
        features_scaled = self._scaler.fit_transform(features)

        # Build response matrix: (n_agents, n_tasks)
        response_matrix = np.full((n_agents, n_tasks), np.nan)
        for i, agent_id in enumerate(agent_ids):
            for j, task_id in enumerate(task_ids):
                if task_id in responses.get(agent_id, {}):
                    response_matrix[i, j] = responses[agent_id][task_id]

        # ===== INITIALIZATION =====
        # 1. Compute empirical task difficulty for warm-start
        eps = 1e-3
        task_difficulty_init = np.zeros(n_tasks)
        for j, task_id in enumerate(task_ids):
            successes = sum(
                1 for r in responses.values() if r.get(task_id, None) == 1
            )
            total = sum(1 for r in responses.values() if task_id in r)
            if total > 0:
                acc = max(eps, min(1 - eps, successes / total))
                task_difficulty_init[j] = -np.log(acc / (1 - acc))  # -logit(acc)

        # 2. Warm-start feature weights via Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(features_scaled, task_difficulty_init)
        w_init = ridge.coef_.astype(np.float32)
        bias_init = float(ridge.intercept_)

        # 3. Compute empirical agent ability for initialization
        theta_init = np.zeros(n_agents, dtype=np.float32)
        for i, agent_id in enumerate(agent_ids):
            agent_resp = responses.get(agent_id, {})
            correct = sum(agent_resp.values())
            total = len(agent_resp)
            if total > 0:
                acc = max(eps, min(1 - eps, correct / total))
                theta_init[i] = np.log(acc / (1 - acc))  # logit(acc)

        # Convert to PyTorch tensors
        features_tensor = torch.tensor(
            features_scaled, dtype=torch.float32, device=device
        )
        response_tensor = torch.tensor(
            response_matrix, dtype=torch.float32, device=device
        )
        mask = ~torch.isnan(response_tensor)

        # Initialize learnable parameters
        w = torch.tensor(w_init, requires_grad=True, device=device)
        b = torch.tensor([bias_init], requires_grad=True, device=device)
        theta = torch.tensor(theta_init, requires_grad=True, device=device)

        if self.use_residuals:
            residuals_tensor = torch.zeros(n_tasks, requires_grad=True, device=device)
            params = [w, b, theta, residuals_tensor]
        else:
            residuals_tensor = None
            params = [w, b, theta]

        # L-BFGS optimizer
        optim = LBFGS(
            params,
            lr=self.lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self._training_loss_history = []
        self._loss_components = []
        l2_w = self.l2_weight
        l2_r = self.l2_residual
        l2_a = self.l2_ability

        def closure():
            optim.zero_grad()

            # Compute difficulties: b_i = features @ w + bias + residual
            diff = torch.matmul(features_tensor, w) + b
            if residuals_tensor is not None:
                diff = diff + residuals_tensor

            # P(success) = sigmoid(theta - diff)
            # theta: (n_agents,) -> (n_agents, 1)
            # diff: (n_tasks,) -> (1, n_tasks)
            probs = torch.sigmoid(theta[:, None] - diff[None, :])

            # Negative log-likelihood (only for valid responses)
            nll = -Bernoulli(probs=probs[mask]).log_prob(response_tensor[mask]).mean()

            # Regularization components
            weight_reg = l2_w * torch.sum(w**2)
            residual_reg = l2_r * torch.sum(residuals_tensor**2) if residuals_tensor is not None else torch.tensor(0.0)
            ability_reg = l2_a * (theta.mean() ** 2)

            loss = nll + weight_reg + residual_reg + ability_reg
            loss.backward()

            # Track loss components for diagnostics
            self._loss_components.append({
                'iter': len(self._loss_components),
                'nll': nll.item(),
                'weight_reg': weight_reg.item(),
                'residual_reg': residual_reg.item() if residuals_tensor is not None else 0.0,
                'ability_reg': ability_reg.item(),
                'total': loss.item(),
            })

            return loss

        # Training loop
        if self.verbose:
            print(f"   Feature-IRT Training: {n_tasks} tasks, {n_agents} agents")
            print(f"   Feature dim: {feature_dim}, Device: {device}")
            print(f"   Valid response pairs: {mask.sum().item()}")
            print(f"   Hyperparams: l2_weight={l2_w}, l2_residual={l2_r}")

        for iteration in range(self.max_iter):
            if iteration > 0:
                previous_loss = loss.clone().detach()

            loss = optim.step(closure)
            self._training_loss_history.append(loss.item())

            if iteration > 0:
                d_loss = abs((previous_loss - loss).item())

                if self.verbose and (iteration % 50 == 0 or iteration < 5):
                    print(
                        f"     Iter {iteration}: loss={loss.item():.6f}, d_loss={d_loss:.2e}"
                    )

                # Check convergence
                if d_loss < self.tol:
                    if self.verbose:
                        print(f"   Converged at iteration {iteration}")
                    break

        # Store learned parameters (move to CPU for numpy conversion)
        self._weights = w.detach().cpu().numpy()
        self._bias = b.detach().cpu().item()
        self._abilities = theta.detach().cpu().numpy()

        # Store residuals as dict for task_id lookup
        if residuals_tensor is not None:
            residuals_np = residuals_tensor.detach().cpu().numpy()
            self._residuals = {
                task_id: float(residuals_np[i])
                for i, task_id in enumerate(task_ids)
            }
        else:
            self._residuals = None

        self._is_fitted = True

        if self.verbose:
            print(f"   Final loss: {loss.item():.6f}")

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers to predict for.

        Returns:
            Dictionary mapping task_id -> predicted difficulty.

        Raises:
            RuntimeError: If predict() is called before fit().
            ValueError: If any task was not seen during training.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict()")

        # Validate all tasks were seen during training
        train_task_set = set(self._train_task_ids)
        unseen = [t for t in task_ids if t not in train_task_set]
        if unseen:
            raise ValueError(
                f"FeatureIRTPredictor cannot predict on unseen tasks. "
                f"{len(unseen)} tasks were not in training set. "
                f"First few: {unseen[:3]}. "
                f"Use FeatureBasedPredictor for unseen task prediction."
            )

        # Get features from source
        features = self.source.get_features(task_ids)

        # Scale features using fitted scaler
        features_scaled = self._scaler.transform(features)

        # Predict: diff = features @ w + bias + residual
        preds = features_scaled @ self._weights + self._bias

        # Add residuals
        if self._residuals is not None:
            for i, task_id in enumerate(task_ids):
                preds[i] += self._residuals[task_id]

        return {task_id: float(pred) for task_id, pred in zip(task_ids, preds)}

    @property
    def feature_weights(self) -> Optional[Dict[str, float]]:
        """Return feature weights if feature names are available."""
        if not self._is_fitted:
            return None

        feature_names = self.source.feature_names
        if feature_names is None:
            return None

        return {name: float(w) for name, w in zip(feature_names, self._weights)}

    @property
    def learned_abilities(self) -> Optional[Dict[str, float]]:
        """Return learned agent abilities."""
        if not self._is_fitted or self._agent_ids is None:
            return None

        return {
            agent_id: float(theta)
            for agent_id, theta in zip(self._agent_ids, self._abilities)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self._is_fitted:
            return {"is_fitted": False}

        info = {
            "is_fitted": True,
            "feature_source": self.source.name,
            "n_features": len(self._weights),
            "n_agents": len(self._abilities),
            "n_training_tasks": len(self._train_task_ids) if self._train_task_ids else 0,
            "use_residuals": self.use_residuals,
            "l2_weight": self.l2_weight,
            "l2_residual": self.l2_residual,
            "final_loss": (
                self._training_loss_history[-1]
                if self._training_loss_history
                else None
            ),
            "bias": self._bias,
        }

        # Add feature weights if names available
        feature_weights = self.feature_weights
        if feature_weights is not None:
            info["feature_weights"] = feature_weights

        return info

    def print_model_summary(self) -> None:
        """Print a summary of the fitted model."""
        info = self.get_model_info()

        if not info["is_fitted"]:
            print("Model not fitted yet")
            return

        print(f"  Feature source: {info['feature_source']}")
        print(f"  Number of features: {info['n_features']}")
        print(f"  Number of agents: {info['n_agents']}")
        print(f"  Training tasks: {info['n_training_tasks']}")
        print(f"  Use residuals: {info['use_residuals']}")
        print(f"  Final loss: {info['final_loss']:.6f}")
        print(f"  Bias: {info['bias']:.4f}")

        if "feature_weights" in info:
            print("  Feature weights (sorted by |weight|):")
            sorted_weights = sorted(
                info["feature_weights"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for name, weight in sorted_weights:
                print(f"    {name}: {weight:+.4f}")

    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Get training diagnostics including loss history and components."""
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before getting diagnostics")

        return {
            "loss_history": self._training_loss_history,
            "loss_components": self._loss_components,
            "n_iterations": len(self._training_loss_history),
            "final_loss": self._training_loss_history[-1] if self._training_loss_history else None,
        }

    def analyze_contributions(self) -> Dict[str, Any]:
        """Analyze variance contributions from features vs residuals.

        Computes what fraction of the predicted difficulty variance comes from
        the feature-based component vs the residual component.

        Returns:
            Dictionary with variance ratios and covariance.

        Raises:
            RuntimeError: If predictor is not fitted or residuals are not enabled.
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before analyzing contributions")

        if self._residuals is None:
            raise RuntimeError(
                "analyze_contributions() requires residuals to be enabled. "
                "This predictor was fitted with use_residuals=False."
            )

        # Get scaled features for training tasks
        features = self.source.get_features(self._train_task_ids)
        features_scaled = self._scaler.transform(features)

        # Compute feature-based difficulty component
        feature_component = features_scaled @ self._weights

        # Compute residual component
        residual_component = np.array([
            self._residuals[t] for t in self._train_task_ids
        ])

        # Total difficulty (excluding constant bias which doesn't affect variance)
        total_difficulty = feature_component + residual_component

        # Compute variances
        var_feature = float(np.var(feature_component))
        var_residual = float(np.var(residual_component))
        var_total = float(np.var(total_difficulty))

        # Compute covariance
        if len(feature_component) > 1:
            covariance = float(np.cov(feature_component, residual_component)[0, 1])
        else:
            covariance = 0.0

        return {
            "var_feature": var_feature,
            "var_residual": var_residual,
            "var_total": var_total,
            "feature_ratio": var_feature / var_total if var_total > 0 else 0.0,
            "residual_ratio": var_residual / var_total if var_total > 0 else 0.0,
            "covariance": covariance,
        }


# =============================================================================
# Feature Source Builders
# =============================================================================


def build_feature_sources(
    config: DatasetConfig,
    embeddings_path_override: Optional[Path] = None,
    llm_judge_path_override: Optional[Path] = None,
) -> List[Tuple[str, TaskFeatureSource]]:
    """Build list of available feature sources.

    Args:
        config: Dataset configuration
        embeddings_path_override: Optional path to override config embeddings
        llm_judge_path_override: Optional path to override config LLM judge features

    Returns:
        List of (source_name, feature_source) tuples
    """
    embeddings_path = embeddings_path_override or config.embeddings_path
    llm_judge_path = llm_judge_path_override or config.llm_judge_path

    feature_sources: List[Tuple[str, TaskFeatureSource]] = []

    if embeddings_path and embeddings_path.exists():
        feature_sources.append(("Embedding", EmbeddingFeatureSource(embeddings_path)))
    else:
        print(f"\nEmbeddings not found: {embeddings_path}")

    if llm_judge_path and llm_judge_path.exists():
        feature_sources.append((
            "LLM Judge",
            CSVFeatureSource(
                llm_judge_path,
                feature_cols=config.llm_judge_feature_cols,
                name="LLM Judge",
            ),
        ))
    else:
        print(f"\nLLM Judge features not found: {llm_judge_path}")

    return feature_sources


# =============================================================================
# Prediction Collection Functions
# =============================================================================


def collect_ridge_predictions(
    feature_sources: List[Tuple[str, TaskFeatureSource]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
    all_task_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Train Ridge regressors on available feature sources.

    Args:
        feature_sources: List of (source_name, feature_source) tuples
        train_task_ids: Task IDs for training
        ground_truth_b: Ground truth difficulties from baseline IRT
        all_task_ids: All task IDs to predict for

    Returns:
        Dict mapping method_name -> predictions dict
    """
    predictions: Dict[str, Dict[str, float]] = {}

    for source_name, source in feature_sources:
        method_name = f"{source_name} + Ridge"
        print(f"\nTraining {method_name}...")
        print(f"  Training on {len(train_task_ids)} tasks")
        try:
            predictor = FeatureBasedPredictor(source)
            predictor.fit(train_task_ids, ground_truth_b)
            predictions[method_name] = predictor.predict(all_task_ids)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    return predictions


def collect_feature_irt_predictions(
    feature_sources: List[Tuple[str, TaskFeatureSource]],
    train_task_ids: List[str],
    ground_truth_b: np.ndarray,
    train_responses: Dict[str, Dict[str, int]],
    oracle_items: pd.DataFrame,
    oracle_abilities: pd.DataFrame,
    responses: Dict[str, Dict[str, int]],
    frontier_task_ids: List[str],
    anchor_task_ids: List[str],
    post_frontier_agents: List[str],
    alignment_method: str = "affine",
    grid_search: bool = False,
    diagnostic_mode: bool = False,
    verbose: bool = False,
) -> FeatureIRTResults:
    """Run Feature-IRT with optional grid search.

    Args:
        feature_sources: List of (source_name, feature_source) tuples
        train_task_ids: Task IDs for training (all tasks)
        ground_truth_b: Ground truth difficulties from baseline IRT
        train_responses: Pre-filtered responses (pre-frontier agents only)
        oracle_items: Oracle IRT items for evaluation
        oracle_abilities: Oracle abilities for evaluation
        responses: Full responses for evaluation
        frontier_task_ids: Frontier tasks for metric computation
        anchor_task_ids: Anchor tasks for scale alignment
        post_frontier_agents: Post-frontier agents for evaluation
        alignment_method: Scale alignment method
        grid_search: Whether to run grid search
        diagnostic_mode: Whether to run extended diagnostics
        verbose: Print verbose output

    Returns:
        FeatureIRTResults with predictions, abilities, and optional diagnostics
    """
    # Import here to avoid circular dependency
    from experiment_b.shared.evaluation import compute_method_metrics

    if diagnostic_mode:
        l2_weight_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        l2_residual_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
        use_residuals_grid = [True]
    elif grid_search:
        l2_weight_grid = [0.001, 0.01, 0.1]
        l2_residual_grid = [10.0]
        use_residuals_grid = [False]
    else:
        l2_weight_grid = [0.01]
        l2_residual_grid = [10.0]
        use_residuals_grid = [False]

    predictions: Dict[str, Dict[str, float]] = {}
    abilities: Dict[str, Dict[str, float]] = {}
    diagnostics: Optional[List[Dict[str, Any]]] = [] if diagnostic_mode else None
    best_predictors: Dict[str, Any] = {}

    for source_name, source in feature_sources:
        method_name = f"Feature-IRT ({source_name})"
        best_auc = -1.0
        best_predictions: Optional[Dict[str, float]] = None
        best_abilities: Optional[Dict[str, float]] = None
        best_predictor: Optional[Any] = None

        if grid_search or diagnostic_mode:
            print(f"\nRunning {method_name} grid search...")
            print(f"  Training on {len(train_task_ids)} tasks")
            if diagnostic_mode:
                print(f"  Extended diagnostic grid: {len(l2_weight_grid)}x{len(l2_residual_grid)}x{len(use_residuals_grid)} configs")

        for l2_w in l2_weight_grid:
            for l2_r in l2_residual_grid:
                for use_res in use_residuals_grid:
                    if grid_search or diagnostic_mode:
                        print(f"    Testing: l2_w={l2_w}, l2_r={l2_r}, res={use_res}")

                    try:
                        predictor = FeatureIRTPredictor(
                            source,
                            use_residuals=use_res,
                            l2_weight=l2_w,
                            l2_residual=l2_r,
                            verbose=verbose and not (grid_search or diagnostic_mode),
                        )

                        if not (grid_search or diagnostic_mode):
                            print(f"\nTraining {method_name}...")
                            print(f"  Training on {len(train_task_ids)} tasks with {len(train_responses)} pre-frontier agents")

                        predictor.fit(
                            task_ids=train_task_ids,
                            ground_truth_b=ground_truth_b,
                            responses=train_responses,
                        )

                        preds = predictor.predict(train_task_ids)

                        if grid_search or diagnostic_mode:
                            metrics = compute_method_metrics(
                                predicted_beta=preds,
                                oracle_items=oracle_items,
                                oracle_abilities=oracle_abilities,
                                responses=responses,
                                frontier_task_ids=frontier_task_ids,
                                anchor_task_ids=anchor_task_ids,
                                eval_agents=post_frontier_agents,
                                alignment_method=alignment_method,
                            )
                            auc = metrics.get('auc', 0) or 0
                            print(f"      AUC: {auc:.4f}")

                            if diagnostic_mode and use_res:
                                contributions = predictor.analyze_contributions()
                                diag = predictor.get_training_diagnostics()
                                diagnostics.append({
                                    'source': source_name,
                                    'l2_weight': l2_w,
                                    'l2_residual': l2_r,
                                    'use_residuals': use_res,
                                    'auc': auc,
                                    'final_loss': diag['final_loss'],
                                    'n_iterations': diag['n_iterations'],
                                    'loss_history': diag['loss_history'],
                                    'loss_components': diag['loss_components'],
                                    'contributions': contributions,
                                })
                        else:
                            auc = 1.0  # Not in grid search mode, just use this config

                        if auc > best_auc:
                            best_auc = auc
                            best_predictions = preds
                            best_abilities = predictor.learned_abilities
                            best_predictor = predictor

                    except Exception as e:
                        if grid_search or diagnostic_mode:
                            print(f"      Error: {e}")
                        else:
                            print(f"  Error: {e}")
                            import traceback
                            traceback.print_exc()

        if best_predictions is not None:
            if grid_search or diagnostic_mode:
                print(f"  Best AUC: {best_auc:.4f}")
            predictions[method_name] = best_predictions
            if best_abilities is not None:
                abilities[method_name] = best_abilities
            if best_predictor is not None:
                best_predictors[source_name] = best_predictor

    return FeatureIRTResults(
        predictions=predictions,
        abilities=abilities,
        diagnostics=diagnostics,
        best_predictors=best_predictors,
    )


def collect_sad_irt_predictions(
    sad_irt_beta_dir: Path,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Load SAD-IRT beta and theta values from CSV files.

    Each SAD-IRT run is loaded as a separate method. The best run can be
    selected by AUC during evaluation.

    Args:
        sad_irt_beta_dir: Directory containing extracted SAD-IRT beta CSV files

    Returns:
        Tuple of (predictions_dict, abilities_dict) where each maps
        method_name -> task/agent -> value
    """
    predictions: Dict[str, Dict[str, float]] = {}
    abilities: Dict[str, Dict[str, float]] = {}

    if not sad_irt_beta_dir.exists():
        print(f"\nSAD-IRT beta directory not found: {sad_irt_beta_dir}")
        print("  To include SAD-IRT results, run experiment_sad_irt and extract beta values")
        return predictions, abilities

    beta_files = list(sad_irt_beta_dir.glob("*.csv"))
    print(f"\nLoading SAD-IRT beta values from {sad_irt_beta_dir}...")
    print(f"  Found {len(beta_files)} beta CSV files")

    sad_irt_theta_dir = Path("chris_output/sad_irt_theta_values")
    loaded_count = 0

    for beta_file in beta_files:
        beta_df = pd.read_csv(beta_file, index_col=0)
        if "beta" not in beta_df.columns:
            print(f"  Skipping {beta_file.name}: no 'beta' column")
            continue

        stem = beta_file.stem
        method_name = f"SAD-IRT ({stem})"
        predictions[method_name] = beta_df["beta"].to_dict()
        loaded_count += 1

        # Load matching theta file if available
        theta_file = sad_irt_theta_dir / f"{stem}.csv"
        if theta_file.exists():
            theta_df = pd.read_csv(theta_file, index_col=0)
            if "theta" in theta_df.columns:
                abilities[method_name] = theta_df["theta"].to_dict()

    print(f"  Loaded {loaded_count} valid SAD-IRT beta files")

    return predictions, abilities
