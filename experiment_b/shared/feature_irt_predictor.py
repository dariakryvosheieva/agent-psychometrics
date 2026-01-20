"""Feature-based IRT predictor with joint ability learning.

Unlike FeatureBasedPredictor which uses Ridge regression on ground-truth
difficulties, this predictor learns feature weights jointly with agent
abilities by optimizing the IRT log-likelihood directly.

This predictor is designed for Experiment B where:
- Training uses responses from pre-frontier agents only
- ALL tasks are seen during training (no held-out tasks)
- Evaluation uses post-frontier agents on frontier tasks

Example usage:
    from shared.feature_source import EmbeddingFeatureSource, CSVFeatureSource
    from experiment_b.shared.feature_irt_predictor import FeatureIRTPredictor

    # With embeddings
    source = EmbeddingFeatureSource(Path("embeddings.npz"))
    predictor = FeatureIRTPredictor(source, use_residuals=True)

    # Fit on pre-frontier agent responses (all tasks)
    predictor.fit(task_ids, ground_truth_b, pre_frontier_responses)

    # Predict difficulties for tasks
    predictions = predictor.predict(task_ids)
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from shared.feature_source import TaskFeatureSource


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
        use_residuals: bool = True,
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
            use_residuals: Include per-task residuals (strongly regularized).
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

            # Regularization
            reg = l2_w * torch.sum(w**2)
            if residuals_tensor is not None:
                reg = reg + l2_r * torch.sum(residuals_tensor**2)
            # Identifiability: soft constraint on mean(theta) = 0
            reg = reg + l2_a * (theta.mean() ** 2)

            loss = nll + reg
            loss.backward()
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

        All tasks must have been seen during training. This predictor does
        not support prediction on unseen tasks (use FeatureBasedPredictor
        for that use case).

        Args:
            task_ids: List of task identifiers to predict for. All must have
                     been included in the training task_ids.

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
        """Return feature weights if feature names are available.

        Returns:
            Dictionary mapping feature_name -> weight, or None if
            the feature source doesn't provide feature names.
        """
        if not self._is_fitted:
            return None

        feature_names = self.source.feature_names
        if feature_names is None:
            return None

        return {name: float(w) for name, w in zip(feature_names, self._weights)}

    @property
    def learned_abilities(self) -> Optional[Dict[str, float]]:
        """Return learned agent abilities.

        Returns:
            Dictionary mapping agent_id -> ability, or None if not fitted.
        """
        if not self._is_fitted or self._agent_ids is None:
            return None

        return {
            agent_id: float(theta)
            for agent_id, theta in zip(self._agent_ids, self._abilities)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model.

        Returns:
            Dictionary with model information.
        """
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
