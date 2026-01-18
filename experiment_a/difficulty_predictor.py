"""Difficulty predictor base class and implementations.

All difficulty predictors inherit from DifficultyPredictorBase and implement
the fit() and predict() methods for use in Experiment A.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DifficultyPredictorBase(ABC):
    """Abstract base class for all difficulty predictors.

    Provides the common interface that all predictors must implement.
    """

    @abstractmethod
    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Train on tasks with known IRT difficulties.

        Args:
            task_ids: List of task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        ...

    @abstractmethod
    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable predictor name."""
        ...


class EmbeddingPredictor(DifficultyPredictorBase):
    """Difficulty predictor using pre-computed embeddings + Ridge regression.

    Based on Daria's predict_question_difficulty.py pipeline.
    Requires a pre-computed embeddings .npz file.
    """

    def __init__(
        self,
        embeddings_path: Path,
        ridge_alpha: Optional[float] = None,
        ridge_alphas: Optional[List[float]] = None,
    ):
        """Initialize embedding predictor.

        Args:
            embeddings_path: Path to pre-computed embeddings .npz file
            ridge_alpha: Fixed Ridge alpha (if provided, skips CV)
            ridge_alphas: List of alphas for RidgeCV (default: use CV with standard alphas)
        """
        self.embeddings_path = embeddings_path
        self.ridge_alpha = ridge_alpha
        self.ridge_alphas = ridge_alphas or [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        self._model: Optional[Pipeline] = None
        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._embedding_dim: Optional[int] = None
        self._best_alpha: Optional[float] = None

        # Load embeddings immediately
        self._load_embeddings()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Embedding"

    def _load_embeddings(self) -> None:
        """Load embeddings from .npz file."""
        data = np.load(self.embeddings_path, allow_pickle=True)

        # Extract task IDs and embedding matrix
        task_ids = [str(x) for x in data["task_ids"].tolist()]
        X = data["X"].astype(np.float32)

        self._embedding_dim = int(X.shape[1])
        self._embeddings = {task_id: X[i] for i, task_id in enumerate(task_ids)}

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Ridge regression on task embeddings.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for training tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]
        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from embeddings")

        # Build training matrix
        X = np.stack([self._embeddings[t] for t in available_tasks])
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Fit StandardScaler + Ridge (with CV or fixed alpha)
        if self.ridge_alpha is not None:
            ridge = Ridge(alpha=self.ridge_alpha)
            self._best_alpha = self.ridge_alpha
        else:
            ridge = RidgeCV(alphas=self.ridge_alphas, cv=5)

        self._model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", ridge),
        ])
        self._model.fit(X, y)

        # Store best alpha if using CV
        if hasattr(self._model.named_steps["ridge"], "alpha_"):
            self._best_alpha = float(self._model.named_steps["ridge"].alpha_)

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for prediction tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]

        if not available_tasks:
            return {}

        X = np.stack([self._embeddings[t] for t in available_tasks])
        preds = self._model.predict(X)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    @property
    def n_embeddings(self) -> int:
        """Return number of loaded embeddings."""
        return len(self._embeddings) if self._embeddings else 0

    @property
    def best_alpha(self) -> Optional[float]:
        """Return the best alpha (from CV or fixed)."""
        return self._best_alpha


class MLEEmbeddingPredictor(DifficultyPredictorBase):
    """Difficulty predictor using direct MLE on embeddings.

    Instead of fitting regression on ground-truth difficulties (plug-in estimator),
    this directly maximizes the IRT log-likelihood:

        maximize: Σ_agents Σ_tasks log P(y_ij | θ_j, β_i)

    where β_i = embeddings[i] @ w + b is a linear function of embeddings.

    This approach trains the embedding-to-difficulty mapping end-to-end to predict
    agent responses, rather than fitting to IRT difficulty estimates as an
    intermediate target.

    Approach inspired by Truong et al. (2025) "Reliable and Efficient Amortized
    Model-based Evaluation" (https://arxiv.org/pdf/2503.13335)
    """

    def __init__(
        self,
        embeddings_path: Path,
        lr: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-5,
        l2_lambda: float = 0.01,
        use_mc_abilities: bool = False,
        n_mc_samples: int = 100,
        verbose: bool = True,
    ):
        """Initialize MLE embedding predictor.

        Args:
            embeddings_path: Path to pre-computed embeddings .npz file
            lr: Learning rate for L-BFGS optimizer
            max_iter: Maximum number of L-BFGS iterations
            tol: Convergence tolerance for loss, weight change, and gradient
            l2_lambda: L2 regularization strength on weights
            use_mc_abilities: If True, marginalize over abilities using MC sampling
                from N(0,1) prior instead of using fixed IRT-estimated abilities.
                This matches the approach in Truong et al. (2025).
            n_mc_samples: Number of MC samples for ability marginalization
            verbose: Whether to print training progress
        """
        self.embeddings_path = embeddings_path
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.l2_lambda = l2_lambda
        self.use_mc_abilities = use_mc_abilities
        self.n_mc_samples = n_mc_samples
        self.verbose = verbose

        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._embedding_dim: Optional[int] = None
        self._weights: Optional[np.ndarray] = None  # (embed_dim,)
        self._bias: Optional[float] = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._training_loss_history: List[float] = []

        # Load embeddings immediately
        self._load_embeddings()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        if self.use_mc_abilities:
            return "Embedding (MLE-MC)"
        return "Embedding (MLE)"

    def _load_embeddings(self) -> None:
        """Load embeddings from .npz file."""
        data = np.load(self.embeddings_path, allow_pickle=True)

        # Extract task IDs and embedding matrix
        task_ids = [str(x) for x in data["task_ids"].tolist()]
        X = data["X"].astype(np.float32)

        self._embedding_dim = int(X.shape[1])
        self._embeddings = {task_id: X[i] for i, task_id in enumerate(task_ids)}

    def fit(
        self,
        task_ids: List[str],
        ground_truth_b: np.ndarray,
        abilities: pd.DataFrame,
        responses: Dict[str, Dict[str, int]],
    ) -> None:
        """Fit by maximizing IRT log-likelihood.

        Note: This method requires additional arguments (abilities, responses)
        compared to the base class, since we train on the response matrix
        rather than ground-truth difficulties.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values (unused, kept for API)
            abilities: DataFrame with index=agent_id, column 'theta'
            responses: Dict mapping agent_id -> {task_id -> 0|1}
        """
        try:
            import torch
            from torch.optim import LBFGS
            from torch.distributions import Bernoulli
        except ImportError:
            raise ImportError("PyTorch is required for MLEEmbeddingPredictor")

        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for training tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]
        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from embeddings")

        if len(available_tasks) == 0:
            raise ValueError("No tasks available for training")

        # Build embedding matrix for training tasks: (n_tasks, embed_dim)
        embed_matrix = np.stack([self._embeddings[t] for t in available_tasks])

        # Standardize embeddings
        self._scaler_mean = embed_matrix.mean(axis=0)
        self._scaler_std = embed_matrix.std(axis=0) + 1e-8
        embed_scaled = (embed_matrix - self._scaler_mean) / self._scaler_std

        # Get agent IDs that have responses for at least one training task
        agent_ids = [a for a in abilities.index if a in responses]

        # Build response matrix: (n_agents, n_tasks)
        # Value is 0, 1, or NaN (missing)
        response_matrix = np.full((len(agent_ids), len(available_tasks)), np.nan)
        for i, agent_id in enumerate(agent_ids):
            for j, task_id in enumerate(available_tasks):
                if task_id in responses.get(agent_id, {}):
                    response_matrix[i, j] = responses[agent_id][task_id]

        # Get agent abilities: (n_agents,)
        thetas = np.array([abilities.loc[a, "theta"] for a in agent_ids])

        # Convert to PyTorch tensors
        device = "cpu"  # Small enough for CPU
        embed_tensor = torch.tensor(embed_scaled, dtype=torch.float32, device=device)
        response_tensor = torch.tensor(response_matrix, dtype=torch.float32, device=device)
        theta_tensor = torch.tensor(thetas, dtype=torch.float32, device=device)

        # Mask for valid (non-NaN) responses
        mask = ~torch.isnan(response_tensor)

        # Initialize weights and bias
        # Note: z = embed @ w + b, and P(success) = sigmoid(theta - z)
        # So z represents difficulty (positive z = harder)
        w = torch.zeros(self._embedding_dim, requires_grad=True, device=device)
        b = torch.zeros(1, requires_grad=True, device=device)

        # L-BFGS optimizer
        optim = LBFGS(
            [w, b],
            lr=self.lr,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        self._training_loss_history = []

        l2_lambda = self.l2_lambda
        use_mc = self.use_mc_abilities
        n_mc = self.n_mc_samples
        n_agents = len(agent_ids)
        n_tasks = len(available_tasks)

        # Pre-generate MC samples for ability marginalization (if enabled)
        # Shape: (n_mc_samples, n_agents)
        if use_mc:
            torch.manual_seed(42)  # Reproducibility
            mc_thetas = torch.randn(n_mc, n_agents, device=device)

        def closure():
            optim.zero_grad()
            # z = difficulty prediction: (n_tasks,)
            z = torch.matmul(embed_tensor, w) + b

            if use_mc:
                # Monte Carlo marginalization over abilities
                # mc_thetas: (n_mc, n_agents)
                # z: (n_tasks,)
                # probs: (n_mc, n_agents, n_tasks)
                probs = torch.sigmoid(mc_thetas[:, :, None] - z[None, None, :])

                # Log-likelihood for each MC sample: (n_mc,)
                # Average over MC samples in log-space using logsumexp
                log_probs = Bernoulli(probs=probs).log_prob(response_tensor[None, :, :])
                # Mask and sum over valid responses: (n_mc,)
                log_probs_masked = (log_probs * mask[None, :, :]).sum(dim=(1, 2))
                # Monte Carlo average (log-space): log(1/M * sum(exp(log_p)))
                log_marginal = torch.logsumexp(log_probs_masked, dim=0) - np.log(n_mc)
                # Normalize by number of valid responses
                nll = -log_marginal / mask.sum()
            else:
                # Fixed abilities approach
                # P(success) = sigmoid(theta - z)
                # theta_tensor: (n_agents,) -> (n_agents, 1)
                # z: (n_tasks,) -> (1, n_tasks)
                probs = torch.sigmoid(theta_tensor[:, None] - z[None, :])

                # Negative log-likelihood (only for valid responses)
                nll = -Bernoulli(probs=probs[mask]).log_prob(response_tensor[mask]).mean()

            # L2 regularization on weights
            l2_reg = l2_lambda * torch.sum(w ** 2)

            loss = nll + l2_reg
            loss.backward()
            return loss

        # Training loop
        if self.verbose:
            mode = "MC marginalization" if use_mc else "fixed abilities"
            print(f"   MLE Training ({mode}): {n_tasks} tasks, {n_agents} agents")
            if use_mc:
                print(f"   MC samples: {n_mc}")
            print(f"   Valid response pairs: {mask.sum().item()}")

        for iteration in range(self.max_iter):
            if iteration > 0:
                previous_w = w.clone().detach()
                previous_loss = loss.clone().detach()

            loss = optim.step(closure)
            self._training_loss_history.append(loss.item())

            if iteration > 0:
                d_loss = (previous_loss - loss).item()
                d_w = torch.norm(previous_w - w.detach(), p=2).item()
                grad_norm = w.grad.abs().max().item() if w.grad is not None else 0

                if self.verbose and (iteration % 10 == 0 or iteration < 5):
                    print(f"     Iter {iteration}: loss={loss.item():.6f}, "
                          f"d_loss={d_loss:.2e}, d_w={d_w:.2e}, grad={grad_norm:.2e}")

                # Check convergence
                if abs(d_loss) < self.tol and d_w < self.tol and grad_norm < self.tol:
                    if self.verbose:
                        print(f"   Converged at iteration {iteration}")
                    break

        # Store learned parameters
        self._weights = w.detach().numpy()
        self._bias = b.detach().item()

        if self.verbose:
            print(f"   Final loss: {loss.item():.6f}")

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._weights is None or self._bias is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for prediction tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]

        if not available_tasks:
            return {}

        # Build embedding matrix and scale
        embed_matrix = np.stack([self._embeddings[t] for t in available_tasks])
        embed_scaled = (embed_matrix - self._scaler_mean) / self._scaler_std

        # Predict: z = embed @ w + b
        preds = embed_scaled @ self._weights + self._bias

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    @property
    def n_embeddings(self) -> int:
        """Return number of loaded embeddings."""
        return len(self._embeddings) if self._embeddings else 0

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Return learned weights."""
        return self._weights

    @property
    def bias(self) -> Optional[float]:
        """Return learned bias."""
        return self._bias

    @property
    def training_loss_history(self) -> List[float]:
        """Return training loss history."""
        return self._training_loss_history


class EmbeddingSimilarityPredictor(DifficultyPredictorBase):
    """Difficulty predictor using embedding similarity distributions.

    Instead of fitting a linear model directly on embeddings, this predictor:
    1. Computes cosine similarity between each task and all training tasks
    2. Extracts distributional statistics from these similarities
    3. Fits a Ridge regression on these statistics to predict difficulty

    This approach captures how "typical" or "isolated" a task is relative
    to the training distribution, using only ~10 features instead of 4096.
    """

    FEATURE_NAMES = [
        "sim_mean", "sim_std", "sim_min", "sim_max", "sim_median",
        "sim_p25", "sim_p75", "sim_p90", "sim_skew", "sim_kurtosis"
    ]

    def __init__(
        self,
        embeddings_path: Path,
        ridge_alpha: float = 1.0,
    ):
        """Initialize embedding similarity predictor.

        Args:
            embeddings_path: Path to pre-computed embeddings .npz file
            ridge_alpha: Ridge regression regularization parameter
        """
        self.embeddings_path = embeddings_path
        self.ridge_alpha = ridge_alpha

        # Embeddings storage
        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._embedding_dim: Optional[int] = None

        # Training state
        self._train_task_ids: Optional[List[str]] = None
        self._train_embeddings: Optional[np.ndarray] = None  # (n_train, dim)

        # Model
        self._model: Optional[Pipeline] = None
        self._feature_coefficients: Optional[Dict[str, float]] = None

        # Load embeddings immediately
        self._load_embeddings()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Embedding Similarity"

    def _load_embeddings(self) -> None:
        """Load embeddings from .npz file."""
        data = np.load(self.embeddings_path, allow_pickle=True)

        # Extract task IDs and embedding matrix
        task_ids = [str(x) for x in data["task_ids"].tolist()]
        X = data["X"].astype(np.float32)

        self._embedding_dim = int(X.shape[1])
        self._embeddings = {task_id: X[i] for i, task_id in enumerate(task_ids)}

    def _compute_similarity_features(
        self,
        query_embedding: np.ndarray,
        reference_embeddings: np.ndarray,
        exclude_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute distributional statistics from cosine similarity scores.

        Args:
            query_embedding: (dim,) embedding of the query task
            reference_embeddings: (n_ref, dim) embeddings of reference tasks
            exclude_idx: If provided, exclude this index (for leave-one-out)

        Returns:
            (n_features,) array of distributional statistics
        """
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        ref_norms = reference_embeddings / (
            np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Compute cosine similarities
        similarities = ref_norms @ query_norm  # (n_ref,)

        # Exclude self if needed (for leave-one-out during training)
        if exclude_idx is not None:
            similarities = np.delete(similarities, exclude_idx)

        # Extract distributional statistics
        return np.array([
            np.mean(similarities),
            np.std(similarities),
            np.min(similarities),
            np.max(similarities),
            np.median(similarities),
            np.percentile(similarities, 25),
            np.percentile(similarities, 75),
            np.percentile(similarities, 90),
            scipy_stats.skew(similarities),
            scipy_stats.kurtosis(similarities),
        ])

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Ridge regression on similarity-based features.

        Uses leave-one-out during training to avoid data leakage.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")

        # Get embeddings for training tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]
        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from embeddings")

        # Store training state
        self._train_task_ids = available_tasks
        self._train_embeddings = np.stack(
            [self._embeddings[t] for t in available_tasks]
        )

        # Get corresponding ground truth values
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Compute features for each training task using leave-one-out
        features = []
        for i in range(len(available_tasks)):
            feat = self._compute_similarity_features(
                self._train_embeddings[i],
                self._train_embeddings,
                exclude_idx=i,  # Exclude self to avoid leakage
            )
            features.append(feat)

        X = np.stack(features)

        # Fit StandardScaler + Ridge
        self._model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=self.ridge_alpha)),
        ])
        self._model.fit(X, y)

        # Store coefficients for interpretability
        ridge_coefs = self._model.named_steps["ridge"].coef_
        self._feature_coefficients = dict(zip(self.FEATURE_NAMES, ridge_coefs.tolist()))

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks using similarity to training set.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded")
        if self._train_embeddings is None:
            raise RuntimeError("Training embeddings not stored")

        # Get embeddings for prediction tasks
        available_tasks = [t for t in task_ids if t in self._embeddings]

        if not available_tasks:
            return {}

        # Compute features for each test task (no exclusion needed)
        features = []
        for t in available_tasks:
            feat = self._compute_similarity_features(
                self._embeddings[t],
                self._train_embeddings,
                exclude_idx=None,  # Compare to all training tasks
            )
            features.append(feat)

        X = np.stack(features)
        preds = self._model.predict(X)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    @property
    def n_embeddings(self) -> int:
        """Return number of loaded embeddings."""
        return len(self._embeddings) if self._embeddings else 0

    @property
    def n_train_tasks(self) -> int:
        """Return number of training tasks."""
        return len(self._train_task_ids) if self._train_task_ids else 0

    @property
    def feature_names(self) -> List[str]:
        """Return names of similarity features."""
        return self.FEATURE_NAMES

    @property
    def feature_coefficients(self) -> Optional[Dict[str, float]]:
        """Return coefficients of features."""
        return self._feature_coefficients

    def print_feature_coefficients(self) -> None:
        """Print feature coefficients for interpretability."""
        if self._feature_coefficients is None:
            print("Model not fitted yet")
            return

        print(f"\n   Similarity feature coefficients:")
        sorted_features = sorted(
            self._feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, coef in sorted_features:
            sign = "+" if coef >= 0 else ""
            print(f"     {name:15s}: {sign}{coef:.4f}")


class ConstantPredictor(DifficultyPredictorBase):
    """Baseline: predict mean difficulty for all tasks."""

    def __init__(self):
        self._mean_b: Optional[float] = None

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Constant"

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Compute mean difficulty from training data.

        Args:
            task_ids: List of training task identifiers (unused)
            ground_truth_b: Array of ground truth difficulty values
        """
        self._mean_b = float(np.mean(ground_truth_b))

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict mean difficulty for all tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to mean difficulty
        """
        if self._mean_b is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return {t: self._mean_b for t in task_ids}


class GroundTruthPredictor(DifficultyPredictorBase):
    """Oracle: use actual IRT difficulties (upper bound baseline)."""

    def __init__(self, items_df: pd.DataFrame):
        """Initialize with ground truth items.

        Args:
            items_df: DataFrame with index=task_id, column 'b' for difficulty
        """
        self._items = items_df

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Oracle"

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """No training needed for oracle.

        Args:
            task_ids: Unused
            ground_truth_b: Unused
        """
        pass  # No training needed

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Return actual IRT difficulties.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to ground truth difficulty
        """
        predictions = {}
        for t in task_ids:
            if t in self._items.index:
                predictions[t] = float(self._items.loc[t, "b"])
        return predictions


class LunettePredictor(DifficultyPredictorBase):
    """Difficulty predictor using Lunette-extracted features + Ridge regression.

    Includes automatic feature selection using LassoCV for sparse selection.
    """

    # Default feature columns to use (exclude metadata and reasoning)
    DEFAULT_FEATURE_COLS = [
        "repo_file_count",
        "repo_line_count",
        "patch_file_count",
        "patch_line_count",
        "test_file_count",
        "related_file_count",
        "import_count",
        "class_count_in_file",
        "function_count_in_file",
        "test_count_fail_to_pass",
        "test_count_pass_to_pass",
        "git_commit_count",
        "directory_depth",
        "has_conftest",
        "has_init",
        "fix_in_description",
        "problem_clarity",
        "error_message_provided",
        "reproduction_steps",
        "fix_locality",
        "domain_knowledge_required",
        "fix_complexity",
        "logical_reasoning_required",
        "atypicality",
    ]

    def __init__(
        self,
        features_path: Path,
        ridge_alpha: Optional[float] = None,
        ridge_alphas: Optional[List[float]] = None,
        feature_selection: str = "lasso_cv",
        max_features: Optional[int] = 10,
        feature_cols: Optional[List[str]] = None,
    ):
        """Initialize Lunette predictor.

        Args:
            features_path: Path to CSV file with Lunette features
            ridge_alpha: Fixed Ridge alpha (if provided, skips CV)
            ridge_alphas: List of alphas for RidgeCV (default: use CV with standard alphas)
            feature_selection: Method for feature selection ("lasso_cv" or "select_k_best")
            max_features: Maximum number of features to select (None = no limit)
            feature_cols: List of feature columns to use (None = use defaults)
        """
        self.features_path = Path(features_path)
        self.ridge_alpha = ridge_alpha
        self.ridge_alphas = ridge_alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS

        self._model: Optional[Ridge] = None
        self._best_alpha: Optional[float] = None
        self._scaler: Optional[StandardScaler] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        self._feature_coefficients: Optional[Dict[str, float]] = None

        # Load features immediately
        self._load_features()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "Lunette"

    def _load_features(self) -> None:
        """Load features from CSV file."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        self._features_df = pd.read_csv(self.features_path)

        # Set index to task/instance ID column
        if "_instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("_instance_id")
        elif "instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("instance_id")
        elif "task_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("task_id")

        # Filter to available feature columns
        available_cols = [c for c in self.feature_cols if c in self._features_df.columns]
        if len(available_cols) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_cols)
            print(f"Warning: Missing feature columns: {missing}")

        self.feature_cols = available_cols

    def _get_feature_matrix(self, task_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix for given task IDs.

        Returns:
            (X, available_task_ids) where X is (n_tasks, n_features)
        """
        if self._features_df is None:
            raise RuntimeError("Features not loaded")

        # Filter to available tasks
        available_tasks = [t for t in task_ids if t in self._features_df.index]

        if not available_tasks:
            return np.array([]).reshape(0, len(self.feature_cols)), []

        # Extract feature matrix
        X = self._features_df.loc[available_tasks, self.feature_cols].values.astype(np.float32)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        return X, available_tasks

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Ridge regression with feature selection.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from Lunette features")

        if len(available_tasks) == 0:
            raise ValueError("No tasks available for training")

        # Get corresponding ground truth values
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Step 1: Feature selection
        if self.feature_selection == "lasso_cv":
            self._fit_with_lasso_selection(X, y)
        elif self.feature_selection == "select_k_best":
            self._fit_with_kbest_selection(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")

    def _fit_with_lasso_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using Lasso for feature selection, then Ridge for final model."""
        # Normalize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Lasso for feature selection
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y)

        # Get non-zero coefficients
        coef_abs = np.abs(lasso.coef_)
        nonzero_mask = coef_abs > 1e-6

        # Select features
        if self.max_features and np.sum(nonzero_mask) > self.max_features:
            # Take top k by absolute coefficient
            top_k_idx = np.argsort(coef_abs)[-self.max_features:]
            selected_mask = np.zeros(len(self.feature_cols), dtype=bool)
            selected_mask[top_k_idx] = True
        elif np.sum(nonzero_mask) == 0:
            # No features selected, use top k by correlation
            print("Warning: Lasso selected 0 features, falling back to top-k correlation")
            k = self.max_features or 5
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
        else:
            selected_mask = nonzero_mask

        self._selected_features = [
            self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]
        ]

        # Fit Ridge on selected features (with CV or fixed alpha)
        X_selected = X_scaled[:, selected_mask]
        if self.ridge_alpha is not None:
            self._model = Ridge(alpha=self.ridge_alpha)
            self._best_alpha = self.ridge_alpha
        else:
            self._model = RidgeCV(alphas=self.ridge_alphas, cv=5)
        self._model.fit(X_selected, y)

        # Store best alpha if using CV
        if hasattr(self._model, "alpha_"):
            self._best_alpha = float(self._model.alpha_)

        # Store coefficients for reporting
        self._feature_coefficients = dict(
            zip(self._selected_features, self._model.coef_.tolist())
        )
        self._selected_mask = selected_mask

    def _fit_with_kbest_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using SelectKBest for feature selection, then Ridge for final model."""
        # Normalize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # SelectKBest
        k = self.max_features or 10
        selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_mask = selector.get_support()

        self._selected_features = [
            self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]
        ]

        # Fit Ridge on selected features
        self._model = Ridge(alpha=self.ridge_alpha)
        self._model.fit(X_selected, y)

        # Store coefficients for reporting
        self._feature_coefficients = dict(
            zip(self._selected_features, self._model.coef_.tolist())
        )
        self._selected_mask = selected_mask

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if not available_tasks:
            return {}

        # Transform and select features
        X_scaled = self._scaler.transform(X)
        X_selected = X_scaled[:, self._selected_mask]

        # Predict
        preds = self._model.predict(X_selected)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def selected_features(self) -> Optional[List[str]]:
        """Return names of selected features."""
        return self._selected_features

    @property
    def feature_coefficients(self) -> Optional[Dict[str, float]]:
        """Return coefficients of selected features."""
        return self._feature_coefficients

    @property
    def n_features(self) -> int:
        """Return number of available features."""
        return len(self.feature_cols)

    @property
    def n_tasks(self) -> int:
        """Return number of tasks with features."""
        return len(self._features_df) if self._features_df is not None else 0

    @property
    def best_alpha(self) -> Optional[float]:
        """Return the best alpha (from CV or fixed)."""
        return self._best_alpha

    def print_selected_features(self) -> None:
        """Print selected features and their coefficients."""
        if self._feature_coefficients is None:
            print("Model not fitted yet")
            return

        print(f"\nSelected features ({self.feature_selection}, n={len(self._selected_features)}):")
        sorted_features = sorted(
            self._feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, coef in sorted_features:
            sign = "+" if coef >= 0 else ""
            print(f"  {name:30s}: {sign}{coef:.4f}")


class LLMJudgePredictor(DifficultyPredictorBase):
    """Difficulty predictor using LLM-extracted semantic features + Lasso/Ridge regression.

    Uses only the 9 semantic features (no environment/sandbox features):
    - fix_in_description, problem_clarity, error_message_provided, reproduction_steps
    - fix_locality, domain_knowledge_required, fix_complexity
    - logical_reasoning_required, atypicality

    This is the ablation of LunettePredictor that doesn't use shell commands.
    """

    # The 9 semantic features (matching llm_judge_prompt.py)
    DEFAULT_FEATURE_COLS = [
        "fix_in_description",
        "problem_clarity",
        "error_message_provided",
        "reproduction_steps",
        "fix_locality",
        "domain_knowledge_required",
        "fix_complexity",
        "logical_reasoning_required",
        "atypicality",
    ]

    def __init__(
        self,
        features_path: Path,
        ridge_alpha: Optional[float] = None,
        ridge_alphas: Optional[List[float]] = None,
        max_features: Optional[int] = None,  # None = use all 9
        feature_cols: Optional[List[str]] = None,
    ):
        """Initialize LLM Judge predictor.

        Args:
            features_path: Path to CSV file with LLM judge features
            ridge_alpha: Fixed Ridge alpha (if provided, skips CV)
            ridge_alphas: List of alphas for RidgeCV (default: use CV with standard alphas)
            max_features: Maximum number of features to select (None = no limit)
            feature_cols: List of feature columns to use (None = use defaults)
        """
        self.features_path = Path(features_path)
        self.ridge_alpha = ridge_alpha
        self.ridge_alphas = ridge_alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.max_features = max_features
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS

        self._model: Optional[Ridge] = None
        self._best_alpha: Optional[float] = None
        self._scaler: Optional[StandardScaler] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        self._feature_coefficients: Optional[Dict[str, float]] = None
        self._selected_mask: Optional[np.ndarray] = None

        # Load features immediately
        self._load_features()

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return "LLM Judge"

    def _load_features(self) -> None:
        """Load features from CSV file."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        self._features_df = pd.read_csv(self.features_path)

        # Set index to task/instance ID column
        if "_instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("_instance_id")
        elif "instance_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("instance_id")
        elif "task_id" in self._features_df.columns:
            self._features_df = self._features_df.set_index("task_id")

        # Filter to available feature columns
        available_cols = [c for c in self.feature_cols if c in self._features_df.columns]
        if len(available_cols) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_cols)
            print(f"Warning: Missing feature columns: {missing}")

        self.feature_cols = available_cols

    def _get_feature_matrix(self, task_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix for given task IDs.

        Returns:
            (X, available_task_ids) where X is (n_tasks, n_features)
        """
        if self._features_df is None:
            raise RuntimeError("Features not loaded")

        # Filter to available tasks
        available_tasks = [t for t in task_ids if t in self._features_df.index]

        if not available_tasks:
            return np.array([]).reshape(0, len(self.feature_cols)), []

        # Extract feature matrix
        X = self._features_df.loc[available_tasks, self.feature_cols].values.astype(np.float32)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        return X, available_tasks

    def fit(self, task_ids: List[str], ground_truth_b: np.ndarray) -> None:
        """Fit Lasso for feature selection, then Ridge for final model.

        Args:
            task_ids: List of training task identifiers
            ground_truth_b: Array of ground truth difficulty values
        """
        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if len(available_tasks) < len(task_ids):
            missing = len(task_ids) - len(available_tasks)
            print(f"Warning: {missing} tasks missing from LLM Judge features")

        if len(available_tasks) == 0:
            raise ValueError("No tasks available for training")

        # Get corresponding ground truth values
        y = np.array([ground_truth_b[task_ids.index(t)] for t in available_tasks])

        # Normalize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Lasso for feature selection
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y)

        # Get non-zero coefficients
        coef_abs = np.abs(lasso.coef_)
        nonzero_mask = coef_abs > 1e-6

        # Select features
        if self.max_features and np.sum(nonzero_mask) > self.max_features:
            # Take top k by absolute coefficient
            top_k_idx = np.argsort(coef_abs)[-self.max_features:]
            selected_mask = np.zeros(len(self.feature_cols), dtype=bool)
            selected_mask[top_k_idx] = True
        elif np.sum(nonzero_mask) == 0:
            # No features selected, use top k by correlation
            print("Warning: Lasso selected 0 features, falling back to top-k correlation")
            k = self.max_features or 5
            selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
            selector.fit(X_scaled, y)
            selected_mask = selector.get_support()
        else:
            selected_mask = nonzero_mask

        self._selected_features = [
            self.feature_cols[i] for i in range(len(self.feature_cols)) if selected_mask[i]
        ]

        # Fit Ridge on selected features (with CV or fixed alpha)
        X_selected = X_scaled[:, selected_mask]
        if self.ridge_alpha is not None:
            self._model = Ridge(alpha=self.ridge_alpha)
            self._best_alpha = self.ridge_alpha
        else:
            self._model = RidgeCV(alphas=self.ridge_alphas, cv=5)
        self._model.fit(X_selected, y)

        # Store best alpha if using CV
        if hasattr(self._model, "alpha_"):
            self._best_alpha = float(self._model.alpha_)

        # Store coefficients for reporting
        self._feature_coefficients = dict(
            zip(self._selected_features, self._model.coef_.tolist())
        )
        self._selected_mask = selected_mask

    def predict(self, task_ids: List[str]) -> Dict[str, float]:
        """Predict difficulty for tasks.

        Args:
            task_ids: List of task identifiers

        Returns:
            Dict mapping task_id to predicted difficulty
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get feature matrix
        X, available_tasks = self._get_feature_matrix(task_ids)

        if not available_tasks:
            return {}

        # Transform and select features
        X_scaled = self._scaler.transform(X)
        X_selected = X_scaled[:, self._selected_mask]

        # Predict
        preds = self._model.predict(X_selected)

        return dict(zip(available_tasks, preds.tolist()))

    @property
    def selected_features(self) -> Optional[List[str]]:
        """Return names of selected features."""
        return self._selected_features

    @property
    def feature_coefficients(self) -> Optional[Dict[str, float]]:
        """Return coefficients of selected features."""
        return self._feature_coefficients

    @property
    def n_features(self) -> int:
        """Return number of available features."""
        return len(self.feature_cols)

    @property
    def n_tasks(self) -> int:
        """Return number of tasks with features."""
        return len(self._features_df) if self._features_df is not None else 0

    @property
    def best_alpha(self) -> Optional[float]:
        """Return the best alpha (from CV or fixed)."""
        return self._best_alpha

    def print_selected_features(self) -> None:
        """Print selected features and their coefficients."""
        if self._feature_coefficients is None:
            print("Model not fitted yet")
            return

        print(f"\nSelected features (lasso_cv, n={len(self._selected_features)}):")
        sorted_features = sorted(
            self._feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for name, coef in sorted_features:
            sign = "+" if coef >= 0 else ""
            print(f"  {name:30s}: {sign}{coef:.4f}")
