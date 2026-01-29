"""MLP predictor that directly predicts P(success) from (agent, task) pairs.

Architecture (IRTStyleMLP):
    Mirrors the IRT formula: P = sigmoid(θ_agent - β_task)
    - Agent pathway: agent_index → Embedding → θ (scalar ability per agent)
    - Feature pathway: task_features → Linear → β (scalar difficulty)
    - Output: sigmoid(θ - β)

    This is structurally identical to IRT, just learned end-to-end with BCE loss
    instead of maximum likelihood on the response matrix.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiment_ab_shared.dataset import BinomialExperimentData, ExperimentData
from experiment_ab_shared.feature_source import TaskFeatureSource


def build_input_vector(
    agent_idx: int,
    n_agents: int,
    task_features: np.ndarray,
) -> np.ndarray:
    """Build input vector for MLP: [agent_one_hot | task_features].

    Args:
        agent_idx: Index of the agent in the agent list.
        n_agents: Total number of agents.
        task_features: Scaled task feature vector.

    Returns:
        Concatenated input vector of shape (n_agents + feature_dim,).
    """
    agent_one_hot = np.zeros(n_agents, dtype=np.float32)
    agent_one_hot[agent_idx] = 1.0
    return np.concatenate([agent_one_hot, task_features])


class IRTStyleMLP(nn.Module):
    """IRT-style architecture that explicitly learns θ_agent - β_task.

    Mirrors the IRT formula: P = sigmoid(θ_agent - β_task)
    - Agent pathway: agent_one_hot → θ (learned ability per agent)
    - Feature pathway: task_features → β (learned difficulty from features)
    - Output: sigmoid(θ - β)

    This is structurally identical to IRT, just learned end-to-end with BCE loss.
    """

    def __init__(self, n_agents: int, feature_dim: int, dropout: float = 0.0):
        super().__init__()
        # Agent abilities: one θ per agent (like IRT)
        self.agent_abilities = nn.Embedding(n_agents, 1)

        # Difficulty from features: features → scalar β
        self.difficulty_layer = nn.Linear(feature_dim, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, agent_indices: torch.Tensor, task_features: torch.Tensor
    ) -> torch.Tensor:
        # Get agent abilities: (batch,) → (batch, 1)
        theta = self.agent_abilities(agent_indices)  # (batch, 1)

        # Get task difficulties: (batch, feature_dim) → (batch, 1)
        beta = self.difficulty_layer(task_features)  # (batch, 1)
        beta = self.dropout(beta)

        # IRT formula: P = sigmoid(θ - β)
        logit = theta - beta  # (batch, 1)
        prob = self.sigmoid(logit)

        return prob.squeeze(-1)  # (batch,)


class MLPPredictor:
    """IRT-style predictor that learns P = sigmoid(θ_agent - β_task) end-to-end.

    This predictor mirrors the IRT formula exactly:
    - θ: learned ability per agent (via embedding)
    - β: learned difficulty from task features (via linear layer)
    - P = sigmoid(θ - β)

    The key difference from standard IRT is that β is predicted from features
    rather than learned as free parameters per task.

    Training uses binary cross-entropy loss with Adam optimizer.
    """

    def __init__(
        self,
        source: TaskFeatureSource,
        hidden_size: int = 64,  # Unused, kept for API compatibility
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        feature_weight_decay: Optional[float] = None,
        n_epochs: int = 200,
        verbose: bool = False,
        freeze_abilities: bool = False,
    ):
        """Initialize IRT-style predictor.

        Args:
            source: TaskFeatureSource providing features for tasks.
            hidden_size: Unused (kept for API compatibility).
            dropout: Dropout probability on difficulty prediction (0.0 = no dropout).
            learning_rate: Learning rate for Adam optimizer.
            weight_decay: L2 regularization on agent abilities (Adam weight_decay).
            feature_weight_decay: L2 regularization on feature weights. If None, uses weight_decay.
                Use higher values (e.g., 1.0-10.0) for high-dim features like embeddings.
            n_epochs: Number of training epochs.
            verbose: Print training progress.
            freeze_abilities: If True, initialize agent abilities from IRT and freeze them.
                This isolates the difficulty layer learning and prevents agent memorization.
        """
        self.source = source
        self.hidden_size = hidden_size  # Unused but kept for compatibility
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feature_weight_decay = feature_weight_decay if feature_weight_decay is not None else weight_decay
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.freeze_abilities = freeze_abilities

        # Model state (set after fit())
        self._model: Optional[IRTStyleMLP] = None
        self._scaler: Optional[StandardScaler] = None
        self._agent_to_idx: Optional[Dict[str, int]] = None
        self._n_agents: int = 0
        self._feature_dim: int = 0
        self._is_fitted: bool = False

        # Training diagnostics
        self._training_losses: List[float] = []
        self._train_auc: Optional[float] = None  # AUC on training data

        # Prediction cache
        self._task_feature_cache: Dict[str, np.ndarray] = {}

    def fit(self, data: ExperimentData, train_task_ids: List[str]) -> None:
        """Fit the IRT-style model on training data.

        Args:
            data: ExperimentData containing responses and agent information.
            train_task_ids: List of task IDs to train on.
        """
        # Clear caches
        self._task_feature_cache = {}
        self._training_losses = []

        # Build agent-to-index mapping
        all_agents = data.get_all_agents()
        self._agent_to_idx = {agent: i for i, agent in enumerate(all_agents)}
        self._n_agents = len(all_agents)

        # Get task features and fit scaler
        task_features = self.source.get_features(train_task_ids)
        self._scaler = StandardScaler()
        task_features_scaled = self._scaler.fit_transform(task_features)
        self._feature_dim = task_features_scaled.shape[1]

        # Build task_id -> scaled features mapping for train tasks
        task_to_features = {
            task_id: task_features_scaled[i]
            for i, task_id in enumerate(train_task_ids)
        }

        # Build training data: (agent_idx, task_features, response)
        agent_indices_list: List[int] = []
        features_list: List[np.ndarray] = []
        y_list: List[float] = []

        is_binomial = isinstance(data, BinomialExperimentData)

        for task_id in train_task_ids:
            task_feat = task_to_features[task_id]

            for agent_id in all_agents:
                if agent_id not in data.responses:
                    continue
                if task_id not in data.responses[agent_id]:
                    continue

                agent_idx = self._agent_to_idx[agent_id]
                response = data.responses[agent_id][task_id]

                if is_binomial:
                    # Expand binomial to individual observations
                    k = response["successes"]
                    n = response["trials"]
                    # Add k success observations
                    for _ in range(k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(1.0)
                    # Add (n-k) failure observations
                    for _ in range(n - k):
                        agent_indices_list.append(agent_idx)
                        features_list.append(task_feat)
                        y_list.append(0.0)
                else:
                    # Binary response
                    agent_indices_list.append(agent_idx)
                    features_list.append(task_feat)
                    y_list.append(float(response))

        if len(y_list) == 0:
            raise ValueError("No training examples found")

        agent_indices = np.array(agent_indices_list, dtype=np.int64)
        features = np.array(features_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if self.verbose:
            print(f"   Training IRT-style MLP: {len(y)} samples")
            print(f"   Agents: {self._n_agents}, Feature dim: {self._feature_dim}")
            print(f"   freeze_abilities={self.freeze_abilities}, weight_decay={self.weight_decay}, feature_weight_decay={self.feature_weight_decay}")

        # Create model
        self._model = IRTStyleMLP(self._n_agents, self._feature_dim, dropout=self.dropout)

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)

        if self.verbose and device.type == "cuda":
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

        # Initialize agent abilities from IRT if requested
        if self.freeze_abilities:
            with torch.no_grad():
                for agent_id, idx in self._agent_to_idx.items():
                    if agent_id in data.train_abilities.index:
                        ability = data.train_abilities.loc[agent_id, "ability"]
                        self._model.agent_abilities.weight.data[idx, 0] = float(ability)
                    else:
                        # Default to 0 for agents without IRT abilities
                        self._model.agent_abilities.weight.data[idx, 0] = 0.0
            # Freeze agent abilities
            self._model.agent_abilities.requires_grad_(False)
            if self.verbose:
                print(f"   Initialized and froze agent abilities from IRT")

        # Convert to tensors
        agent_tensor = torch.tensor(agent_indices, dtype=torch.long, device=device)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        # Optimizer with per-parameter group regularization
        if self.freeze_abilities:
            # Only optimize difficulty layer when abilities are frozen
            optimizer = torch.optim.Adam(
                self._model.difficulty_layer.parameters(),
                lr=self.learning_rate,
                weight_decay=self.feature_weight_decay,
            )
        else:
            # Separate weight_decay for agent abilities vs feature weights
            optimizer = torch.optim.Adam([
                {'params': self._model.agent_abilities.parameters(), 'weight_decay': self.weight_decay},
                {'params': self._model.difficulty_layer.parameters(), 'weight_decay': self.feature_weight_decay},
            ], lr=self.learning_rate)
        criterion = nn.BCELoss()

        # Training loop (full-batch)
        self._model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Forward pass
            y_pred = self._model(agent_tensor, features_tensor)
            loss = criterion(y_pred, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss per iteration
            loss_val = loss.item()
            self._training_losses.append(loss_val)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"      Epoch {epoch + 1}/{self.n_epochs}: Loss = {loss_val:.6f}")

        self._is_fitted = True

        # Compute train AUC for diagnostics
        self._model.eval()
        with torch.no_grad():
            y_pred_train = self._model(agent_tensor, features_tensor).cpu().numpy()
        if len(np.unique(y)) > 1:  # Need both classes for AUC
            self._train_auc = roc_auc_score(y, y_pred_train)
        else:
            self._train_auc = None

        if self.verbose:
            train_auc_str = f"{self._train_auc:.4f}" if self._train_auc else "N/A"
            print(f"   Final loss: {self._training_losses[-1]:.6f}, Train AUC: {train_auc_str}")

    def predict_probability(
        self, data: ExperimentData, agent_id: str, task_id: str
    ) -> float:
        """Predict success probability for a specific (agent, task) pair.

        Uses IRT formula: P = sigmoid(θ_agent - β_task)

        Args:
            data: ExperimentData (used for test_tasks list).
            agent_id: Agent identifier.
            task_id: Task identifier.

        Returns:
            Predicted probability of success (0 to 1).
        """
        if not self._is_fitted:
            raise RuntimeError("Predictor must be fit before calling predict_probability()")

        # Lazily cache test task features
        if task_id not in self._task_feature_cache:
            self._cache_test_task_features(data.test_tasks)

        if task_id not in self._task_feature_cache:
            raise ValueError(f"No features for task {task_id}")

        if agent_id not in self._agent_to_idx:
            raise ValueError(f"Unknown agent {agent_id}")

        agent_idx = self._agent_to_idx[agent_id]
        task_feat = self._task_feature_cache[task_id]

        # Get device
        device = next(self._model.parameters()).device

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            agent_tensor = torch.tensor([agent_idx], dtype=torch.long, device=device)
            feat_tensor = torch.tensor(task_feat, dtype=torch.float32, device=device).unsqueeze(0)
            prob = self._model(agent_tensor, feat_tensor).item()

        return prob

    def _cache_test_task_features(self, test_tasks: List[str]) -> None:
        """Cache scaled features for test tasks."""
        features = self.source.get_features(test_tasks)
        features_scaled = self._scaler.transform(features)

        for i, task_id in enumerate(test_tasks):
            self._task_feature_cache[task_id] = features_scaled[i]

    def get_training_losses(self) -> List[float]:
        """Return list of loss values per iteration for convergence plots."""
        return self._training_losses.copy()

    def get_train_auc(self) -> Optional[float]:
        """Return AUC computed on training data after fit().

        This is useful for diagnosing overfitting: if train AUC >> test AUC,
        the model is overfitting to the training data.
        """
        return self._train_auc

    @property
    def name(self) -> str:
        """Human-readable predictor name."""
        return f"MLP ({self.source.name})"
