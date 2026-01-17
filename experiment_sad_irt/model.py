"""SAD-IRT model with Qwen3 trajectory encoder."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class SADIRT(nn.Module):
    """State-Aware Deep Item Response Theory model.

    P(y_ij = 1 | θ_j, β_i, ψ_ij) = σ(θ_j - (β_i + ψ_ij))

    Where:
    - θ_j: Agent ability (learnable embedding)
    - β_i: Task difficulty (learnable embedding)
    - ψ_ij: Trajectory-based interaction term (predicted by encoder)

    ψ_ij is constrained to be zero-mean via BatchNorm for identifiability.
    """

    def __init__(
        self,
        num_agents: int,
        num_tasks: int,
        model_name: str = "Qwen/Qwen3-0.6B",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = None,
        use_gradient_checkpointing: bool = True,
        psi_normalization: str = "batchnorm",
    ):
        """Initialize SAD-IRT model.

        Args:
            num_agents: Number of unique agents
            num_tasks: Number of unique tasks
            model_name: HuggingFace model name for trajectory encoder
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: LoRA dropout
            lora_target_modules: Modules to apply LoRA to
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            psi_normalization: How to normalize ψ values:
                - "batchnorm": Full BatchNorm (zero-mean + unit variance) - default
                - "center": Just subtract mean (zero-mean only) - for frozen IRT
                - "none": No normalization - raw ψ values
        """
        super().__init__()

        self.num_agents = num_agents
        self.num_tasks = num_tasks

        # IRT parameters (learnable embeddings)
        self.theta = nn.Embedding(num_agents, 1)  # Agent abilities
        self.beta = nn.Embedding(num_tasks, 1)  # Task difficulties

        # Initialize with small values
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0.0, std=0.1)

        # Load base model
        logger.info(f"Loading base model: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, peft_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"LoRA applied: {trainable_params:,} trainable params "
            f"({trainable_params / total_params * 100:.2f}% of {total_params:,} total)"
        )

        # Get hidden size from config
        encoder_dim = config.hidden_size

        # Linear projection for ψ prediction (single layer)
        # Simpler than MLP and sufficient for predicting a scalar correction term
        # Uses PyTorch default Kaiming uniform initialization
        self.psi_head = nn.Linear(encoder_dim, 1)

        # Normalization for ψ to enforce zero-mean constraint
        self.psi_normalization = psi_normalization
        if psi_normalization == "batchnorm":
            # Full BatchNorm: zero-mean + unit variance
            self.psi_bn = nn.BatchNorm1d(1, affine=False, momentum=0.1)
            logger.info("Using BatchNorm for ψ (zero-mean + unit variance)")
        elif psi_normalization == "center":
            # Just centering: zero-mean only (no variance scaling)
            self.psi_bn = None
            logger.info("Using centering for ψ (zero-mean only)")
        elif psi_normalization == "none":
            # No normalization
            self.psi_bn = None
            logger.info("No normalization for ψ (raw values)")
        else:
            raise ValueError(f"Unknown psi_normalization: {psi_normalization}")

    def forward(
        self,
        agent_idx: torch.Tensor,
        task_idx: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            agent_idx: Agent indices (batch_size,)
            task_idx: Task indices (batch_size,)
            input_ids: Tokenized input (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            logits: Unnormalized log-odds of success (batch_size,)
        """
        # Get IRT parameters
        theta = self.theta(agent_idx)  # (batch_size, 1)
        beta = self.beta(task_idx)  # (batch_size, 1)

        # Encode trajectory
        # Use last token representation (common for causal LMs)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Get representation from last non-padding token
        # Find the position of the last non-padding token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch_size,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_hidden = hidden_states[batch_indices, seq_lengths]  # (batch_size, hidden_dim)

        # Predict ψ
        psi_raw = self.psi_head(last_token_hidden.float())  # (batch_size, 1)

        # Apply normalization for zero-mean constraint
        if self.psi_normalization == "batchnorm" and self.psi_bn is not None:
            # Full BatchNorm: zero-mean + unit variance
            if self.training and psi_raw.size(0) > 1:
                psi = self.psi_bn(psi_raw)
            else:
                # During eval or batch_size=1, use running stats
                self.psi_bn.eval()
                psi = self.psi_bn(psi_raw)
                if self.training:
                    self.psi_bn.train()
        elif self.psi_normalization == "center":
            # Just subtract mean (zero-mean only, no variance scaling)
            psi = psi_raw - psi_raw.mean()
        else:
            # No normalization - use raw ψ values
            psi = psi_raw

        # IRT formula: logit = θ - (β + ψ)
        logits = theta - (beta + psi)

        return logits.squeeze(-1)

    def get_abilities(self) -> torch.Tensor:
        """Get agent ability parameters."""
        return self.theta.weight.detach().squeeze(-1)

    def get_difficulties(self) -> torch.Tensor:
        """Get task difficulty parameters."""
        return self.beta.weight.detach().squeeze(-1)

    def get_psi_stats(self) -> dict:
        """Get statistics about ψ normalization."""
        stats = {"normalization": self.psi_normalization}
        if self.psi_bn is not None:
            stats["running_mean"] = self.psi_bn.running_mean.item() if self.psi_bn.running_mean is not None else None
            stats["running_var"] = self.psi_bn.running_var.item() if self.psi_bn.running_var is not None else None
        return stats

    def initialize_from_pretrained_irt(
        self,
        agent_ids: list,
        task_ids: list,
        abilities_df,
        items_df,
    ):
        """Initialize θ and β from pre-trained IRT parameters.

        This gives the model a much better starting point than random initialization,
        following the py_irt 'difficulty_from_accuracy' initializer approach.

        Args:
            agent_ids: List of agent IDs in order matching model indices
            task_ids: List of task IDs in order matching model indices
            abilities_df: DataFrame with 'theta' column, indexed by agent_id
            items_df: DataFrame with 'b' column, indexed by task_id
        """
        with torch.no_grad():
            initialized_agents = 0
            for i, agent_id in enumerate(agent_ids):
                if agent_id in abilities_df.index:
                    self.theta.weight[i] = abilities_df.loc[agent_id, "theta"]
                    initialized_agents += 1

            initialized_tasks = 0
            for i, task_id in enumerate(task_ids):
                if task_id in items_df.index:
                    self.beta.weight[i] = items_df.loc[task_id, "b"]
                    initialized_tasks += 1

        logger.info(
            f"Initialized θ for {initialized_agents}/{len(agent_ids)} agents, "
            f"β for {initialized_tasks}/{len(task_ids)} tasks from pre-trained IRT"
        )

    def initialize_from_accuracy(
        self,
        agent_ids: list,
        task_ids: list,
        theta_init: dict,
        beta_init: dict,
    ):
        """Initialize θ and β from accuracy-based estimates.

        This follows the py_irt 'difficulty_from_accuracy' approach:
        β = logit(1 - task_accuracy)
        θ = logit(agent_accuracy)

        Args:
            agent_ids: List of agent IDs in order matching model indices
            task_ids: List of task IDs in order matching model indices
            theta_init: Dict mapping agent_id -> initial θ value
            beta_init: Dict mapping task_id -> initial β value
        """
        with torch.no_grad():
            initialized_agents = 0
            for i, agent_id in enumerate(agent_ids):
                if agent_id in theta_init:
                    self.theta.weight[i] = theta_init[agent_id]
                    initialized_agents += 1

            initialized_tasks = 0
            for i, task_id in enumerate(task_ids):
                if task_id in beta_init:
                    self.beta.weight[i] = beta_init[task_id]
                    initialized_tasks += 1

        logger.info(
            f"Initialized θ for {initialized_agents}/{len(agent_ids)} agents, "
            f"β for {initialized_tasks}/{len(task_ids)} tasks from accuracy"
        )


class StandardIRT(nn.Module):
    """Standard 1PL IRT model without trajectory features (baseline).

    P(y_ij = 1 | θ_j, β_i) = σ(θ_j - β_i)
    """

    def __init__(self, num_agents: int, num_tasks: int):
        super().__init__()

        self.num_agents = num_agents
        self.num_tasks = num_tasks

        # IRT parameters
        self.theta = nn.Embedding(num_agents, 1)
        self.beta = nn.Embedding(num_tasks, 1)

        # Initialize
        nn.init.normal_(self.theta.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.beta.weight, mean=0.0, std=0.1)

    def forward(
        self,
        agent_idx: torch.Tensor,
        task_idx: torch.Tensor,
        **kwargs,  # Ignore other inputs
    ) -> torch.Tensor:
        """Forward pass."""
        theta = self.theta(agent_idx)
        beta = self.beta(task_idx)
        logits = theta - beta
        return logits.squeeze(-1)

    def get_abilities(self) -> torch.Tensor:
        return self.theta.weight.detach().squeeze(-1)

    def get_difficulties(self) -> torch.Tensor:
        return self.beta.weight.detach().squeeze(-1)

    def initialize_from_pretrained_irt(
        self,
        agent_ids: list,
        task_ids: list,
        abilities_df,  # pandas DataFrame with agent abilities
        items_df,  # pandas DataFrame with item difficulties
    ):
        """Initialize θ and β from pre-trained IRT parameters.

        Args:
            agent_ids: List of agent IDs in order matching embedding indices
            task_ids: List of task IDs in order matching embedding indices
            abilities_df: DataFrame with agent abilities (column 'ability' or 'theta')
            items_df: DataFrame with item difficulties (column 'b' or 'difficulty')
        """
        # Initialize θ from abilities
        ability_col = 'ability' if 'ability' in abilities_df.columns else 'theta'
        initialized_agents = 0
        for i, agent_id in enumerate(agent_ids):
            if agent_id in abilities_df.index:
                self.theta.weight.data[i, 0] = abilities_df.loc[agent_id, ability_col]
                initialized_agents += 1

        # Initialize β from difficulties
        diff_col = 'b' if 'b' in items_df.columns else 'difficulty'
        initialized_tasks = 0
        for i, task_id in enumerate(task_ids):
            if task_id in items_df.index:
                self.beta.weight.data[i, 0] = items_df.loc[task_id, diff_col]
                initialized_tasks += 1

        logger.info(
            f"Initialized θ for {initialized_agents}/{len(agent_ids)} agents, "
            f"β for {initialized_tasks}/{len(task_ids)} tasks from pre-trained IRT"
        )
