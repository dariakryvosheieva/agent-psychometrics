#!/usr/bin/env python3
"""Minimal repro to debug LoRA gradient flow in SAD-IRT."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoConfig

def test_lora_gradients():
    """Test if LoRA gradients flow through a minimal setup."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load small model
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    encoder = get_peft_model(encoder, peft_config)
    encoder = encoder.to(device)

    # Simple head
    psi_head = nn.Linear(config.hidden_size, 1).to(device)

    # Print trainable params
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Create dummy input (small to avoid OOM)
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    # Forward pass
    print("\n=== Forward pass ===")
    outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    hidden = outputs.last_hidden_state  # (batch, seq, hidden)
    print(f"Hidden shape: {hidden.shape}, dtype: {hidden.dtype}")

    # Get last token
    last_hidden = hidden[:, -1, :]  # (batch, hidden)
    print(f"Last hidden shape: {last_hidden.shape}")

    # Predict psi
    psi = psi_head(last_hidden.float())  # (batch, 1)
    print(f"Psi shape: {psi.shape}, values: {psi.squeeze().tolist()}")

    # Simple loss: just mean of psi (so gradient should flow)
    loss = psi.mean()
    print(f"Loss: {loss.item()}")

    # Backward
    print("\n=== Backward pass ===")
    loss.backward()

    # Check gradients
    print("\n=== Gradient check ===")

    # psi_head
    print(f"psi_head.weight.grad norm: {psi_head.weight.grad.norm().item():.8f}")
    print(f"psi_head.bias.grad norm: {psi_head.bias.grad.norm().item():.8f}")

    # LoRA params
    lora_grads = []
    lora_no_grad = []
    for name, param in encoder.named_parameters():
        if "lora" in name and param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                lora_grads.append((name, grad_norm))
            else:
                lora_no_grad.append(name)

    print(f"\nLoRA params with gradients: {len(lora_grads)}")
    print(f"LoRA params without gradients: {len(lora_no_grad)}")

    if lora_grads:
        # Sort by grad norm
        lora_grads.sort(key=lambda x: -x[1])
        print("\nTop 5 LoRA gradients:")
        for name, norm in lora_grads[:5]:
            print(f"  {name}: {norm:.8f}")

        total_lora_grad = sum(g for _, g in lora_grads)
        print(f"\nTotal LoRA grad norm: {total_lora_grad:.8f}")

    if lora_no_grad:
        print("\nLoRA params with None gradients:")
        for name in lora_no_grad[:5]:
            print(f"  {name}")

    # Check if any gradient is non-zero
    any_lora_grad = any(g > 0 for _, g in lora_grads)
    print(f"\n{'✅ LoRA gradients ARE flowing' if any_lora_grad else '❌ LoRA gradients are NOT flowing'}")

    return any_lora_grad


def test_with_centering():
    """Test with centering normalization (like SAD-IRT uses)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Testing with centering normalization (psi - psi.mean())")
    print(f"{'='*60}")

    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    encoder = get_peft_model(encoder, peft_config)
    encoder = encoder.to(device)

    psi_head = nn.Linear(config.hidden_size, 1).to(device)

    # Dummy IRT params
    theta = nn.Embedding(10, 1).to(device)  # 10 agents
    beta = nn.Embedding(100, 1).to(device)  # 100 tasks

    # Create dummy input
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    agent_idx = torch.randint(0, 10, (batch_size,), device=device)
    task_idx = torch.randint(0, 100, (batch_size,), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device).float()

    # Forward
    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state[:, -1, :]

    psi_raw = psi_head(hidden.float())

    # Centering: subtract mean
    psi = psi_raw - psi_raw.mean()

    # IRT formula
    theta_val = theta(agent_idx)
    beta_val = beta(task_idx)
    logits = theta_val - (beta_val + psi)

    # BCE loss
    loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), labels)
    print(f"Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Check LoRA gradients
    lora_grad_norm = 0.0
    lora_count = 0
    for name, param in encoder.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
            lora_count += 1

    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"LoRA total grad norm: {lora_grad_norm:.8f} ({lora_count} params)")
    print(f"psi_head.weight grad norm: {psi_head.weight.grad.norm().item():.8f}")
    print(f"theta.weight grad norm: {theta.weight.grad.norm().item():.8f}")
    print(f"beta.weight grad norm: {beta.weight.grad.norm().item():.8f}")

    print(f"\n{'✅ LoRA gradients ARE flowing' if lora_grad_norm > 1e-10 else '❌ LoRA gradients are NOT flowing'}")

    return lora_grad_norm > 1e-10


def test_with_grad_accumulation():
    """Test with gradient accumulation (like the real training)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Testing with gradient accumulation (2 steps)")
    print(f"{'='*60}")

    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    encoder = get_peft_model(encoder, peft_config)
    encoder = encoder.to(device)

    psi_head = nn.Linear(config.hidden_size, 1).to(device)
    theta = nn.Embedding(10, 1).to(device)
    beta = nn.Embedding(100, 1).to(device)

    # Optimizer with different param groups (like SAD-IRT trainer)
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-4},
        {"params": psi_head.parameters(), "lr": 1e-4},
        {"params": theta.parameters(), "lr": 1e-2},
        {"params": beta.parameters(), "lr": 1e-2},
    ])

    gradient_accumulation_steps = 2

    # Zero gradients
    optimizer.zero_grad()

    total_loss = 0
    for accum_step in range(gradient_accumulation_steps):
        # Create dummy input
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        agent_idx = torch.randint(0, 10, (batch_size,), device=device)
        task_idx = torch.randint(0, 100, (batch_size,), device=device)
        labels = torch.randint(0, 2, (batch_size,), device=device).float()

        # Forward
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]
        psi_raw = psi_head(hidden.float())
        psi = psi_raw - psi_raw.mean()  # Centering

        theta_val = theta(agent_idx)
        beta_val = beta(task_idx)
        logits = theta_val - (beta_val + psi)

        loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), labels)
        loss = loss / gradient_accumulation_steps  # Scale for accumulation

        loss.backward()
        total_loss += loss.item()

    print(f"Total loss: {total_loss:.6f}")

    # Check gradients BEFORE optimizer step
    lora_grad_norm = 0.0
    lora_count = 0
    for name, param in encoder.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
            lora_count += 1

    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"LoRA total grad norm (pre-step): {lora_grad_norm:.8f} ({lora_count} params)")
    print(f"psi_head.weight grad norm: {psi_head.weight.grad.norm().item():.8f}")
    print(f"theta.weight grad norm: {theta.weight.grad.norm().item():.8f}")

    # Now do optimizer step
    optimizer.step()

    print(f"\n{'✅ LoRA gradients ARE flowing' if lora_grad_norm > 1e-10 else '❌ LoRA gradients are NOT flowing'}")

    return lora_grad_norm > 1e-10


def test_with_actual_sadirt_model():
    """Test using the actual SADIRT model class."""

    import sys
    sys.path.insert(0, '.')

    from experiment_sad_irt.model import SADIRT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Testing with actual SADIRT model class")
    print(f"{'='*60}")

    # Create model
    model = SADIRT(
        num_agents=10,
        num_tasks=100,
        model_name="Qwen/Qwen3-0.6B",
        psi_normalization="center",
    ).to(device)

    # Create dummy input
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    agent_idx = torch.randint(0, 10, (batch_size,), device=device)
    task_idx = torch.randint(0, 100, (batch_size,), device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device).float()

    # Forward
    logits = model(agent_idx, task_idx, input_ids, attention_mask)

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    # Check LoRA gradients
    lora_grad_norm = 0.0
    lora_count = 0
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
            lora_count += 1

    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"LoRA total grad norm: {lora_grad_norm:.8f} ({lora_count} params)")
    print(f"psi_head.weight grad norm: {model.psi_head.weight.grad.norm().item():.8f}")
    print(f"theta.weight grad norm: {model.theta.weight.grad.norm().item():.8f}")
    print(f"beta.weight grad norm: {model.beta.weight.grad.norm().item():.8f}")

    print(f"\n{'✅ LoRA gradients ARE flowing' if lora_grad_norm > 1e-10 else '❌ LoRA gradients are NOT flowing'}")

    return lora_grad_norm > 1e-10


if __name__ == "__main__":
    print("="*60)
    print("Test 1: Simple forward-backward through LoRA encoder")
    print("="*60)
    test1_pass = test_lora_gradients()

    test2_pass = test_with_centering()

    test3_pass = test_with_grad_accumulation()

    test4_pass = test_with_actual_sadirt_model()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (simple): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (centering): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (grad accumulation): {'✅ PASS' if test3_pass else '❌ FAIL'}")
    print(f"Test 4 (actual SADIRT model): {'✅ PASS' if test4_pass else '❌ FAIL'}")
