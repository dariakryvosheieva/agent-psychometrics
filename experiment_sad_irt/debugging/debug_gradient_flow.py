#!/usr/bin/env python3
"""Debug gradient flow through centering normalization.

This script investigates why LoRA gradients are zero in SAD-IRT training.
Run this on the cluster with your actual data.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from experiment_sad_irt.model import SADIRT
from experiment_sad_irt.dataset import TrajectoryIRTDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def analyze_gradient_flow():
    """Analyze gradient flow through the centering operation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset with same params as cluster training
    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=1024,  # Same as cluster
    )

    # Create model
    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",  # Same as cluster
    ).to(device)

    # Use same batch size as cluster (32)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"\nBatch size: {batch['input_ids'].shape[0]}")
    print(f"Sequence length: {batch['input_ids'].shape[1]}")

    # Forward pass - step by step to analyze
    print("\n" + "="*60)
    print("STEP-BY-STEP GRADIENT ANALYSIS")
    print("="*60)

    # Step 1: Get IRT params
    theta = model.theta(batch["agent_idx"])
    beta = model.beta(batch["task_idx"])
    print(f"\nθ stats: mean={theta.mean().item():.4f}, std={theta.std().item():.4f}")
    print(f"β stats: mean={beta.mean().item():.4f}, std={beta.std().item():.4f}")

    # Step 2: Encode trajectory
    outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    hidden_states = outputs.last_hidden_state

    # Step 3: Get last token hidden states (using model's logic)
    batch_size = hidden_states.size(0)
    seq_len = hidden_states.size(1)
    positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = batch["attention_mask"]
    masked_positions = positions * attention_mask.long() - (1 - attention_mask.long())
    last_token_positions = masked_positions.argmax(dim=1)

    batch_indices = torch.arange(batch_size, device=hidden_states.device)
    last_token_hidden = hidden_states[batch_indices, last_token_positions]

    print(f"\nlast_token_hidden dtype: {last_token_hidden.dtype}")
    print(f"last_token_hidden stats: mean={last_token_hidden.float().mean().item():.4f}, std={last_token_hidden.float().std().item():.4f}")

    # Step 4: Predict psi (raw)
    psi_raw = model.psi_head(last_token_hidden.float())
    print(f"\npsi_raw stats: mean={psi_raw.mean().item():.6f}, std={psi_raw.std().item():.6f}, min={psi_raw.min().item():.6f}, max={psi_raw.max().item():.6f}")

    # Step 5: Apply centering
    psi_mean = psi_raw.mean()
    psi = psi_raw - psi_mean
    print(f"psi (centered) stats: mean={psi.mean().item():.6f}, std={psi.std().item():.6f}")

    # Step 6: IRT formula
    logits = theta - (beta + psi)
    print(f"\nlogits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

    # Step 7: Compute loss
    loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), batch["response"])
    print(f"Loss: {loss.item():.6f}")

    # Now backward and analyze gradients
    print("\n" + "="*60)
    print("GRADIENT ANALYSIS")
    print("="*60)

    loss.backward()

    # Key insight: analyze gradient of psi
    # The gradient of loss w.r.t. psi is stored in psi.grad
    # But we need to trace through the centering manually

    # Get gradient of loss w.r.t. logits (stored after backward)
    # We need to compute this manually
    probs = torch.sigmoid(logits.squeeze())
    grad_logits = probs - batch["response"]  # gradient of BCE loss

    print(f"\nGradient of loss w.r.t. logits:")
    print(f"  grad_logits stats: mean={grad_logits.mean().item():.6f}, std={grad_logits.std().item():.6f}")
    print(f"  grad_logits sum: {grad_logits.sum().item():.6f}")

    # Gradient to psi (before centering transforms it)
    # d(loss)/d(psi) = -d(loss)/d(logits) = -(probs - y)
    grad_psi = -grad_logits
    print(f"\nGradient to psi (direct, before centering inverse):")
    print(f"  grad_psi stats: mean={grad_psi.mean().item():.6f}, std={grad_psi.std().item():.6f}")
    print(f"  grad_psi sum: {grad_psi.sum().item():.6f}")

    # The gradient through centering: psi = psi_raw - psi_raw.mean()
    # d(loss)/d(psi_raw[i]) = sum_j d(loss)/d(psi[j]) * d(psi[j])/d(psi_raw[i])
    # d(psi[j])/d(psi_raw[i]) = delta(i,j) - 1/N
    # So d(loss)/d(psi_raw[i]) = d(loss)/d(psi[i]) - (1/N) * sum_j d(loss)/d(psi[j])
    #                         = d(loss)/d(psi[i]) - (1/N) * grad_psi.sum()

    grad_psi_raw = grad_psi - grad_psi.mean()
    print(f"\nGradient to psi_raw (after centering backprop):")
    print(f"  grad_psi_raw stats: mean={grad_psi_raw.mean().item():.8f}, std={grad_psi_raw.std().item():.8f}")
    print(f"  grad_psi_raw sum: {grad_psi_raw.sum().item():.8f}")

    # THE KEY INSIGHT: If grad_psi has near-zero variance, grad_psi_raw will be tiny!
    # This happens when probs ≈ y for all samples, which happens when theta-beta dominates

    # Check actual gradients
    print(f"\n" + "="*60)
    print("ACTUAL PARAMETER GRADIENTS")
    print("="*60)

    print(f"\npsi_head.weight grad norm: {model.psi_head.weight.grad.norm().item():.8f}")
    print(f"psi_head.bias grad norm: {model.psi_head.bias.grad.norm().item():.8f}")
    print(f"theta.weight grad norm: {model.theta.weight.grad.norm().item():.8f}")
    print(f"beta.weight grad norm: {model.beta.weight.grad.norm().item():.8f}")

    # LoRA gradients
    lora_grad_norm = 0.0
    lora_count = 0
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
            lora_count += 1
    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"\nLoRA total grad norm: {lora_grad_norm:.8f} ({lora_count} params)")

    # DIAGNOSIS
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if psi_raw.std().item() < 0.01:
        print("⚠️  psi_raw has very low variance - encoder may not be learning meaningful differences")
        print("   This could be due to initialization or the encoder outputting similar values for all inputs")

    if abs(grad_psi.sum().item()) > 0.1 * grad_psi.std().item() * batch_size ** 0.5:
        print("✓  grad_psi has significant sum - centering should not kill gradients")
    else:
        print("⚠️  grad_psi sum is near zero relative to its variance")
        print("   This means centering backprop will attenuate gradients!")
        print(f"   Ratio: |sum|/(std*sqrt(N)) = {abs(grad_psi.sum().item()) / (grad_psi.std().item() * batch_size ** 0.5 + 1e-8):.4f}")

    if lora_grad_norm < 1e-6:
        print("\n❌ LoRA gradients are effectively zero")
        print("   Possible causes:")
        print("   1. Centering kills gradients when grad_psi.sum() ≈ 0")
        print("   2. psi_raw variance is too low")
        print("   3. theta-beta dominate, making psi irrelevant to loss")
    else:
        print("\n✅ LoRA gradients are flowing")

    return {
        "psi_raw_std": psi_raw.std().item(),
        "grad_psi_sum": grad_psi.sum().item(),
        "grad_psi_std": grad_psi.std().item(),
        "lora_grad_norm": lora_grad_norm,
        "psi_head_grad_norm": model.psi_head.weight.grad.norm().item(),
    }


def test_without_centering():
    """Test if gradients flow without centering (as control)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n{'='*60}")
    print("CONTROL TEST: No centering (psi_normalization='none')")
    print("="*60)

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=1024,
    )

    # Create model WITHOUT centering
    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="none",  # No centering!
    ).to(device)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward
    logits = model(
        agent_idx=batch["agent_idx"],
        task_idx=batch["task_idx"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["response"])
    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    # Check gradients
    print(f"\npsi_head.weight grad norm: {model.psi_head.weight.grad.norm().item():.8f}")
    print(f"theta.weight grad norm: {model.theta.weight.grad.norm().item():.8f}")
    print(f"beta.weight grad norm: {model.beta.weight.grad.norm().item():.8f}")

    lora_grad_norm = 0.0
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"LoRA total grad norm: {lora_grad_norm:.8f}")

    if lora_grad_norm > 1e-6:
        print("\n✅ LoRA gradients flow without centering!")
        print("   → The centering operation is killing gradients")
    else:
        print("\n❌ LoRA gradients still zero without centering")
        print("   → Problem is elsewhere (not centering)")


def test_gradient_scaling():
    """Test if scaling gradients fixes the issue."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n{'='*60}")
    print("POTENTIAL FIX: Scaled centering")
    print("="*60)

    # The issue: psi = psi_raw - psi_raw.mean() has gradient that gets killed
    # when sum of upstream gradients is near zero.
    #
    # One fix: Use detached mean so gradient flows directly:
    # psi = psi_raw - psi_raw.mean().detach()
    #
    # This means psi is still centered (zero mean), but gradients don't
    # go through the mean computation.

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=1024,
    )

    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="none",  # We'll do centering manually with detach
    ).to(device)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Manual forward with detached centering
    theta = model.theta(batch["agent_idx"])
    beta = model.beta(batch["task_idx"])

    outputs = model.encoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    hidden_states = outputs.last_hidden_state

    batch_size = hidden_states.size(0)
    seq_len = hidden_states.size(1)
    positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = batch["attention_mask"]
    masked_positions = positions * attention_mask.long() - (1 - attention_mask.long())
    last_token_positions = masked_positions.argmax(dim=1)

    batch_indices = torch.arange(batch_size, device=hidden_states.device)
    last_token_hidden = hidden_states[batch_indices, last_token_positions]

    psi_raw = model.psi_head(last_token_hidden.float())

    # DETACHED CENTERING - gradients flow directly through psi_raw
    psi = psi_raw - psi_raw.mean().detach()

    logits = theta - (beta + psi)
    loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), batch["response"])
    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    print(f"\npsi_head.weight grad norm: {model.psi_head.weight.grad.norm().item():.8f}")
    print(f"theta.weight grad norm: {model.theta.weight.grad.norm().item():.8f}")
    print(f"beta.weight grad norm: {model.beta.weight.grad.norm().item():.8f}")

    lora_grad_norm = 0.0
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"LoRA total grad norm: {lora_grad_norm:.8f}")

    if lora_grad_norm > 1e-6:
        print("\n✅ DETACHED CENTERING FIXES GRADIENT FLOW!")
        print("   psi is still zero-mean but gradients flow directly")
    else:
        print("\n❌ Still no gradients with detached centering")


if __name__ == "__main__":
    print("="*60)
    print("GRADIENT FLOW DEBUGGING")
    print("="*60)

    # Main analysis
    results = analyze_gradient_flow()

    # Control test without centering
    test_without_centering()

    # Test potential fix
    test_gradient_scaling()
