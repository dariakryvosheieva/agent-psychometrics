#!/usr/bin/env python3
"""Debug gradient flow mimicking EXACT training loop conditions.

This replicates the Trainer's behavior to find where gradients die.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from experiment_sad_irt.model import SADIRT
from experiment_sad_irt.dataset import TrajectoryIRTDataset
from experiment_sad_irt.train import compute_gradient_norms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW


def test_with_trainer_setup():
    """Test with exact Trainer setup - optimizer, param groups, etc."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        psi_normalization="center",
    ).to(device)

    # ========================================
    # EXACT SAME OPTIMIZER SETUP AS TRAINER
    # ========================================
    encoder_params = []
    embedding_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "theta" in name or "beta" in name:
            embedding_params.append(param)
        else:
            encoder_params.append(param)

    print(f"\nEncoder params: {len(encoder_params)}")
    print(f"Embedding params: {len(embedding_params)}")

    # Check what's in encoder_params
    encoder_param_names = [n for n, p in model.named_parameters() if p.requires_grad and "theta" not in n and "beta" not in n]
    print(f"\nEncoder param names (first 10):")
    for name in encoder_param_names[:10]:
        print(f"  {name}")

    optimizer = AdamW([
        {"params": encoder_params, "lr": 1e-4},
        {"params": embedding_params, "lr": 1e-3},
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()

    # ========================================
    # SIMULATE GRADIENT ACCUMULATION
    # ========================================
    gradient_accumulation_steps = 8
    loader = DataLoader(dataset, batch_size=4, shuffle=True)  # batch_size * accum = 32 effective

    print(f"\n{'='*60}")
    print(f"Simulating {gradient_accumulation_steps} gradient accumulation steps")
    print(f"{'='*60}")

    optimizer.zero_grad()

    for accum_step, batch in enumerate(loader):
        if accum_step >= gradient_accumulation_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward (using model directly, not _forward_with_psi_tracking)
        logits = model(
            agent_idx=batch["agent_idx"],
            task_idx=batch["task_idx"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = criterion(logits, batch["response"])
        loss = loss / gradient_accumulation_steps

        loss.backward()

        if accum_step == 0:
            print(f"\nAfter accum step 0:")
            grad_norms = compute_gradient_norms(model, detailed=False)
            print(f"  Gradient norms: {grad_norms}")

    # Check gradients after full accumulation
    print(f"\nAfter {gradient_accumulation_steps} accumulation steps:")
    grad_norms = compute_gradient_norms(model, detailed=True)
    print(f"  total={grad_norms['total_norm']:.6f}")
    print(f"  embedding={grad_norms.get('embedding_norm', 0):.6f}")
    print(f"  encoder={grad_norms.get('encoder_norm', 0):.6f}")
    print(f"  head={grad_norms.get('head_norm', 0):.6f}")

    # Check specific params
    detailed = grad_norms.get('detailed', {})
    print(f"\nKey param gradients:")
    for name in ['psi_head.weight', 'psi_head.bias', 'theta.weight', 'beta.weight']:
        for full_name, norm in detailed.items():
            if name in full_name:
                print(f"  {name}: {norm:.8f}")
                break

    # LoRA gradients
    lora_grads = [(k, v) for k, v in detailed.items() if 'lora' in k]
    if lora_grads:
        lora_grads.sort(key=lambda x: -x[1])
        print(f"\nTop 5 LoRA gradients:")
        for name, norm in lora_grads[:5]:
            print(f"  {name}: {norm:.8f}")

        total_lora = sum(v for _, v in lora_grads)
        print(f"\nTotal LoRA grad (sum): {total_lora:.8f}")


def test_with_forward_psi_tracking():
    """Test with _forward_with_psi_tracking code path (what debug_gradients=True uses)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n{'='*60}")
    print("Testing _forward_with_psi_tracking code path")
    print("(This is what runs when debug_gradients=True)")
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

    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",
    ).to(device)

    # Same optimizer setup
    encoder_params = []
    embedding_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "theta" in name or "beta" in name:
            embedding_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = AdamW([
        {"params": encoder_params, "lr": 1e-4},
        {"params": embedding_params, "lr": 1e-3},
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()

    # ========================================
    # EXACT COPY OF _forward_with_psi_tracking
    # ========================================
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

    # Apply centering
    psi = psi_raw - psi_raw.mean()

    logits = theta - (beta + psi)
    logits = logits.squeeze(-1)

    # ========================================

    loss = criterion(logits, batch["response"])
    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    # Check gradients
    grad_norms = compute_gradient_norms(model, detailed=True)
    print(f"\nGradient norms:")
    print(f"  total={grad_norms['total_norm']:.6f}")
    print(f"  embedding={grad_norms.get('embedding_norm', 0):.6f}")
    print(f"  encoder={grad_norms.get('encoder_norm', 0):.6f}")
    print(f"  head={grad_norms.get('head_norm', 0):.6f}")

    detailed = grad_norms.get('detailed', {})
    print(f"\nKey param gradients:")
    print(f"  psi_head.weight: {detailed.get('psi_head.weight', 0):.8f}")
    print(f"  psi_head.bias: {detailed.get('psi_head.bias', 0):.8f}")

    lora_grads = [(k, v) for k, v in detailed.items() if 'lora' in k]
    total_lora = sum(v for _, v in lora_grads) if lora_grads else 0
    print(f"\nTotal LoRA grad (sum): {total_lora:.8f}")

    if total_lora < 1e-6:
        print("\n❌ LoRA gradients are zero in _forward_with_psi_tracking!")
    else:
        print("\n✅ LoRA gradients flowing in _forward_with_psi_tracking")


def test_multiple_optimizer_steps():
    """Test if gradients die after multiple optimizer steps."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n{'='*60}")
    print("Testing multiple optimizer steps (simulating step 60)")
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

    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",
    ).to(device)

    encoder_params = []
    embedding_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "theta" in name or "beta" in name:
            embedding_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = AdamW([
        {"params": encoder_params, "lr": 1e-4},
        {"params": embedding_params, "lr": 1e-3},
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    gradient_accumulation_steps = 8

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    loader_iter = iter(loader)

    # Simulate 60 optimizer steps
    num_steps = 60
    print(f"Running {num_steps} optimizer steps...")

    for step in range(num_steps):
        optimizer.zero_grad()

        for accum_step in range(gradient_accumulation_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                agent_idx=batch["agent_idx"],
                task_idx=batch["task_idx"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = criterion(logits, batch["response"])
            loss = loss / gradient_accumulation_steps
            loss.backward()

        # Log at steps 0, 10, 30, 59
        if step in [0, 10, 30, 59]:
            grad_norms = compute_gradient_norms(model, detailed=True)
            detailed = grad_norms.get('detailed', {})
            lora_grads = [(k, v) for k, v in detailed.items() if 'lora' in k]
            total_lora = sum(v for _, v in lora_grads) if lora_grads else 0

            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            print(f"  Total grad norm: {grad_norms['total_norm']:.6f}")
            print(f"  Embedding grad norm: {grad_norms.get('embedding_norm', 0):.6f}")
            print(f"  Encoder grad norm: {grad_norms.get('encoder_norm', 0):.6f}")
            print(f"  Head grad norm: {grad_norms.get('head_norm', 0):.6f}")
            print(f"  LoRA grad (sum): {total_lora:.6f}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    print("\n" + "="*60)
    if total_lora < 1e-6:
        print("❌ LoRA gradients died during training!")
    else:
        print("✅ LoRA gradients still flowing after 60 steps")


if __name__ == "__main__":
    test_with_trainer_setup()
    test_with_forward_psi_tracking()
    test_multiple_optimizer_steps()
