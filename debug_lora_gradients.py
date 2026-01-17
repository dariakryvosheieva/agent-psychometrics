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


def test_with_real_data():
    """Test using actual TrajectoryIRTDataset with real data."""

    import sys
    sys.path.insert(0, '.')

    from experiment_sad_irt.model import SADIRT
    from experiment_sad_irt.dataset import TrajectoryIRTDataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Testing with REAL trajectory data")
    print(f"{'='*60}")

    # Load tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (just a subset for testing)
    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=512,  # Shorter for speed
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Num agents: {dataset.num_agents}, Num tasks: {dataset.num_tasks}")

    # Create model with actual dimensions
    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",
    ).to(device)

    # Create dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get a batch
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"Batch shapes: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}")
    print(f"Agent indices: {batch['agent_idx'].tolist()}")
    print(f"Task indices: {batch['task_idx'].tolist()}")
    print(f"Labels: {batch['response'].tolist()}")

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


def test_debug_token_positions():
    """Debug: check what's happening with token positions in real data."""

    import sys
    sys.path.insert(0, '.')

    from experiment_sad_irt.model import SADIRT
    from experiment_sad_irt.dataset import TrajectoryIRTDataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("DEBUG: Token positions and gradient flow analysis")
    print(f"{'='*60}")

    # Load tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"pad_token_id: {tokenizer.pad_token_id}")

    # Load dataset
    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=512,
    )

    # Get one sample
    sample = dataset[0]
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    print(f"\nSample 0:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")
    print(f"  attention_mask sum (non-pad tokens): {attention_mask.sum().item()}")

    # Find first and last non-padding token
    non_pad_positions = (attention_mask == 1).nonzero(as_tuple=True)[0]
    first_non_pad = non_pad_positions[0].item() if len(non_pad_positions) > 0 else -1
    last_non_pad = non_pad_positions[-1].item() if len(non_pad_positions) > 0 else -1

    print(f"  First non-pad position: {first_non_pad}")
    print(f"  Last non-pad position: {last_non_pad}")
    print(f"  Padding is on the {'LEFT' if first_non_pad > 0 else 'NONE or RIGHT'}")

    # Check OLD seq_lengths calculation (the bug)
    old_calc = attention_mask.sum() - 1
    print(f"  OLD calculation (mask.sum() - 1): {old_calc.item()}")
    print(f"  This was {'INCORRECT' if old_calc != last_non_pad else 'correct'} for left-padded data")

    # Check NEW calculation (the fix)
    seq_len = attention_mask.size(0)
    positions = torch.arange(seq_len)
    masked_positions = positions * attention_mask.long() - (1 - attention_mask.long())
    new_calc = masked_positions.argmax()
    print(f"  NEW calculation (argmax of masked positions): {new_calc.item()}")
    print(f"  This is {'CORRECT' if new_calc == last_non_pad else 'INCORRECT'} for left-padded data")

    if old_calc != last_non_pad:
        print(f"\n  ✅ LEFT PADDING BUG CONFIRMED:")
        print(f"     OLD code would use position {old_calc.item()} (inside padding!)")
        print(f"     NEW code correctly uses position {new_calc.item()} (last real token)")

    # Now test with a batch
    print(f"\n{'='*60}")
    print("Testing gradient flow with batch of real data")
    print(f"{'='*60}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print(f"\nBatch:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")

    # Check each sample in batch
    for i in range(input_ids.shape[0]):
        mask = attention_mask[i]
        non_pad_positions = (mask == 1).nonzero(as_tuple=True)[0]
        first_non_pad = non_pad_positions[0].item() if len(non_pad_positions) > 0 else -1
        last_non_pad = non_pad_positions[-1].item() if len(non_pad_positions) > 0 else -1
        seq_len_calc = mask.sum().item() - 1
        print(f"  Sample {i}: non-pad range [{first_non_pad}, {last_non_pad}], mask.sum()-1 = {seq_len_calc}")
        if seq_len_calc != last_non_pad:
            print(f"    ⚠️  MISMATCH!")

    # Create model and test forward/backward
    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",
    ).to(device)

    # Forward pass
    logits = model(
        agent_idx=batch["agent_idx"],
        task_idx=batch["task_idx"],
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["response"])
    print(f"\nLoss: {loss.item():.6f}")

    # Backward
    loss.backward()

    # Check gradients
    lora_grad_norm = 0.0
    lora_count = 0
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
            lora_count += 1

    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"LoRA total grad norm: {lora_grad_norm:.8f} ({lora_count} params)")
    print(f"psi_head.weight grad norm: {model.psi_head.weight.grad.norm().item():.8f}")

    if lora_grad_norm < 1e-10:
        print(f"\n❌ LoRA gradients are NOT flowing")
        return False
    else:
        print(f"\n✅ LoRA gradients ARE flowing")
        return True


def test_with_trainer():
    """Test using actual Trainer class with one step."""

    import sys
    sys.path.insert(0, '.')

    from experiment_sad_irt.model import SADIRT
    from experiment_sad_irt.dataset import TrajectoryIRTDataset
    from experiment_sad_irt.train import Trainer
    from experiment_sad_irt.config import SADIRTConfig
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Testing with actual Trainer class (1 epoch)")
    print(f"{'='*60}")

    # Config
    config = SADIRTConfig(
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=2,
        debug_gradients=True,
        logging_steps=1,
        output_dir="/tmp/debug_trainer",
    )

    # Load tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (just a subset for testing)
    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=512,
    )

    # Use just first 20 samples
    dataset.samples = dataset.samples[:20]
    print(f"Dataset size: {len(dataset)} samples")

    # Create model with actual dimensions
    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",
    ).to(device)

    # Create dataloader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        eval_loader=None,
        config=config,
        device=device,
        is_sad_irt=True,
    )

    # Run one epoch (should log gradients at each step)
    print("\nRunning one epoch with Trainer...")
    trainer.train()

    # Check final state
    lora_grad_norm = 0.0
    lora_count = 0
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad and param.grad is not None:
            lora_grad_norm += param.grad.norm().item() ** 2
            lora_count += 1

    # Note: Gradients are zeroed after optimizer step, so this checks what's left
    lora_grad_norm = lora_grad_norm ** 0.5
    print(f"\nFinal LoRA grad norm: {lora_grad_norm:.8f} (may be 0 if optimizer.zero_grad was called)")

    # The real test is whether the gradients were logged as non-zero during training
    print("\n✅ Check the log output above for gradient values during training")

    return True  # Can't easily verify from final state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="Skip tests that need real data")
    args = parser.parse_args()

    print("="*60)
    print("Test 1: Simple forward-backward through LoRA encoder")
    print("="*60)
    test1_pass = test_lora_gradients()

    test2_pass = test_with_centering()

    test3_pass = test_with_grad_accumulation()

    test4_pass = test_with_actual_sadirt_model()

    test5_pass = None
    test6_pass = None
    if not args.skip_data:
        test5_pass = test_with_real_data()
        test6_pass = test_with_trainer()

    # Critical test: token position debugging
    test7_pass = None
    if not args.skip_data:
        test7_pass = test_debug_token_positions()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test 1 (simple): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (centering): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (grad accumulation): {'✅ PASS' if test3_pass else '❌ FAIL'}")
    print(f"Test 4 (actual SADIRT model): {'✅ PASS' if test4_pass else '❌ FAIL'}")
    if test5_pass is not None:
        print(f"Test 5 (real data): {'✅ PASS' if test5_pass else '❌ FAIL'}")
    if test6_pass is not None:
        print(f"Test 6 (actual Trainer): {'✅ PASS (see logs)' if test6_pass else '❌ FAIL'}")
    if test7_pass is not None:
        print(f"Test 7 (token positions): {'✅ PASS' if test7_pass else '❌ FAIL'}")
