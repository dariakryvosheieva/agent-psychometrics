#!/usr/bin/env python3
"""Debug with EXACT SLURM config: batch_size=32, gradient_accumulation_steps=2."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from experiment_sad_irt.model import SADIRT
from experiment_sad_irt.dataset import TrajectoryIRTDataset
from experiment_sad_irt.train import compute_gradient_norms, Trainer
from experiment_sad_irt.config import SADIRTConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW


def test_exact_slurm_config():
    """Test with exact SLURM script configuration."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TrajectoryIRTDataset(
        response_matrix_path="clean_data/swebench_verified/swebench_verified_20251120_full.jsonl",
        trajectory_dir="chris_output/trajectory_summaries_api",
        tokenizer=tokenizer,
        max_length=1024,  # Same as SLURM
    )

    print(f"\nDataset: {len(dataset)} samples, {dataset.num_agents} agents, {dataset.num_tasks} tasks")

    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",  # Default when not specified
    ).to(device)

    # Print model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total params")

    # ========================================
    # EXACT SLURM CONFIG
    # ========================================
    BATCH_SIZE = 32  # From SLURM script
    GRADIENT_ACCUMULATION_STEPS = 2  # From SLURM script

    print(f"\nUsing SLURM config: batch_size={BATCH_SIZE}, grad_accum={GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

    # Same optimizer setup as Trainer
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

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n{'='*60}")
    print("Running 60 optimizer steps with SLURM config...")
    print("="*60)

    loader_iter = iter(loader)

    for step in range(60):
        optimizer.zero_grad()

        for accum_step in range(GRADIENT_ACCUMULATION_STEPS):
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
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

        # Log at key steps
        if step in [0, 10, 30, 59]:
            grad_norms = compute_gradient_norms(model, detailed=True)
            detailed = grad_norms.get('detailed', {})

            lora_grads = [(k, v) for k, v in detailed.items() if 'lora' in k]
            total_lora = sum(v for _, v in lora_grads) if lora_grads else 0

            # Get top 3 LoRA grads
            lora_grads.sort(key=lambda x: -x[1])
            top3_lora = lora_grads[:3]

            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            print(f"  total={grad_norms['total_norm']:.6f}, embedding={grad_norms.get('embedding_norm', 0):.6f}, encoder={grad_norms.get('encoder_norm', 0):.6f}, head={grad_norms.get('head_norm', 0):.6f}")
            print(f"  psi_head.weight: {detailed.get('psi_head.weight', 0):.8f}")
            print(f"  psi_head.bias: {detailed.get('psi_head.bias', 0):.8f}")
            print(f"  theta.weight: {detailed.get('theta.weight', 0):.8f}")
            print(f"  beta.weight: {detailed.get('beta.weight', 0):.8f}")
            print(f"  Top 3 LoRA grads: {top3_lora}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print("\n" + "="*60)
    if grad_norms.get('encoder_norm', 0) < 1e-6:
        print("❌ Encoder (LoRA) gradients are zero!")
    else:
        print("✅ Encoder (LoRA) gradients are flowing")

    if grad_norms.get('head_norm', 0) < 1e-6:
        print("❌ Head (psi_head) gradients are zero!")
    else:
        print("✅ Head (psi_head) gradients are flowing")


def test_with_actual_trainer():
    """Test using the actual Trainer class (closest to real training)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\n{'='*60}")
    print("Testing with ACTUAL Trainer class")
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

    # Limit to first 500 samples for speed
    dataset.samples = dataset.samples[:500]
    print(f"Using {len(dataset)} samples for quick test")

    model = SADIRT(
        num_agents=dataset.num_agents,
        num_tasks=dataset.num_tasks,
        model_name=model_name,
        psi_normalization="center",
    ).to(device)

    # Create config matching SLURM
    config = SADIRTConfig(
        batch_size=32,
        gradient_accumulation_steps=2,
        epochs=1,
        debug_gradients=True,
        logging_steps=10,
        output_dir="/tmp/debug_trainer_test",
    )

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        eval_loader=None,
        config=config,
        device=device,
        is_sad_irt=True,
    )

    print("\nRunning Trainer.train() - watch for gradient logs...")
    print("(Trainer uses _forward_with_psi_tracking when debug_gradients=True)")
    trainer.train()

    print("\n✅ Check logs above for gradient values")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-trainer", action="store_true", help="Also test with actual Trainer class")
    args = parser.parse_args()

    test_exact_slurm_config()

    if args.with_trainer:
        test_with_actual_trainer()
