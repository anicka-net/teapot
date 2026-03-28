#!/usr/bin/env python3
"""
Teapot Full Fine-Tuning Backend — DeepSpeed ZeRO-3 with CPU Offload

Full parameter training (no LoRA) for cases where format-level behavior
needs to reach embedding and output layers. Uses HuggingFace Trainer
with DeepSpeed ZeRO-3 and CPU offload for optimizer states.

Launch via:
    deepspeed --num_gpus 2 teapot/train_full_hf.py \
        --model swiss-ai/Apertus-70B-Instruct-2509 \
        --data train.jsonl \
        --deepspeed deepspeed_configs/zero3-offload.json \
        --epochs 2 --lr 2e-5 --template apertus-think
"""

import argparse
import json
import sys

import torch
from teapot.train_common import FormattedDataset, collate_fn, verify_template_tokens
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description="Full fine-tuning with DeepSpeed ZeRO-3")
    parser.add_argument("--model", required=True, help="HuggingFace model or local path")
    parser.add_argument("--data", required=True, help="Training JSONL (from teapot compose)")
    parser.add_argument("--output", default="output-full", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--template", default=None)
    parser.add_argument("--deepspeed", default=None, help="DeepSpeed config JSON")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")

    args = parser.parse_args()

    print("=" * 60)
    print("TEAPOT FULL FINE-TUNING (DeepSpeed ZeRO-3)")
    print(f"Model:  {args.model}")
    print(f"Data:   {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Batch:  {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nVerifying template tokens for: {args.template or 'none'}")
    if not verify_template_tokens(tokenizer, args.template):
        sys.exit(1)

    # Training arguments — must be created BEFORE model loading so
    # the Trainer's DeepSpeed integration can manage device placement.
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    # Load model with ZeRO-3 aware loading via Accelerate.
    # The key: create TrainingArguments first (with deepspeed config),
    # then use Trainer's model_init or load with the HF DeepSpeed plugin.
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(args.model)
    config.use_cache = False

    # Load on CPU with eager attention — SDPA/FlashAttention can fail on
    # some CUDA/cuDNN version combinations with non-standard architectures.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"Model: {model.config.num_hidden_layers} layers, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")

    # Load dataset
    print("\nLoading dataset...")
    dataset = FormattedDataset(
        args.data, tokenizer,
        max_length=args.max_length,
        template=args.template,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving to {args.output}/final/")
    trainer.save_model(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
