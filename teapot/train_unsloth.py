#!/usr/bin/env python3
"""
Teapot Unsloth Backend — fast QLoRA training for 8B models.

Unsloth provides 2-4x speedup over HF Trainer for QLoRA on single GPU.
Best for quick iterations on 8B models (ai01 L40, consumer GPUs).

Usage:
    python3 -m teapot.train_unsloth \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --data train.jsonl \\
        --epochs 3 --lr 2e-4 --template auto

Or via teapot:
    teapot train configs/ke-v11-reward-model.config --backend unsloth
"""

import argparse
import json
import sys
from pathlib import Path

from teapot.templates import TEMPLATES, format_conversation


def verify_unsloth():
    """Check unsloth is installed."""
    try:
        from unsloth import FastLanguageModel
        return True
    except ImportError:
        print("ERROR: unsloth not installed.")
        print("  pip install unsloth")
        print("  See: https://github.com/unslothai/unsloth")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fast QLoRA training with unsloth")
    parser.add_argument("--model", required=True, help="HuggingFace model or local path")
    parser.add_argument("--data", required=True, help="Training JSONL (from teapot compose)")
    parser.add_argument("--output", default="output-unsloth", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--template", default=None,
                        help="Chat template (if data doesn't have pre-formatted text)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    args = parser.parse_args()

    if not verify_unsloth():
        sys.exit(1)

    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print("=" * 60)
    print("TEAPOT UNSLOTH TRAINING (fast QLoRA)")
    print(f"Model:  {args.model}")
    print(f"Data:   {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"LoRA:   r={args.lora_r}, alpha={args.lora_alpha}")
    print("=" * 60)

    # Load model with unsloth (handles 4bit quantization automatically)
    print("\nLoading model with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,  # unsloth optimized for 0 dropout
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and format data
    print("\nLoading data...")
    examples = []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            if "text" in ex and ex["text"]:
                # Pre-formatted by compose
                examples.append({"text": ex["text"]})
            else:
                # Apply template
                convs = ex.get("conversations", ex.get("messages", []))
                if args.template:
                    text, _ = format_conversation(convs, args.template, thinking=True)
                    if text:
                        examples.append({"text": text})
                else:
                    # Use tokenizer's chat template
                    text = tokenizer.apply_chat_template(
                        convs, tokenize=False, add_generation_prompt=False
                    )
                    examples.append({"text": text})

    print(f"Loaded {len(examples)} examples")
    dataset = Dataset.from_list(examples)

    # Training
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        packing=True,
    )

    print("\nStarting training...")
    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output}/final/")
    model.save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")

    # Also save merged model if space allows
    print(f"Saving merged model to {args.output}/merged/")
    try:
        model.save_pretrained_merged(
            f"{args.output}/merged",
            tokenizer,
            save_method="merged_16bit",
        )
    except Exception as e:
        print(f"Merged save failed (likely OOM): {e}")
        print("LoRA adapter saved successfully — merge manually later.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  LoRA: {args.output}/final/")
    print(f"  Merged: {args.output}/merged/ (if available)")
    print("=" * 60)


if __name__ == "__main__":
    main()
