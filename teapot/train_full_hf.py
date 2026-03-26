#!/usr/bin/env python3
"""
Teapot Full Fine-Tuning Backend — DeepSpeed ZeRO-3 with CPU Offload

Full parameter training (no LoRA) for cases where format-level behavior
needs to reach embedding and output layers. Uses HuggingFace Trainer
with DeepSpeed ZeRO-3 and CPU offload for optimizer states.

Launch via:
    deepspeed --num_gpus 2 -m teapot.train_full_hf \\
        --model swiss-ai/Apertus-70B-Instruct-2509 \\
        --data train.jsonl \\
        --deepspeed deepspeed_configs/zero3-offload.json \\
        --epochs 2 --lr 2e-5 --template apertus-think

Or via teapot:
    teapot train configs/apertus-70b-full.config --backend full-hf
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset


class FormattedDataset(Dataset):
    """Dataset that reads pre-formatted text from compose output.

    Each JSONL line must have:
    - 'text': fully formatted training string (from templates.py)
    - 'assistant_spans': [[start, end], ...] character spans for loss masking

    OR (fallback):
    - 'conversations': list of {role, content} dicts (template applied at load time)
    """

    def __init__(self, data_path, tokenizer, max_length=4096, template=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)

                if "text" in ex and ex["text"]:
                    # Pre-formatted by compose — use directly
                    text = ex["text"]
                    spans = ex.get("assistant_spans", [])
                else:
                    # Fallback: apply template at load time
                    convs = ex.get("conversations", ex.get("messages", []))
                    if template:
                        from teapot.templates import format_conversation
                        text, spans = format_conversation(convs, template, thinking=True)
                        if text is None:
                            # Template returned None — skip
                            continue
                    else:
                        # No template — concatenate as-is (not recommended)
                        text = "\n".join(m.get("content", "") for m in convs)
                        spans = []

                self.examples.append({"text": text, "spans": spans})

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        spans = ex["spans"]

        # Tokenize with special tokens recognized
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        offsets = encoding.get("offset_mapping", [])

        # Build labels: -100 for non-assistant tokens
        labels = [-100] * len(input_ids)

        if spans and offsets:
            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:
                    continue
                # Check if this token falls within any assistant span
                for span_start, span_end in spans:
                    if tok_start >= span_start and tok_end <= span_end:
                        labels[token_idx] = input_ids[token_idx]
                        break
        else:
            # No spans — train on everything (fallback)
            labels = list(input_ids)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(torch.cat([ex["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def verify_special_tokens(tokenizer):
    """Verify that Apertus special tokens tokenize as single IDs."""
    critical = {
        "<|system_start|>": 61,
        "<|system_end|>": 62,
        "<|user_start|>": 65,
        "<|user_end|>": 66,
        "<|assistant_start|>": 67,
        "<|assistant_end|>": 68,
        "<|inner_prefix|>": 69,
        "<|inner_suffix|>": 70,
    }

    ok = True
    for token, expected_id in critical.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids != [expected_id]:
            print(f"WARNING: {token} encodes as {ids}, expected [{expected_id}]")
            ok = False
        else:
            print(f"  OK: {token} → [{expected_id}]")

    if not ok:
        print("\nWARNING: Special tokens not recognized. The model may learn")
        print("to generate them as text instead of control tokens.")
        print("Check tokenizer_config.json and added_tokens.json.")

    return ok


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
    parser.add_argument("--template", default=None,
                        help="Chat template (if data doesn't have pre-formatted text)")
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

    # Load tokenizer and verify special tokens
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nVerifying special tokens:")
    verify_special_tokens(tokenizer)

    # Load model in bf16 (no quantization)
    print("\nLoading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Required for gradient checkpointing
    )
    model.gradient_checkpointing_enable()

    print(f"Model: {model.config.num_hidden_layers} layers, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")

    # Load dataset
    print("\nLoading dataset...")
    dataset = FormattedDataset(
        args.data, tokenizer,
        max_length=args.max_length,
        template=args.template,
    )

    # Training arguments
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

    # Trainer
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
