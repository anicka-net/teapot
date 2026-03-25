#!/usr/bin/env python3
"""
Training script for openSUSE CVE Backport Model — H100 runs.

Supports QLoRA fine-tuning of dense models (Qwen2.5-Coder-32B, Qwen3-32B,
Qwen3.5-27B) with label masking on assistant tokens.

Usage:
  python3 scripts/train.py \
      --model Qwen/Qwen3.5-27B \
      --data data/combined-train.jsonl \
      --eval data/curated-eval.jsonl \
      --qlora --max-length 8192
"""

import faulthandler
faulthandler.enable()

import json
import argparse
import shutil
import signal
import sys
import time
import torch

def _signal_handler(signum, frame):
    print(f"\n!!! Received signal {signum} ({signal.Signals(signum).name}) !!!", flush=True)
    sys.exit(128 + signum)

for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(sig, _signal_handler)

from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def parse_args():
    p = argparse.ArgumentParser(description="Train CVE backport model with QLoRA")

    # Data
    p.add_argument("--data", required=True, help="Training JSONL")
    p.add_argument("--eval", required=True, help="Eval JSONL")

    # Model
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct",
                   help="Base model HuggingFace ID")
    p.add_argument("--output", default="./output",
                   help="Output directory")
    p.add_argument("--resume", default=None,
                   help="Resume from checkpoint directory")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=8192,
                   help="Max sequence length in tokens")
    p.add_argument("--warmup-ratio", type=float, default=0.05)

    # LoRA
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Hardware
    p.add_argument("--qlora", action="store_true",
                   help="Use 4-bit QLoRA")
    p.add_argument("--no-flash-attn", action="store_true",
                   help="Disable Flash Attention 2 (use SDPA instead)")

    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default="cve-backport")
    p.add_argument("--run-name", default=None)

    # HuggingFace upload
    p.add_argument("--hf-upload", default=None,
                   help="HuggingFace repo ID to upload model")
    p.add_argument("--hf-token", default=None,
                   help="HF token (or path to file containing token)")

    return p.parse_args()


def load_chatml(path):
    """Load ChatML JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def tokenize_with_label_masking(example, tokenizer, max_length):
    """Tokenize ChatML example with label masking.

    Only compute loss on assistant tokens, not system/user tokens.
    Supports both 3-turn (system/user/assistant) and multi-turn
    conversations (system/user/assistant/user/assistant/...).
    """
    messages = example.get("messages") or example.get("conversations", [])

    # Disable thinking mode for Qwen3.5 (our training data has no <think> blocks)
    template_kwargs = {}
    if hasattr(tokenizer, 'chat_template') and 'enable_thinking' in str(tokenizer.chat_template or ''):
        template_kwargs['enable_thinking'] = False

    # Tokenize the full conversation
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **template_kwargs
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # Build labels by finding assistant message boundaries.
    # Tokenize each prefix to find where each assistant segment
    # starts and ends, then unmask only those regions.
    labels = [-100] * len(full_tokens["input_ids"])

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Prefix up to (but not including) this assistant message
        prefix_text = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        prefix_len = len(tokenizer(
            prefix_text, truncation=True, max_length=max_length,
            padding=False, return_tensors=None,
        )["input_ids"])

        # Prefix including this assistant message
        through_text = tokenizer.apply_chat_template(
            messages[:i + 1], tokenize=False, add_generation_prompt=False, **template_kwargs
        )
        through_len = len(tokenizer(
            through_text, truncation=True, max_length=max_length,
            padding=False, return_tensors=None,
        )["input_ids"])

        # Unmask the assistant segment
        for j in range(prefix_len, min(through_len, len(labels))):
            labels[j] = full_tokens["input_ids"][j]

    full_tokens["labels"] = labels
    return full_tokens


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("CVE Backport Model — H100 Training")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Eval:     {args.eval}")
    print(f"Output:   {args.output}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"LR:       {args.lr}")
    print(f"LoRA:     r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Max len:  {args.max_length}")
    print(f"QLoRA:    {args.qlora}")
    print(f"GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM:     {vram:.0f} GB")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load datasets
    print("Loading datasets...")
    train_raw = load_chatml(args.data)
    eval_raw = load_chatml(args.eval)
    print(f"  Train: {len(train_raw)} examples")
    print(f"  Eval:  {len(eval_raw)} examples")

    # Tier distribution
    tiers = {}
    for ex in train_raw:
        tier = ex.get("metadata", {}).get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
    print(f"  Tiers: {tiers}")

    # Tokenize with label masking
    print("\nTokenizing with label masking...")
    train_tokenized = []
    skipped = 0
    for ex in train_raw:
        tok = tokenize_with_label_masking(ex, tokenizer, args.max_length)
        n_train_tokens = sum(1 for l in tok["labels"] if l != -100)
        if n_train_tokens < 10:
            skipped += 1
            continue
        train_tokenized.append(tok)

    eval_tokenized = []
    for ex in eval_raw:
        tok = tokenize_with_label_masking(ex, tokenizer, args.max_length)
        n_train_tokens = sum(1 for l in tok["labels"] if l != -100)
        if n_train_tokens >= 10:
            eval_tokenized.append(tok)

    if skipped:
        print(f"  Skipped {skipped} examples with no trainable tokens")

    # Stats
    train_lengths = [len(t["input_ids"]) for t in train_tokenized]
    train_label_counts = [sum(1 for l in t["labels"] if l != -100) for t in train_tokenized]
    print(f"  Train: {len(train_tokenized)} examples")
    print(f"    Sequence length: min={min(train_lengths)}, max={max(train_lengths)}, "
          f"avg={sum(train_lengths)//len(train_lengths)}")
    print(f"    Trainable tokens: min={min(train_label_counts)}, max={max(train_label_counts)}, "
          f"avg={sum(train_label_counts)//len(train_label_counts)}")
    print(f"  Eval: {len(eval_tokenized)} examples")

    train_dataset = Dataset.from_list(train_tokenized)
    eval_dataset = Dataset.from_list(eval_tokenized)

    # Load model
    print(f"\nLoading {args.model}...")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Attention implementation: let SDPA be the default for Qwen3.5
    # (DeltaNet layers don't use flash attention, standard attn layers
    # get SDPA which is fast and safe). Use flash_attention_2 explicitly
    # only for pure-transformer models.
    if not args.no_flash_attn and "qwen3.5" not in args.model.lower():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if args.qlora:
        print("  Using 4-bit QLoRA quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Apply LoRA
    mode_str = "QLoRA" if args.qlora else "LoRA"
    print(f"Applying {mode_str} (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    report_to = "wandb" if args.wandb else "none"
    run_name = args.run_name or f"cve-backport-{Path(args.model).name}"

    training_args = TrainingArguments(
        output_dir=args.output,
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=report_to,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    total_steps = (len(train_tokenized) * args.epochs) // (args.batch_size * args.grad_accum)
    print(f"\nStarting training (~{total_steps} steps)...")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.1f} hours")

    # Save adapter
    print(f"\nSaving LoRA adapter to {args.output}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    # Clean up checkpoints
    output_path = Path(args.output)
    for ckpt in sorted(output_path.glob("checkpoint-*")):
        print(f"  Removing checkpoint: {ckpt.name}")
        shutil.rmtree(ckpt)

    # Save merged model (for QLoRA: save adapter only)
    merged_dir = output_path / "merged"
    if args.qlora:
        print("QLoRA mode: adapter saved (merge separately with merge.py)")
        merged_dir = output_path
    else:
        print(f"Merging LoRA weights to {merged_dir}...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    # Upload to HuggingFace if requested
    if args.hf_upload:
        print(f"\nUploading to HuggingFace: {args.hf_upload}")
        token = args.hf_token
        if token and Path(token).is_file():
            token = Path(token).read_text().strip()
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(args.hf_upload, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=str(merged_dir),
                repo_id=args.hf_upload,
                commit_message=f"CVE backport model ({Path(args.model).name})",
            )
            print(f"  Uploaded to: https://huggingface.co/{args.hf_upload}")
        except Exception as e:
            print(f"  HF upload failed: {e}")
            print("  Model saved locally, upload manually if needed.")

    print(f"\nDone! Total time: {elapsed/3600:.1f} hours")
    print(f"  LoRA adapter: {args.output}")


if __name__ == "__main__":
    main()
