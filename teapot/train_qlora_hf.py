#!/usr/bin/env python3
"""
Generic QLoRA/LoRA training backend using HuggingFace Trainer + PEFT.

This is Teapot's stable backend contract for direct Trainer-based runs
without Axolotl. It accepts Teapot-composed chat JSONL with either
`conversations` or `messages` and computes loss only on assistant turns.
"""

import argparse
import faulthandler
import json
import shutil
import signal
import sys
import time
from pathlib import Path

import torch

faulthandler.enable()


def _signal_handler(signum, frame):
    print(f"\n!!! Received signal {signum} ({signal.Signals(signum).name}) !!!", flush=True)
    sys.exit(128 + signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(sig, _signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Teapot dataset with QLoRA/LoRA via HuggingFace Trainer")

    parser.add_argument("--data", required=True, help="Training JSONL")
    parser.add_argument("--eval", default=None, help="Optional eval JSONL")

    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model HuggingFace ID")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint directory")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=8192, help="Max sequence length in tokens")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable Flash Attention 2 (use SDPA instead)")

    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="teapot-train")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--hf-upload", default=None, help="HuggingFace repo ID to upload model")
    parser.add_argument("--hf-token", default=None, help="HF token (or path to file containing token)")

    return parser.parse_args()


def load_chat_jsonl(path):
    examples = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def apply_chat_template(tokenizer, messages, add_generation_prompt):
    template_kwargs = {}
    if hasattr(tokenizer, "chat_template") and "enable_thinking" in str(tokenizer.chat_template or ""):
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


def tokenize_with_label_masking(example, tokenizer, max_length):
    """Tokenize a chat example and mask all non-assistant tokens."""
    if "text" in example:
        full_text = example["text"]
        spans = example.get("assistant_spans", [])
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        labels = [-100] * len(full_tokens["input_ids"])
        for start_char, end_char in spans:
            prefix_len = len(tokenizer(
                full_text[:start_char],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"])
            through_len = len(tokenizer(
                full_text[:end_char],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"])
            for j in range(prefix_len, min(through_len, len(labels))):
                labels[j] = full_tokens["input_ids"][j]
        full_tokens["labels"] = labels
        return full_tokens

    messages = example.get("messages") or example.get("conversations", [])

    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    labels = [-100] * len(full_tokens["input_ids"])

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        prefix_text = apply_chat_template(tokenizer, messages[:i], add_generation_prompt=True)
        prefix_len = len(tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"])

        through_text = apply_chat_template(tokenizer, messages[: i + 1], add_generation_prompt=False)
        through_len = len(tokenizer(
            through_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"])

        for j in range(prefix_len, min(through_len, len(labels))):
            labels[j] = full_tokens["input_ids"][j]

    full_tokens["labels"] = labels
    return full_tokens


def build_dataset(raw_examples, tokenizer, max_length):
    tokenized = []
    skipped = 0
    for ex in raw_examples:
        tok = tokenize_with_label_masking(ex, tokenizer, max_length)
        n_train_tokens = sum(1 for label in tok["labels"] if label != -100)
        if n_train_tokens < 10:
            skipped += 1
            continue
        tokenized.append(tok)
    return tokenized, skipped


def main():
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("Teapot QLoRA/LoRA Trainer")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Eval:     {args.eval or 'disabled'}")
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

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading datasets...")
    train_raw = load_chat_jsonl(args.data)
    eval_raw = load_chat_jsonl(args.eval) if args.eval else []
    print(f"  Train: {len(train_raw)} examples")
    print(f"  Eval:  {len(eval_raw)} examples")

    tiers = {}
    for ex in train_raw:
        tier = ex.get("metadata", {}).get("tier", "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
    print(f"  Tiers: {tiers}")

    print("\nTokenizing with label masking...")
    train_tokenized, skipped_train = build_dataset(train_raw, tokenizer, args.max_length)
    eval_tokenized, skipped_eval = build_dataset(eval_raw, tokenizer, args.max_length) if eval_raw else ([], 0)

    if skipped_train:
        print(f"  Skipped {skipped_train} train examples with no trainable tokens")
    if skipped_eval:
        print(f"  Skipped {skipped_eval} eval examples with no trainable tokens")

    train_lengths = [len(t["input_ids"]) for t in train_tokenized]
    train_label_counts = [sum(1 for l in t["labels"] if l != -100) for t in train_tokenized]
    print(f"  Train: {len(train_tokenized)} examples")
    print(f"    Sequence length: min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)//len(train_lengths)}")
    print(f"    Trainable tokens: min={min(train_label_counts)}, max={max(train_label_counts)}, avg={sum(train_label_counts)//len(train_label_counts)}")
    print(f"  Eval: {len(eval_tokenized)} examples")

    train_dataset = Dataset.from_list(train_tokenized)
    eval_dataset = Dataset.from_list(eval_tokenized) if eval_tokenized else None

    print(f"\nLoading {args.model}...")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if not args.no_flash_attn:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            print("  Flash Attention not installed, using default attention")
            pass

    if args.qlora:
        print("  Using 4-bit QLoRA quantization...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    report_to = "wandb" if args.wandb else "none"
    run_name = args.run_name or f"teapot-qlora-{Path(args.model).name}"

    training_kwargs = {
        "output_dir": args.output,
        "run_name": run_name,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 2,
        "report_to": report_to,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch_fused",
    }

    if eval_dataset is not None:
        training_kwargs.update({
            "per_device_eval_batch_size": 1,
            "eval_strategy": "steps",
            "eval_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        })
    else:
        training_kwargs.update({
            "eval_strategy": "no",
            "load_best_model_at_end": False,
        })

    training_args = TrainingArguments(**training_kwargs)

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

    print(f"\nSaving LoRA adapter to {args.output}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    output_path = Path(args.output)
    for ckpt in sorted(output_path.glob("checkpoint-*")):
        print(f"  Removing checkpoint: {ckpt.name}")
        shutil.rmtree(ckpt)

    merged_dir = output_path / "merged"
    if args.qlora:
        print("QLoRA mode: adapter saved (merge separately with merge.py)")
        merged_dir = output_path
    else:
        print(f"Merging LoRA weights to {merged_dir}...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

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
                commit_message=f"Teapot QLoRA model ({Path(args.model).name})",
            )
            print(f"  Uploaded to: https://huggingface.co/{args.hf_upload}")
        except Exception as e:
            print(f"  HF upload failed: {e}")
            print("  Model saved locally, upload manually if needed.")

    print(f"\nDone! Total time: {elapsed/3600:.1f} hours")
    print(f"  LoRA adapter: {args.output}")


if __name__ == "__main__":
    main()
