#!/usr/bin/env python3
"""
Teapot Training Adapter — generate framework configs from Teapot configs.

Currently supports Axolotl. The adapter translates Teapot's config
format into the training framework's native format. Switching frameworks
means writing a new adapter (~130 lines), not redesigning the system.

Usage:
    python3 scripts/training_adapter.py configs/apertus-70b-secular.config \\
        --train-data train-apertus-70b-secular.jsonl \\
        --output axolotl-config.yaml

    python3 scripts/training_adapter.py configs/cve-backport.config \\
        --train-data train-cve-backport.jsonl \\
        --backend axolotl
"""

import argparse
import sys
from pathlib import Path

import yaml

from teapot.root import find_root
from teapot.templates import TEMPLATES
TEAPOT_ROOT = find_root()


def estimate_batch_config(vram_gb, n_gpus, model_size_hint):
    """Estimate batch size and gradient accumulation from hardware."""
    total_vram = vram_gb * n_gpus

    if "70b" in model_size_hint.lower() or "32b" in model_size_hint.lower():
        # Large model
        if total_vram >= 160:
            return 2, 4
        elif total_vram >= 80:
            return 1, 8
        else:
            return 1, 16
    else:
        # 8B or smaller
        if total_vram >= 80:
            return 4, 4
        elif total_vram >= 40:
            return 2, 8
        else:
            return 1, 16


def detect_or_default_hardware(hardware, model_name, method, default_vram, default_gpus):
    """Resolve hardware from config or runtime detection with a safe fallback."""
    if hardware and (hardware.get("vram_gb") or hardware.get("gpus")):
        return hardware.get("vram_gb", default_vram), hardware.get("gpus", default_gpus)

    try:
        from teapot.hardware import detect_gpus, suggest_training_params

        gpus = detect_gpus()
        if gpus:
            params = suggest_training_params(gpus, model_name, method)
            return params.get("vram_per_gpu", default_vram), params.get("gpus", default_gpus)
    except ImportError:
        pass
    return default_vram, default_gpus


def generate_axolotl(teapot_config, train_data, output):
    """Generate Axolotl YAML from Teapot config."""
    with open(teapot_config) as f:
        cfg = yaml.safe_load(f)

    base = cfg.get("base", {})
    training = cfg.get("training", {})
    hardware = cfg.get("hardware", {})

    model_name = base.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    method = base.get("method", "qlora")

    # Auto-detect hardware if not specified in config
    if hardware and (hardware.get("vram_gb") or hardware.get("gpus")):
        vram = hardware.get("vram_gb", 24)
        n_gpus = hardware.get("gpus", 1)
    else:
        vram, n_gpus = detect_or_default_hardware(hardware, model_name, method, 24, 1)
        print(f"Auto-detected: {n_gpus} GPU(s), {vram} GB each")

    micro_batch, grad_accum = estimate_batch_config(vram, n_gpus, model_name)

    # Override with explicit values from config if present
    micro_batch = training.get("batch_size", micro_batch)
    grad_accum = training.get("gradient_accumulation", grad_accum)

    chat_template = training.get("chat_template", "auto")
    if chat_template == "auto":
        chat_template = None  # Let Axolotl auto-detect
    elif chat_template in {"apertus", "apertus-think", "apertus-tools", "apertus-full"}:
        raise ValueError(
            f"Axolotl backend does not support Teapot-owned template '{chat_template}'. "
            "Use qlora-hf on compose-formatted output instead."
        )

    axolotl = {
        "base_model": model_name,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "load_in_8bit": False,
        "load_in_4bit": method == "qlora",
        "strict": False,

        # Dataset
        "datasets": [
            {
                "path": str(Path(train_data).resolve()),
                "type": "sharegpt",
                "conversation": "chatml" if not chat_template else chat_template,
            }
        ],
        "dataset_prepared_path": "last_run_prepared",

        # LoRA
        "adapter": "qlora" if method == "qlora" else ("lora" if method == "lora" else None),
        "lora_r": training.get("lora_r", 64),
        "lora_alpha": training.get("lora_alpha", 128),
        "lora_dropout": 0.05,
        "lora_target_linear": True,

        # Training
        "num_epochs": training.get("epochs", 3),
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "sequence_len": training.get("max_length", 4096),
        "sample_packing": True,
        "pad_to_sequence_len": True,

        # Optimization
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": training.get("warmup_ratio", 0.1),
        "gradient_checkpointing": True,
        "bf16": True,
        "tf32": True,
        "flash_attention": True,
        "max_grad_norm": 1.0,

        # Logging
        "logging_steps": 10,
        "save_strategy": "epoch",
        "output_dir": f"output-teapot-{Path(teapot_config).stem}",

        # Eval
        "val_set_size": 0,
    }

    if chat_template:
        axolotl["chat_template"] = chat_template

    # Quantization config for QLoRA
    if method == "qlora":
        quant = base.get("quantization", "nf4")
        axolotl["bnb_4bit_quant_type"] = quant
        axolotl["bnb_4bit_compute_dtype"] = "bfloat16"
        axolotl["bnb_4bit_use_double_quant"] = True

    # Multi-GPU
    if n_gpus > 1:
        axolotl["deepspeed"] = "deepspeed_configs/zero2.json"

    # Clean up None values
    axolotl = {k: v for k, v in axolotl.items() if v is not None}

    with open(output, "w") as f:
        yaml.dump(axolotl, f, default_flow_style=False, sort_keys=False)

    print(f"Generated Axolotl config: {output}")
    print(f"  Base model: {model_name}")
    print(f"  Method: {method}")
    print(f"  Epochs: {axolotl['num_epochs']}")
    print(f"  LR: {axolotl['learning_rate']}")
    print(f"  Batch: {micro_batch} × {grad_accum} = {micro_batch * grad_accum} effective")
    print(f"  Train data: {train_data}")


def generate_qlora_hf(teapot_config, train_data, eval_data, output):
    """Generate a shell script for QLoRA training with HuggingFace Trainer.

    This backend targets Teapot's stable HuggingFace Trainer contract
    for direct QLoRA/LoRA fine-tuning without Axolotl.
    """
    with open(teapot_config) as f:
        cfg = yaml.safe_load(f)

    base = cfg.get("base", {})
    training = cfg.get("training", {})
    hardware = cfg.get("hardware", {})

    model_name = base.get("model", "Qwen/Qwen2.5-Coder-32B-Instruct")
    method = base.get("method", "qlora")
    if hardware and (hardware.get("vram_gb") or hardware.get("gpus")):
        vram = hardware.get("vram_gb", 94)
        n_gpus = hardware.get("gpus", 2)
    else:
        vram, n_gpus = detect_or_default_hardware(hardware, model_name, method, 94, 2)
        print(f"Auto-detected: {n_gpus} GPU(s), {vram} GB each")

    micro_batch, grad_accum = estimate_batch_config(vram, n_gpus, model_name)
    micro_batch = training.get("batch_size", micro_batch)
    grad_accum = training.get("gradient_accumulation", grad_accum)

    epochs = training.get("epochs", 2)
    lr = float(training.get("learning_rate", 1e-4))
    lora_r = training.get("lora_r", 64)
    lora_alpha = training.get("lora_alpha", 128)
    max_length = training.get("max_length", 4096)
    warmup = training.get("warmup_ratio", 0.05)

    config_stem = Path(teapot_config).stem
    output_dir = f"output-teapot-{config_stem}"

    # Build the shell script
    lines = [
        "#!/bin/bash",
        f"# Generated by: teapot train {teapot_config} --backend qlora-hf",
        f"# Config: {teapot_config}",
        f"# Model: {model_name}",
        f"# Method: {method}",
        "",
        "set -euo pipefail",
        "",
    ]

    # Build train.py command
    cmd_parts = [
        "python3 -m teapot.train_qlora_hf",
        f"    --model {model_name}",
        f"    --data {train_data}",
    ]
    if eval_data:
        cmd_parts.append(f"    --eval {eval_data}")
    cmd_parts.extend([
        f"    --output {output_dir}",
        f"    --epochs {epochs}",
        f"    --batch-size {micro_batch}",
        f"    --grad-accum {grad_accum}",
        f"    --lr {lr}",
        f"    --max-length {max_length}",
        f"    --warmup-ratio {warmup}",
        f"    --lora-r {lora_r}",
        f"    --lora-alpha {lora_alpha}",
    ])

    if method == "qlora":
        cmd_parts.append("    --qlora")

    # Check for no-flash-attn hint
    if "no-flash-attn" in str(cfg) or "qwen" in model_name.lower():
        cmd_parts.append("    --no-flash-attn")

    cmd_parts.append(f"    --run-name teapot-{config_stem}")

    lines.append(" \\\n".join(cmd_parts))
    lines.append("")

    script = "\n".join(lines)

    with open(output, "w") as f:
        f.write(script)

    import os
    os.chmod(output, 0o755)

    print(f"Generated training script: {output}")
    print(f"  Base model: {model_name}")
    print(f"  Method: {method}")
    print(f"  Epochs: {epochs}")
    print(f"  LR: {lr}")
    print(f"  LoRA: r={lora_r}, α={lora_alpha}")
    print(f"  Batch: {micro_batch} × {grad_accum} = {micro_batch * grad_accum} effective")
    print(f"  Train data: {train_data}")
    if eval_data:
        print(f"  Eval data: {eval_data}")
    print(f"\nRun with: bash {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate training framework config")
    parser.add_argument("config", help="Teapot config file")
    parser.add_argument("--train-data", required=True, help="Composed JSONL path")
    parser.add_argument("--eval-data", default=None, help="Eval JSONL path (for qlora-hf)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output config/script file")
    parser.add_argument("--backend", choices=["axolotl", "qlora-hf"], default="axolotl",
                        help="Training backend (default: axolotl)")
    args = parser.parse_args()

    if not args.output:
        args.output = "axolotl-config.yaml" if args.backend == "axolotl" else "train-teapot.sh"

    if args.backend == "axolotl":
        generate_axolotl(args.config, args.train_data, args.output)
    elif args.backend == "qlora-hf":
        generate_qlora_hf(args.config, args.train_data, args.eval_data, args.output)


if __name__ == "__main__":
    main()
