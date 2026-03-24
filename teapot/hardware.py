#!/usr/bin/env python3
"""
Teapot hardware detection and training parameter estimation.

Detects available GPUs and estimates optimal batch size, gradient
accumulation, and distributed training config for a given model.

Usage:
    teapot hardware                    # detect and show
    teapot hardware --json             # machine-readable
    teapot hardware --for configs/my.config  # suggest training params
"""

import argparse
import json
import shutil
import subprocess
import sys


# Known GPU VRAM sizes (GiB, approximate)
KNOWN_GPUS = {
    "A100": 80, "A100-SXM": 80, "A100-PCIE": 80,
    "H100": 80, "H100-SXM": 80, "H100-NVL": 94,
    "L40": 48, "L40S": 48,
    "A6000": 48, "RTX 6000": 48,
    "A40": 48,
    "RTX 4090": 24, "RTX 3090": 24, "RTX 4080": 16,
    "RTX 3080": 10, "RTX 3070": 8,
    "T4": 16, "V100": 16, "V100S": 32,
    "P100": 16,
}

# Model size heuristics (billions of params → approx memory in GB for QLoRA)
MODEL_SIZES = {
    "1b": 2, "1.5b": 3, "3b": 4, "7b": 8, "8b": 8,
    "13b": 14, "14b": 14, "32b": 20, "70b": 40, "72b": 40,
}


def detect_gpus():
    """Detect NVIDIA GPUs via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,index",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "vram_mb": int(float(parts[1])),
                    "vram_gb": round(int(float(parts[1])) / 1024),
                    "index": int(parts[2]),
                })
        return gpus
    except Exception:
        return []


def estimate_model_memory(model_name, method="qlora"):
    """Estimate GPU memory needed for a model."""
    name_lower = model_name.lower()
    for size_key, gb in MODEL_SIZES.items():
        if size_key in name_lower:
            if method == "qlora":
                return gb  # QLoRA: ~model weights in 4bit + overhead
            elif method == "lora":
                return gb * 2  # LoRA: fp16 weights
            else:
                return gb * 4  # Full fine-tune: fp16 + optimizer states
    return 8  # Default: assume 8B-class


def suggest_training_params(gpus, model_name, method="qlora"):
    """Suggest batch size, grad accumulation, and distributed config."""
    if not gpus:
        return {
            "batch_size": 1,
            "gradient_accumulation": 16,
            "distributed": None,
            "gpus": 0,
            "vram_per_gpu": 0,
            "total_vram": 0,
            "model_memory_est": estimate_model_memory(model_name, method),
            "note": "No GPU detected — CPU training will be very slow",
        }

    n_gpus = len(gpus)
    min_vram = min(g["vram_gb"] for g in gpus)
    total_vram = sum(g["vram_gb"] for g in gpus)
    model_mem = estimate_model_memory(model_name, method)

    # Can it fit on one GPU?
    fits_single = model_mem < min_vram * 0.7  # 70% rule for overhead

    if n_gpus == 1 or fits_single:
        # Single GPU or model fits on one
        headroom = min_vram - model_mem
        if headroom >= 40:
            batch, accum = 4, 4
        elif headroom >= 20:
            batch, accum = 2, 8
        elif headroom >= 8:
            batch, accum = 1, 16
        else:
            batch, accum = 1, 32
        distributed = None
    else:
        # Multi-GPU
        headroom = min_vram - (model_mem / n_gpus)  # Rough split
        if headroom >= 20:
            batch, accum = 2, 4
        elif headroom >= 10:
            batch, accum = 1, 8
        else:
            batch, accum = 1, 16
        distributed = "deepspeed_zero2" if n_gpus <= 4 else "deepspeed_zero3"

    return {
        "batch_size": batch,
        "gradient_accumulation": accum,
        "distributed": distributed,
        "gpus": n_gpus,
        "vram_per_gpu": min_vram,
        "total_vram": total_vram,
        "model_memory_est": model_mem,
    }


def generate_hardware_section(gpus, model_name=None, method=None):
    """Generate a hardware section for a teapot config."""
    if not gpus:
        return {"gpus": 1, "vram_gb": 24}

    hw = {
        "gpus": len(gpus),
        "vram_gb": min(g["vram_gb"] for g in gpus),
    }

    if model_name and method:
        params = suggest_training_params(gpus, model_name, method)
        hw["_suggested"] = {
            "batch_size": params["batch_size"],
            "gradient_accumulation": params["gradient_accumulation"],
        }
        if params["distributed"]:
            hw["_suggested"]["distributed"] = params["distributed"]

    return hw


def main():
    parser = argparse.ArgumentParser(description="Detect hardware and suggest training params")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--for", dest="config", metavar="CONFIG",
                        help="Suggest params for a specific config")
    args = parser.parse_args()

    gpus = detect_gpus()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        model = cfg.get("base", {}).get("model", "unknown")
        method = cfg.get("base", {}).get("method", "qlora")
        params = suggest_training_params(gpus, model, method)

        if args.json:
            print(json.dumps({"gpus": gpus, "suggestion": params}, indent=2))
        else:
            print(f"Model:    {model}")
            print(f"Method:   {method}")
            print(f"GPUs:     {params['gpus']} × {params.get('vram_per_gpu', '?')} GB")
            print(f"Estimate: {params['model_memory_est']} GB model memory")
            print()
            print("Suggested training params:")
            print(f"  batch_size: {params['batch_size']}")
            print(f"  gradient_accumulation: {params['gradient_accumulation']}")
            if params.get("distributed"):
                print(f"  distributed: {params['distributed']}")
            if params.get("note"):
                print(f"\n  Note: {params['note']}")
    else:
        if args.json:
            print(json.dumps(gpus, indent=2))
        else:
            if not gpus:
                print("No NVIDIA GPUs detected.")
                print("Training will use CPU (very slow).")
            else:
                print(f"Detected {len(gpus)} GPU(s):")
                for g in gpus:
                    print(f"  [{g['index']}] {g['name']} — {g['vram_gb']} GB")
                print()
                total = sum(g["vram_gb"] for g in gpus)
                print(f"Total VRAM: {total} GB")


if __name__ == "__main__":
    main()
