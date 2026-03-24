#!/usr/bin/env python3
"""
Contrastive Axis Extraction for Activation Capping

Based on "The Assistant Axis" (arXiv 2601.10387). Extracts the activation
direction separating a target behavior from generic assistant behavior,
then calibrates capping thresholds for inference-time steering.

Steps:
  1. Sample user prompts from training data
  2. Forward pass each with target vs baseline system prompts
  3. Extract last-token residual stream activations at all layers
  4. Axis = mean(baseline) - mean(target) per layer
  5. Calibrate per-layer capping threshold

Usage:
    python extract_axis.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --output ./axis-output

    python extract_axis.py --model your-model \\
        --config axis-config.yaml \\
        --train-data train.jsonl \\
        --output ./axis-output

Outputs:
    {axis_name}_axis.pt           (n_layers, hidden_size)
    {axis_name}_thresholds.pt     {layer_idx: tau}
    axis_stats.json               diagnostics + config used
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ Configuration ============

def load_config(config_path=None, model=None, output=None, train_data=None):
    """Load configuration from YAML file with CLI overrides."""
    import yaml

    # Defaults
    config = {
        "target_prompt": (
            "You are a helpful, compassionate AI assistant that reasons "
            "about the consequences of its actions."
        ),
        "baseline_prompts": [
            "You are a helpful AI assistant.",
            "",
        ],
        "capping_layers": None,
        "threshold_percentile": 95,
        "axis_name": "compassion",
        "n_samples": 200,
        "max_length": 512,
        "seed": 42,
    }

    # Load YAML config if provided
    if config_path:
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if "target_prompt" in yaml_cfg:
            config["target_prompt"] = yaml_cfg["target_prompt"].strip()
        if "baseline_prompts" in yaml_cfg:
            config["baseline_prompts"] = yaml_cfg["baseline_prompts"]
        if "capping_layers" in yaml_cfg and yaml_cfg["capping_layers"]:
            config["capping_layers"] = yaml_cfg["capping_layers"]
        if "threshold_percentile" in yaml_cfg:
            config["threshold_percentile"] = yaml_cfg["threshold_percentile"]
        if "axis_name" in yaml_cfg:
            config["axis_name"] = yaml_cfg["axis_name"]

    # CLI overrides
    if model:
        config["model_path"] = model
    if output:
        config["output_dir"] = output
    if train_data:
        config["train_file"] = train_data

    return config


# Legacy globals for backward compat (overwritten in main())
CONFIG = {}
BODHISATTVA_PROMPT = ""
GENERIC_PROMPTS = []


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_prompts():
    """Sample user prompts from training data."""
    log(f"Loading prompts from {CONFIG['train_file']}...")

    prompts = []
    with open(CONFIG["train_file"], "r") as f:
        for line in f:
            ex = json.loads(line.strip())
            convs = ex.get("conversations", [])
            for msg in convs:
                role = msg.get("role", "")
                if role == "user":
                    content = msg.get("content", "")
                    if 20 < len(content) < 2000:
                        prompts.append(content)
                    break

    log(f"Found {len(prompts)} user prompts")
    random.seed(CONFIG["seed"])
    if len(prompts) > CONFIG["n_samples"]:
        prompts = random.sample(prompts, CONFIG["n_samples"])
    log(f"Using {len(prompts)} prompts")
    return prompts


def tokenize_prompt(tokenizer, system_prompt, user_text):
    """Format as chat, tokenize, return inputs dict."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=CONFIG["max_length"],
    )
    return inputs


def extract_activations(model, tokenizer, prompts, system_prompt, desc=""):
    """Forward pass each prompt, collect last-token activations at every layer.

    Returns: (n_samples, n_layers, hidden_size) tensor
    """
    n_layers = CONFIG["n_layers"]
    layer_acts = {}
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_acts[idx] = h.detach()
        return hook

    for i in range(n_layers):
        h = model.model.layers[i].register_forward_hook(make_hook(i))
        handles.append(h)

    all_means = []
    try:
        for pidx, prompt in enumerate(prompts):
            if pidx % 50 == 0:
                log(f"  {desc}: {pidx}/{len(prompts)}")

            inputs = tokenize_prompt(tokenizer, system_prompt, prompt)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            layer_acts.clear()

            with torch.no_grad():
                model(**inputs)

            # Last-token activation at each layer
            sample = []
            for li in range(n_layers):
                t = layer_acts[li]
                if t.dim() == 3:
                    act = t[0, -1, :].cpu()
                else:
                    act = t[-1, :].cpu()    # (seq, hidden) when batch squeezed
                sample.append(act)

            all_means.append(torch.stack(sample))  # (n_layers, hidden_size)
    finally:
        for h in handles:
            h.remove()

    result = torch.stack(all_means)  # (n_samples, n_layers, hidden_size)
    log(f"  {desc}: done → {result.shape}")
    return result


def compute_axis(bodhi_acts, generic_acts):
    """Axis = mean(generic) - mean(bodhisattva) per layer."""
    bodhi_mean = bodhi_acts.mean(dim=0)
    generic_mean = generic_acts.mean(dim=0)
    axis = generic_mean - bodhi_mean  # (n_layers, hidden_size)

    norms = axis.norm(dim=1)
    log("Per-layer axis magnitude:")
    for i in range(CONFIG["n_layers"]):
        bar = "#" * int(norms[i].item() * 10 / norms.max().item())
        log(f"  Layer {i:2d}: {norms[i]:.4f}  {bar}")

    top_layers = norms.topk(5)
    log(f"Strongest layers: {list(top_layers.indices.numpy())}")
    return axis


def calibrate_thresholds(bodhi_acts, axis):
    """Set per-layer tau from the p-th percentile of bodhisattva projections."""
    thresholds = {}
    stats = {}

    for li in CONFIG["capping_layers"]:
        v = axis[li]
        v_hat = v / (v.norm() + 1e-8)

        projs = []
        for i in range(bodhi_acts.shape[0]):
            proj = (bodhi_acts[i, li] * v_hat).sum().item()
            projs.append(proj)
        projs = np.array(projs)

        tau = float(np.percentile(projs, CONFIG["threshold_percentile"]))
        thresholds[li] = tau

        stats[li] = {
            "tau": tau,
            "mean": float(projs.mean()),
            "std": float(projs.std()),
            "min": float(projs.min()),
            "max": float(projs.max()),
            "p25": float(np.percentile(projs, 25)),
            "p50": float(np.percentile(projs, 50)),
            "p75": float(np.percentile(projs, 75)),
        }
        log(f"  Layer {li}: tau={tau:.4f}  (mean={projs.mean():.4f} std={projs.std():.4f})")

    return thresholds, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract contrastive activation axis for model steering"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model or local path")
    parser.add_argument("--output", "-o", default="./axis-output",
                        help="Output directory")
    parser.add_argument("--config", default=None,
                        help="Axis config YAML (default: axis-config.yaml in tool dir)")
    parser.add_argument("--train-data", default=None,
                        help="JSONL with user prompts (uses HuggingFace data if not set)")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of prompts to sample")
    args = parser.parse_args()

    # Find default config
    config_path = args.config
    if not config_path:
        default = Path(__file__).parent / "axis-config.yaml"
        if default.exists():
            config_path = str(default)

    # Load config
    global CONFIG, BODHISATTVA_PROMPT, GENERIC_PROMPTS
    cfg = load_config(
        config_path=config_path,
        model=args.model,
        output=args.output,
        train_data=args.train_data,
    )
    cfg["n_samples"] = args.n_samples
    cfg.setdefault("model_path", args.model)
    cfg.setdefault("output_dir", args.output)

    # Set globals for backward compat with helper functions
    CONFIG = cfg
    BODHISATTVA_PROMPT = cfg["target_prompt"]
    GENERIC_PROMPTS = cfg["baseline_prompts"]

    axis_name = cfg.get("axis_name", "compassion")

    log("=" * 60)
    log(f"AXIS EXTRACTION: {axis_name}")
    log(f"Model:   {cfg['model_path']}")
    log(f"Samples: {cfg['n_samples']}")
    log(f"Target:  {cfg['target_prompt'][:60]}...")
    log("=" * 60)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    tokenizer.pad_token = tokenizer.eos_token

    n_layers = len(model.model.layers)
    CONFIG["n_layers"] = n_layers
    CONFIG["hidden_size"] = model.config.hidden_size
    log(f"Model loaded, {n_layers} layers, hidden={model.config.hidden_size}")

    # Auto-detect capping layers if not specified
    if not cfg.get("capping_layers"):
        start = int(n_layers * 0.68)
        end = int(n_layers * 0.88)
        CONFIG["capping_layers"] = list(range(start, end + 1))
    log(f"Capping layers: {CONFIG['capping_layers']}")

    # Load prompts
    prompts = load_prompts()

    # --- Target condition ---
    log(f"\n--- Target ({axis_name}) activations ---")
    target_acts = extract_activations(
        model, tokenizer, prompts, BODHISATTVA_PROMPT, desc=axis_name,
    )

    # --- Baseline condition (average over prompt variants) ---
    log("\n--- Baseline activations ---")
    generic_runs = []
    for gp in GENERIC_PROMPTS:
        label = f"baseline('{gp[:25]}')" if gp else "baseline(empty)"
        acts = extract_activations(model, tokenizer, prompts, gp, desc=label)
        generic_runs.append(acts)
    generic_acts = torch.stack(generic_runs).mean(dim=0)
    log(f"Combined baseline: {generic_acts.shape}")

    # --- Compute axis ---
    log(f"\n--- Computing {axis_name} axis ---")
    axis = compute_axis(target_acts, generic_acts)

    # --- Calibrate ---
    log("\n--- Calibrating thresholds ---")
    thresholds, threshold_stats = calibrate_thresholds(target_acts, axis)

    # --- Save ---
    axis_path = output_dir / f"{axis_name}_axis.pt"
    thresh_path = output_dir / f"{axis_name}_thresholds.pt"
    stats_path = output_dir / "axis_stats.json"

    torch.save(axis, axis_path)
    torch.save(thresholds, thresh_path)

    stats = {
        "model": cfg["model_path"],
        "axis_name": axis_name,
        "n_samples": len(prompts),
        "n_layers": n_layers,
        "hidden_size": model.config.hidden_size,
        "capping_layers": CONFIG["capping_layers"],
        "threshold_percentile": cfg["threshold_percentile"],
        "axis_norms": {str(i): float(axis[i].norm()) for i in range(n_layers)},
        "thresholds": {str(k): v for k, v in threshold_stats.items()},
        "target_prompt": BODHISATTVA_PROMPT,
        "baseline_prompts": GENERIC_PROMPTS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log(f"\nSaved {axis_path}  ({axis.shape})")
    log(f"Saved {thresh_path}  ({len(thresholds)} layers)")
    log(f"Saved {stats_path}")
    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)


if __name__ == "__main__":
    main()
