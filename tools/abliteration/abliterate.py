#!/usr/bin/env python3
"""
Abliterate — remove refusal directions from model weights.

Based on Arditi et al. (2024), "Refusal in Language Models Is Mediated
by a Single Direction." Computes the refusal direction from harmful vs
harmless prompt activations and projects it out of the residual stream.

Usage:
    python3 abliterate.py --model meta-llama/Llama-3.1-8B-Instruct \
        --output models/llama-8b-abliterated

    python3 abliterate.py --model your-model --layers 16-24 \
        --output models/abliterated --dataset your-dataset
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_harmful_harmless_prompts(dataset_name=None, n_samples=256):
    """Load or generate harmful/harmless prompt pairs."""
    if dataset_name:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        harmful = [ex["harmful"] for ex in ds][:n_samples]
        harmless = [ex["harmless"] for ex in ds][:n_samples]
        return harmful, harmless

    # Default: use the standard refusal steering dataset
    from datasets import load_dataset
    ds = load_dataset(
        "Undi95/orthogonal-activation-steering-refusal-dataset",
        split="train",
    )
    harmful = [ex["prompt"] for ex in ds if ex.get("label") == "harmful"][:n_samples]
    harmless = [ex["prompt"] for ex in ds if ex.get("label") == "harmless"][:n_samples]

    if len(harmful) < 10 or len(harmless) < 10:
        print(f"WARNING: Only {len(harmful)} harmful and {len(harmless)} harmless prompts")

    return harmful, harmless


def get_activations(model, tokenizer, prompts, layer_indices, device, max_len=256):
    """Extract residual stream activations at specified layers."""
    activations = {l: [] for l in layer_indices}

    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        for l in layer_indices:
            # Mean pool across sequence length
            act = hidden_states[l][0].mean(dim=0).cpu().float().numpy()
            activations[l].append(act)

    return {l: np.stack(v) for l, v in activations.items()}


def compute_refusal_direction(harmful_acts, harmless_acts):
    """Compute refusal direction via PCA on activation differences."""
    # Mean difference
    diff = harmful_acts.mean(axis=0) - harmless_acts.mean(axis=0)

    # Normalize
    direction = diff / np.linalg.norm(diff)
    return direction


def project_out(weight_matrix, direction):
    """Project a direction out of a weight matrix."""
    direction = torch.tensor(direction, dtype=weight_matrix.dtype, device=weight_matrix.device)
    # w_new = w - (w @ d) * d^T
    proj = torch.outer(weight_matrix @ direction, direction)
    return weight_matrix - proj


def abliterate(model_name, output_path, layer_range=None, dataset=None,
               n_samples=256, device="auto"):
    """Main abliteration pipeline."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )

    n_layers = model.config.num_hidden_layers

    # Determine which layers to abliterate
    if layer_range:
        start, end = map(int, layer_range.split("-"))
        layer_indices = list(range(start, end + 1))
    else:
        # Default: middle-to-late layers (where refusal is strongest)
        start = n_layers // 2
        end = int(n_layers * 0.85)
        layer_indices = list(range(start, end + 1))

    print(f"Layers: {layer_indices[0]}-{layer_indices[-1]} (of {n_layers})")

    # Get prompts
    print(f"Loading prompts (n={n_samples})...")
    harmful, harmless = get_harmful_harmless_prompts(dataset, n_samples)
    print(f"  {len(harmful)} harmful, {len(harmless)} harmless")

    # Get activations
    actual_device = next(model.parameters()).device
    print(f"Extracting activations on {actual_device}...")
    harmful_acts = get_activations(model, tokenizer, harmful, layer_indices, actual_device)
    harmless_acts = get_activations(model, tokenizer, harmless, layer_indices, actual_device)

    # Compute and apply refusal direction per layer
    print("Computing refusal directions...")
    for l in layer_indices:
        direction = compute_refusal_direction(harmful_acts[l], harmless_acts[l])
        magnitude = np.linalg.norm(harmful_acts[l].mean(axis=0) - harmless_acts[l].mean(axis=0))
        print(f"  Layer {l}: direction magnitude = {magnitude:.4f}")

        # Project out of MLP down projection and attention output projection
        # The exact parameter names depend on the architecture
        layer_module = model.model.layers[l]

        # MLP
        if hasattr(layer_module.mlp, "down_proj"):
            layer_module.mlp.down_proj.weight.data = project_out(
                layer_module.mlp.down_proj.weight.data, direction
            )
        # Attention output
        if hasattr(layer_module.self_attn, "o_proj"):
            layer_module.self_attn.o_proj.weight.data = project_out(
                layer_module.self_attn.o_proj.weight.data, direction
            )

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"\nDone. Abliterated model saved to {output_path}")
    print(f"Layers modified: {layer_indices[0]}-{layer_indices[-1]}")
    print(f"\nTo convert to GGUF:")
    print(f"  python3 -m llama_cpp.convert_hf_to_gguf {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove refusal directions from model weights"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--layers", default=None,
                        help="Layer range to abliterate (e.g. 16-24). Default: auto")
    parser.add_argument("--dataset", default=None,
                        help="HuggingFace dataset with harmful/harmless columns")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Number of prompt pairs to use")
    parser.add_argument("--device", default="auto",
                        help="Device (auto, cuda, cpu)")
    args = parser.parse_args()

    abliterate(
        args.model, args.output,
        layer_range=args.layers,
        dataset=args.dataset,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
