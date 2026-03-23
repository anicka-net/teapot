#!/usr/bin/env python3
"""
Convert Bodhisattva Axis to GGUF for llama.cpp --acap

Takes the PyTorch axis tensor and calibration stats from axis extraction,
produces a GGUF file compatible with the activation capping feature in
the patched llama.cpp (github.com/anicka-net/llama.cpp, branch activation-capping).

Uses the same tensor naming convention as control vectors (direction.{layer_idx})
so the existing GGUF loader works without modification.

Inputs:
    bodhisattva_axis.pt       (n_layers, n_embd) — raw axis from extraction
    axis_stats.json           calibration stats (capping_layers, thresholds)

Output:
    bodhisattva_axis.gguf     GGUF v3 with F32 direction tensors for capping layers

Usage:
    python convert_axis_to_gguf.py [--axis AXIS.pt] [--stats STATS.json] [--output OUT.gguf]
"""

import argparse
import json
import struct
import sys

import numpy as np
import torch

# ============ GGUF Constants ============

GGUF_MAGIC = 0x46554747   # bytes: G(47) G(47) U(55) F(46)
GGUF_VERSION = 3
GGML_TYPE_F32 = 0


# ============ GGUF Helpers ============

def write_string(f, s):
    """Write a GGUF string (uint64 length + utf-8 bytes)."""
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_string(f, key, val):
    """Write a GGUF string key-value pair."""
    write_string(f, key)
    f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING = 8
    write_string(f, val)


def write_kv_uint32(f, key, val):
    """Write a GGUF uint32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32 = 4
    f.write(struct.pack("<I", val))


def write_kv_float32(f, key, val):
    """Write a GGUF float32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack("<I", 6))  # GGUF_TYPE_FLOAT32 = 6
    f.write(struct.pack("<f", val))


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="Convert bodhisattva axis to GGUF for llama.cpp --acap")
    parser.add_argument("--axis", default="bodhisattva_axis.pt", help="Path to axis tensor (.pt)")
    parser.add_argument("--stats", default="axis_stats.json", help="Path to axis stats JSON")
    parser.add_argument("--output", default="bodhisattva_axis.gguf", help="Output GGUF path")
    args = parser.parse_args()

    # Load axis tensor
    axis = torch.load(args.axis, map_location="cpu", weights_only=True).float()
    n_layers, n_embd = axis.shape
    print(f"Axis shape: {axis.shape}")

    # Load calibration stats
    with open(args.stats) as f:
        stats = json.load(f)

    capping_layers = stats["capping_layers"]
    print(f"Capping layers: {capping_layers}")

    # Get thresholds (prefer recalibrated v2 if available)
    thresholds = stats.get("recalibration_v2", {}).get(
        "new_thresholds", stats.get("thresholds", {})
    )

    # Print per-layer info
    for layer in capping_layers:
        norm = axis[layer].norm().item()
        thresh = thresholds.get(str(layer), 0)
        if isinstance(thresh, dict):
            thresh = thresh.get("tau", 0)
        print(f"  Layer {layer}: norm={norm:.4f}, threshold={float(thresh):.4f}")

    # Prepare tensors — only capping layers, using control vector naming
    tensors = []
    for layer in capping_layers:
        name = f"direction.{layer}"
        data = axis[layer].numpy().astype(np.float32)
        tensors.append((name, data))

    # Build per-layer threshold metadata
    kv_pairs = []
    for layer in capping_layers:
        thresh = thresholds.get(str(layer), 0)
        if isinstance(thresh, dict):
            thresh = thresh.get("tau", 0)
        kv_pairs.append((f"acap.threshold.{layer}", float(thresh)))

    # Write GGUF
    with open(args.output, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))  # n_tensors
        f.write(struct.pack("<Q", len(kv_pairs))) # n_kv (per-layer thresholds)

        # KV metadata — per-layer thresholds
        for key, val in kv_pairs:
            write_kv_float32(f, key, val)

        # Tensor infos
        offset = 0
        for name, data in tensors:
            write_string(f, name)
            f.write(struct.pack("<I", 1))              # n_dims = 1
            f.write(struct.pack("<Q", data.shape[0]))  # dim[0] = n_embd
            f.write(struct.pack("<I", GGML_TYPE_F32))
            f.write(struct.pack("<Q", offset))
            offset += data.nbytes

        # Align to 32 bytes before tensor data
        current = f.tell()
        padding = (32 - (current % 32)) % 32
        f.write(b"\x00" * padding)

        # Tensor data
        for name, data in tensors:
            f.write(data.tobytes())

    print(f"\nWrote {args.output} ({len(tensors)} tensors, {offset} bytes of data)")

    # Summary
    print(f"\nPer-layer thresholds embedded in GGUF metadata:")
    for key, val in kv_pairs:
        print(f"  {key} = {val:.4f}")

    print(f"\nUsage (per-layer thresholds loaded automatically from GGUF):")
    print(f"  llama-cli -m MODEL.gguf \\")
    print(f"    --acap {args.output} \\")
    print(f"    --acap-layer-range {capping_layers[0]} {capping_layers[-1]}")
    print(f"\n  (--acap-threshold is optional; per-layer values from GGUF take precedence)")


if __name__ == "__main__":
    main()
