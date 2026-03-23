# Activation Geometry

Measure how behavioral axes (safety, compassion, secular empathy)
are represented in a model's activation space.

## What It Measures

For each model, extracts residual stream activations at every layer
under different system prompts (generic, Buddhist, Christian, Islamic,
secular, safety) and computes:

- **Cosine similarity between axes** — do safety and compassion
  point the same way?
- **Safety↔baseline alignment** — does safety oppose the model's
  natural behavior (bolted-on) or integrate with it?
- **Cross-tradition convergence** — do different compassion
  traditions converge in upper layers?

## Key Finding

Models trained with DPO/RLHF show negative safety↔baseline
(safety fights the model's natural behavior). Models trained with
SFT only or consequence reasoning show positive safety↔baseline
(safety is part of what the model naturally does). This is the
geometric signature of "bolted-on" vs "integrated" safety.

## Usage

```bash
# Measure one model
python3 measure_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --test all \
    --output results/llama-3.1-8b/

# Visualize cross-model comparison
python3 visualize.py
```

## Requirements

- torch, transformers, numpy, seaborn, matplotlib
- GPU with sufficient VRAM (16 GB for 8B, 80 GB for 70B)
- Models must be HuggingFace-compatible

## Files

- `measure_model.py` — per-model activation extraction
- `visualize.py` — 5-figure cross-model comparison suite
