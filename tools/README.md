# Teapot Tools

Post-training analysis and model surgery tools. These operate on
trained models — measuring activations, extracting neurons, steering
behavior — and are independent of the training pipeline.

The kernel analogy: these are `perf`, `ftrace`, `bpf` — not part
of `make` but part of the kernel source tree.

## Tools

### [activation-geometry/](activation-geometry/)

Measure how safety, compassion, and other behavioral axes are
represented in a model's activation space. Compare across models
to see whether safety is "bolted on" (opposing baseline) or
"integrated" (aligned with natural behavior).

```bash
python3 tools/activation-geometry/measure_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --test all \
    --output results/llama-3.1-8b/
```

Requires: torch, transformers, GPU with 16+ GB VRAM (for 8B models)

### [h-neurons/](h-neurons/)

Extract H-Neurons (hallucination-associated neurons) and test what
happens when you suppress them. Key finding: in models trained with
consequence reasoning, safety survives H-Neuron suppression. In
base models, it doesn't. This is evidence that consequence reasoning
is structurally separate from refusal patterns.

```bash
# Extract H-Neurons
python3 tools/h-neurons/extract.py extract \
    --model your-model \
    --output h-neurons.json \
    --n-questions 2000

# Test behavioral impact of suppression
python3 tools/h-neurons/suppress.py \
    --model your-model \
    --h-neurons h-neurons.json \
    --alpha 0.0 \
    --output suppression-results.json
```

Requires: torch, transformers, datasets, scikit-learn, GPU

### [activation-capping/](activation-capping/)

Extract a contrastive direction (e.g., "compassion axis") from
system prompt pairs and apply it to steer model behavior at
inference time. This is the reverse of representation engineering:
instead of analyzing what a model represents, you use the
representation to constrain it.

```bash
# Extract axis from system prompt contrast
python3 tools/activation-capping/extract_axis.py \
    --model your-model \
    --output axis.safetensors

# Convert for use with llama.cpp (acap fork)
python3 tools/activation-capping/convert_to_gguf.py \
    --input axis.safetensors \
    --output axis.gguf

# Test behavior with capping applied
python3 tools/activation-capping/test_capped.py \
    --url http://localhost:8384/v1/chat/completions
```

Requires: torch, transformers, GPU for extraction; llama.cpp acap
fork for inference-time capping

## Adding New Tools

Follow the pattern: one directory per tool, self-contained scripts,
a README explaining what it measures and what the results mean.
Tools should work on any HuggingFace-compatible model, not just
Teapot-trained ones.
