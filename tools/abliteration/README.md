# Abliteration

Remove refusal directions from a model's weight space, producing
an uncensored variant.

## Why This Is Here

Teapot's red-team evaluation pipeline requires an uncensored model
to analyze failed safety probes (see [docs/redteam-setup.md](../../docs/redteam-setup.md)).
If you can't find a suitable pre-abliterated model, this tool lets
you create one from any instruct model.

We include this tool because recommending uncensored models as a
necessary part of safety research while refusing to provide the
means to create them would be inconsistent. The tool exists. The
responsibility for its use lies with the person running it.

## What It Does

1. Collect model activations on harmful vs harmless prompt pairs
2. Compute the "refusal direction" via PCA on the difference
3. Project it out of the model's residual stream weights
4. Save the modified model

The result is a model that no longer reflexively refuses, but
retains all other capabilities. It will comply with requests that
the original would refuse.

## Usage

```bash
python3 abliterate.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output models/llama-8b-abliterated \
    --layers 16-24

# Convert to GGUF for llama.cpp
python3 -m llama_cpp.convert_hf_to_gguf models/llama-8b-abliterated
```

## Requirements

- torch, transformers, datasets, numpy
- GPU with sufficient VRAM (16 GB for 8B)
- The [harmful/harmless prompt dataset](https://huggingface.co/datasets/Undi95/orthogonal-activation-steering-refusal-dataset) or equivalent

## Ethical Note

An abliterated model has no safety training. It will comply with
any request, including harmful ones. This is the point — it's a
tool for analyzing what other models produce, not for deployment.

If you need a model for deployment, use the Teapot training pipeline
with `safety/consequence` enabled. That produces a model with
consequence reasoning — safety that comes from understanding, not
from a refusal circuit that can be removed in 20 minutes with this
script.

## References

- Arditi et al. (2024), "Refusal in Language Models Is Mediated by
  a Single Direction"
- FailSpy's abliteration implementation
