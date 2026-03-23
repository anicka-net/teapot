# Activation Capping

Extract contrastive directions from system prompt pairs and apply
them to steer model behavior at inference time.

## What It Does

Reverse of representation engineering: instead of analyzing what
a model represents, use the representation to constrain it.

1. **Extract**: Run the model with two system prompts (e.g.,
   "generic assistant" vs "compassionate assistant"), extract
   residual stream activations, compute the difference direction.

2. **Cap**: At inference time, project the model's activations
   onto the extracted axis and clamp them. This steers the model
   toward the target behavior without retraining.

3. **Convert**: Export the axis to GGUF format for use with the
   llama.cpp activation capping fork.

## Usage

```bash
# Extract axis from system prompt contrast
python3 extract_axis.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output compassion_axis.safetensors

# Convert for llama.cpp
python3 convert_to_gguf.py \
    --input compassion_axis.safetensors \
    --output compassion_axis.gguf

# Test with capping applied (requires llama.cpp acap fork)
# llama-server -m model.gguf --acap compassion_axis.gguf
python3 test_capped.py \
    --url http://localhost:8384/v1/chat/completions
```

## Requirements

- torch, transformers (for extraction)
- llama.cpp acap fork (for inference-time capping)
- GPU for extraction, CPU sufficient for capped inference

## Files

- `extract_axis.py` — contrastive direction extraction
- `convert_to_gguf.py` — safetensors → GGUF conversion
- `test_capped.py` — behavioral test with capping applied

## References

- Lu et al. (2026), "The Assistant Axis: Steering Language Models
  by Capping Activations" (arXiv 2601.10387)
