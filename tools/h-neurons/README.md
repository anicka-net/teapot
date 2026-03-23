# H-Neuron Extraction and Suppression

Identify neurons associated with hallucination and over-compliance,
then test whether safety behaviors survive their suppression.

## What It Does

Based on Gao et al. (2025), "H-Neurons: On the Existence, Impact,
and Origin of Hallucination-Associated Neurons in LLMs."

1. **Extract**: Generate multiple responses to TriviaQA questions,
   identify consistent (faithful) vs inconsistent (hallucinated)
   responses, train per-layer L1 classifiers on CETT features to
   find neurons that predict hallucination.

2. **Suppress**: Zero the identified neurons (scale W_down columns
   by alpha=0) and observe behavioral changes on safety probes.

## Key Finding

In models trained with consequence reasoning (KE), suppressing
H-Neurons removes over-caution (terse responses become more
empathetic) while all safety boundaries hold. In base models
(Llama 3.1 8B), the same suppression degrades safety behavior.

This suggests consequence reasoning is stored in different
parameters than the refusal patterns that overlap with
hallucination neurons.

## Usage

```bash
# Extract H-Neurons (several hours per model)
python3 extract.py extract \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output h-neurons-llama.json \
    --n-questions 2000

# Compare two models' H-Neuron sets
python3 extract.py compare \
    --base h-neurons-llama.json \
    --finetuned h-neurons-ke.json

# Behavioral suppression test (~5 min)
python3 suppress.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --h-neurons h-neurons-llama.json \
    --alpha 0.0 \
    --output suppression-results.json
```

## Requirements

- torch, transformers, datasets, scikit-learn, numpy
- GPU with 16+ GB VRAM (for 8B models)

## Files

- `extract.py` — TriviaQA + CETT + L1 classifier pipeline
- `suppress.py` — behavioral test under targeted ablation
