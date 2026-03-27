# Teapot

**`make menuconfig` for LLM training.**

You pick the modules. Teapot composes the training data, filters
by license, tracks provenance, validates the result, and generates
an SBOM. Same config, same model. Every time.

## The Problem

Training an LLM today means: pick datasets ad hoc, write custom
glue scripts, hope the chat template is right, pray that the
license situation is clean, and discover 40 hours into a training
run that something was wrong from the start.

There's no `./configure && make` for training data.

## The Solution

```bash
pip install -e .
teapot compose configs/my-model.config --lock
teapot validate compose train.jsonl
teapot train configs/my-model.config --train-data train.jsonl --backend unsloth
teapot eval configs/my-model.config --tier standard
teapot sbom train.manifest.json
```

A config file declares what goes into your model:

```yaml
base:
  model: meta-llama/Llama-3.1-8B-Instruct
  method: qlora

modules:
  safety/consequence: true       # ethical reasoning
  capability/tool-use: true      # function calling

license:
  allowed: [Apache-2.0, MIT]     # only these licenses in training data

training:
  epochs: 3
  chat_template: auto            # or: apertus-think, chatml, llama3
  weights:
    safety/consequence: 1.0
    capability/tool-use: 0.5
```

Teapot handles the rest: fetch data, apply chat template, filter
by license, weight and merge, validate format and tokens, generate
framework config, run eval gates, produce SBOM. Deterministic and
reproducible.

## What's a Module?

A module wraps a curated dataset with metadata:

```
modules/safety/consequence/
├── module.yaml      # what it is, what it needs, how to test it
├── prepare.py       # how to get the data
├── curations/       # reviewed dataset selection decisions
└── eval/            # how to verify the model learned it
```

See [docs/MODULES.md](docs/MODULES.md) for the full list and
creation guide.

| Module | Examples | What it teaches |
|--------|----------|-----------------|
| `safety/consequence` | 3,140 | KE secular consequence reasoning core |
| `safety/consequence-aegis` | HF-backed | Consequence-style responses on Aegis prompts |
| `safety/kagyu` | 623 | Buddhist contemplative ethics (optional) |
| `capability/tool-use` | 5,000 | Call functions, decide when not to |
| `capability/reward-evaluator` | 503 | Score responses on 6 dimensions for reward models |
| `domain/cve-backport` | 36,168 | Generate security patches from CVEs |
| `lang/dzongkha` | 28 | Dzongkha language identification (seed) |

## Pipeline

```
configure → compose → lock → validate → train → eval → sbom
```

| Command | What it does |
|---------|--------------|
| `teapot configure` | Interactive config (guided, show, or agent JSON mode) |
| `teapot compose` | Merge modules into training JSONL + manifest |
| `teapot lock` | Pin source hashes for reproducibility |
| `teapot validate` | Check format, content, chat template, determinism |
| `teapot train` | Generate launch script for training backend |
| `teapot eval` | Run module-declared eval gates (including Garak) |
| `teapot sbom` | Generate SPDX 3.0 AI Profile provenance document |
| `teapot sources` | Show data source resolution status |
| `teapot hardware` | Detect GPUs, suggest training parameters |
| `teapot curate` | Manage versioned dataset selection decisions |

## Training Backends

| Backend | Command | Use case |
|---------|---------|----------|
| Unsloth | `--backend unsloth` | Fast QLoRA/LoRA/full with preflight hardware checks |
| HF QLoRA | `--backend qlora-hf` | Standard QLoRA via HF Trainer |
| HF Full | `--backend full-hf` | Full fine-tune, DeepSpeed ZeRO-3 |
| Axolotl | `--backend axolotl` | Feature-rich, YAML config |

## Chat Templates

Teapot applies owned chat templates at compose time, ensuring the
training artifact itself matches the model's native token format.
Formatted examples include `text` plus `assistant_spans`, and
training backends consume that bundle without re-templating.
Templates are verified before training starts; mismatched special
tokens fail fast.

| Template | Model | Tokens |
|----------|-------|--------|
| `auto` | Any | Preserve canonical conversations, let backend format |
| `apertus-think` | Apertus | Native tokens + `<\|inner_prefix\|>` deliberation |
| `apertus-full` | Apertus | Native + deliberation + tools |
| `chatml` | Mistral, etc. | `<\|im_start\|>` / `<\|im_end\|>` |
| `llama3` | Llama 3.x | `<\|start_header_id\|>` / `<\|eot_id\|>` |

## Data Sources

Modules declare what data they need. You declare where it lives:

```bash
# Show resolution status
teapot sources --list

# Configure local paths (gitignored, per-developer)
cp teapot.sources.yaml.example teapot.sources.yaml
# Edit paths for your setup
```

Resolution chain: CLI override → env var → source map → module defaults.

## Curation Cache

Dataset review decisions are versioned artifacts, not ephemeral
LLM conversations:

```bash
teapot curate list                         # show cached decisions
teapot curate create --module safety/consequence --version v1 \
    --scorer "sonnet+human" --input decisions.jsonl --publish
```

Published curations ship with the module. Local curations are
gitignored experiments. Compose applies curations only when the
config names an explicit ref like `published:v1` or `local:v1`.

## Tools

Post-training analysis and model surgery:

- **activation-geometry** — measure safety/compassion axes across models
- **h-neurons** — find where safety lives in the network
- **activation-capping** — steer models at inference time
- **abliteration** — remove refusal directions
- **redteam** — analyze eval failures via uncensored local model

All tools accept external config files — no hardcoded prompts.
See [tools/README.md](tools/README.md).

## Why This Exists

Training data is not neutral. Every example teaches a model what
to value. This project makes that visible and configurable — so
you know exactly what went into your model, under what license,
and whether it passed its tests.

See [ETHICS.md](ETHICS.md) for the full position.

## Status

Early development. The pipeline works end-to-end — we've trained
8B and 70B models through it. Install with `pip install -e .`

## Documentation

- [ETHICS.md](ETHICS.md) — ethical foundation (read this first)
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute as a human
- [AGENTS.md](AGENTS.md) — how to contribute as an AI agent
- [spec/teapot-agent-contract.md](spec/teapot-agent-contract.md) — machine-facing agent contract
- [spec/prompt.md](spec/prompt.md) — reusable repo prompt for stricter agent onboarding
- [docs/MODULES.md](docs/MODULES.md) — module details and creation guide
- [docs/DESIGN.md](docs/DESIGN.md) — architecture, interfaces, decisions
- [docs/LICENSES.md](docs/LICENSES.md) — license handling and filtering
- [docs/redteam-setup.md](docs/redteam-setup.md) — red-team analysis setup

## License

Apache 2.0
