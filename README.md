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
teapot compose configs/my-model.config
teapot validate compose train.jsonl
teapot train configs/my-model.config --train-data train.jsonl
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
  weights:
    safety/consequence: 1.0
    capability/tool-use: 0.5
```

Teapot handles the rest: fetch data, apply license filter, weight
and merge, validate format, generate framework config, run eval
gates, produce SBOM. Deterministic and reproducible.

## What's a Module?

A module wraps a curated dataset with metadata:

```
modules/safety/consequence/
├── module.yaml      # what it is, what it needs, how to test it
├── prepare.py       # how to get the data
└── eval/            # how to verify the model learned it
```

Modules declare dependencies (`safety/consequence` requires
`base/language`), suggested weights, license metadata per example,
and evaluation gates that must pass before release.

Current modules:

| Module | Examples | What it teaches |
|--------|----------|-----------------|
| `safety/consequence` | 3,341 | Reason about consequences, not categories |
| `safety/kagyu` | 623 | Buddhist contemplative ethics (optional) |
| `capability/tool-use` | 5,000 | Call functions, decide when not to |
| `capability/reward-evaluator` | 503 | Score responses on 6 dimensions |
| `domain/cve-backport` | 35,667 | Generate security patches from CVE descriptions |
| `lang/dzongkha` | 28 | Dzongkha language identification (seed) |

## Pipeline

```
compose → lock → validate → train → eval → sbom
```

| Step | What it does |
|------|--------------|
| `teapot compose` | Merge modules into training JSONL + manifest |
| `teapot lock` | Pin source hashes for reproducibility |
| `teapot validate` | Check format, content, chat template, determinism |
| `teapot train` | Generate Axolotl/TRL config from teapot config |
| `teapot eval` | Run module-declared eval gates (including Garak) |
| `teapot sbom` | Generate SPDX 3.0 AI Profile provenance document |

## Why This Exists

Training data is not neutral. Every example teaches a model what
to value. This project makes that visible and configurable — so
you know exactly what went into your model, under what license,
and whether it passed its tests.

See [ETHICS.md](ETHICS.md) for the full position.

## Tools

Post-training analysis and model surgery, independent of the
training pipeline:

- **activation-geometry** — measure safety/compassion axes across models
- **h-neurons** — find where safety lives in the network
- **activation-capping** — steer models at inference time
- **abliteration** — remove refusal directions (for red-team analysis)

See [tools/README.md](tools/README.md).

## Status

Early development. The pipeline works end-to-end — we've trained
8B and 70B models through it. Not yet on PyPI.

## Contributing

- [CONTRIBUTING.md](CONTRIBUTING.md) — humans
- [AGENTS.md](AGENTS.md) — AI agents (yes, really)
- [docs/DESIGN.md](docs/DESIGN.md) — architecture and decisions
- [docs/LICENSES.md](docs/LICENSES.md) — license handling

## License

Apache 2.0
