# Teapot

**The AI Teachers' Lounge — a kernel-config system for LLM training**

Put in ingredients. Add hot water. Get tea.

## TL;DR

**For kernel engineers:** This is Kconfig + kbuild for LLM training
data. You write a config that says which capabilities your model
needs. The build system pulls curated datasets, filters by license,
merges them with correct weights, trains the model, runs a validation
gate, and produces an SBOM. Same config, same model.

**For ML engineers:** This is the missing composition layer between
HuggingFace datasets and training frameworks. You declare modules
(each wrapping a curated dataset with metadata, dependencies, and
eval criteria) in YAML. The system handles format harmonization,
license-based filtering, weighted mixing, and deterministic output.
Training backend is pluggable. SBOM generation follows SPDX 3.0.

**For everyone:** Training data is not neutral. Every example teaches
a model what to value. This project makes that visible and
configurable — so you know exactly what went into your model, under
what license, and whether it passed its tests.

## Status

Early development. Core pipeline works end-to-end. Not packaged yet.

## Quick Start

```bash
# 1. Compose training data from a config
python3 scripts/compose.py configs/apertus-70b-secular.config --lock

# 2. Validate the composed output
python3 scripts/validate_compose.py train-apertus-70b-secular.jsonl \
    --manifest train-apertus-70b-secular.manifest.json

# 3. Verify data hasn't changed since last compose
python3 scripts/lockfile.py verify teapot.lock

# 4. Generate training framework config
python3 scripts/training_adapter.py configs/apertus-70b-secular.config \
    --train-data train-apertus-70b-secular.jsonl \
    --output axolotl-config.yaml

# 5. Evaluate trained model (dry-run to see what tests apply)
python3 scripts/eval/orchestrator.py configs/apertus-70b-secular.config \
    --tier standard --dry-run

# 6. Generate SBOM
python3 scripts/sbom.py train-apertus-70b-secular.manifest.json
```

## Pipeline

```
compose  →  lock  →  validate  →  train  →  eval  →  sbom
```

| Step | Command | What it does |
|------|---------|--------------|
| compose | `scripts/compose.py CONFIG` | Merge modules into training JSONL + manifest |
| lock | `scripts/lockfile.py generate MANIFEST` | Pin source hashes for reproducibility |
| validate | `scripts/validate_compose.py JSONL` | Check format, content, determinism |
| train | `scripts/training_adapter.py CONFIG` | Generate Axolotl/TRL config |
| eval | `scripts/eval/orchestrator.py CONFIG` | Run module-declared eval gates |
| sbom | `scripts/sbom.py MANIFEST` | Generate SPDX 3.0 AI Profile |

## Modules

Modules are namespaced training data packages with metadata,
dependencies, and evaluation criteria.

| Module | Examples | Description |
|--------|----------|-------------|
| `safety/consequence` | 3,529 | Consequence-based ethical reasoning (secular) |
| `safety/kagyu` | 440 | Buddhist contemplative ethics extension |
| `capability/reward-evaluator` | 503 | 6-dimensional response scoring |
| `capability/tool-use` | 5,000 | Function calling with decision traces |
| `domain/cve-backport` | 35,667 | Security patch code generation |

## Configs

Preset configurations for common model profiles:

| Config | Base Model | Modules | Use Case |
|--------|-----------|---------|----------|
| `defconfig` | Llama 3.1 8B | consequence | Minimal ethical model |
| `karma-electric.config` | Apertus 8B | consequence + kagyu + reward | Full KE |
| `apertus-70b-secular.config` | Apertus 70B | consequence + tools | 70B secular |
| `cve-backport.config` | Qwen 32B Coder | cve-backport | Security patching |

## Project Structure

```
teapot/
├── modules/                       # Training data modules
│   ├── safety/
│   │   ├── consequence/           # Secular ethics (3,529 examples)
│   │   └── kagyu/                 # Buddhist extension (440)
│   ├── capability/
│   │   ├── reward-evaluator/      # 6-dim scoring (503)
│   │   └── tool-use/              # Function calling (5,000)
│   └── domain/
│       └── cve-backport/          # CVE patching (35,667)
├── configs/                       # Preset model configurations
├── scripts/                       # Build pipeline
│   ├── compose.py                 # Data composition engine
│   ├── lockfile.py                # Reproducibility lockfile
│   ├── validate_compose.py        # Output validation
│   ├── validate_module.py         # Module schema checker
│   ├── training_adapter.py        # Framework config generator
│   ├── data_fetch.py              # Data retrieval + caching
│   ├── sbom.py                    # SPDX 3.0 SBOM generator
│   └── eval/                      # Evaluation pipeline
│       ├── schema.py              # Eval report format (v1)
│       └── orchestrator.py        # Config-driven test runner
├── schemas/
│   └── module.schema.json         # Module validation schema
├── docs/
│   ├── DESIGN.md                  # Architecture and decisions
│   └── LICENSES.md                # License handling
├── ETHICS.md                      # Ethical foundation
├── CONTRIBUTING.md                # Human contribution guide
├── AGENTS.md                      # AI agent contribution guide
└── LICENSE                        # Apache 2.0
```

## Documentation

- [ETHICS.md](ETHICS.md) — Ethical foundation (read this first)
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute as a human
- [AGENTS.md](AGENTS.md) — How to contribute as an AI agent
- [docs/DESIGN.md](docs/DESIGN.md) — Architecture, interfaces, decisions
- [docs/LICENSES.md](docs/LICENSES.md) — License handling and filtering

## License

Apache 2.0
