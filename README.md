# Teapot

**The AI Teachers' Lounge — a kernel-config system for LLM training**

Put in ingredients. Add hot water. Get tea.

## TL;DR

**For kernel engineers:** This is Kconfig + kbuild for LLM training
data. You write a config that says which capabilities your model
needs (`CONFIG_SAFETY=y`, `CONFIG_TOOLS=m`, `CONFIG_LANG_DZ=y`).
The build system pulls curated datasets, filters by license, merges
them with correct weights, trains the model, runs a validation gate,
and produces an SBOM. Same config, same model.

**For ML engineers:** This is the missing composition layer between
HuggingFace datasets and training frameworks. You declare modules
(each wrapping a curated dataset with metadata, dependencies, and
eval criteria) in YAML. The system handles format harmonization
(chat templates, role mapping), license-based filtering (every
example tagged), weighted mixing, deduplication, and deterministic
output. Training backend is pluggable (Axolotl, TRL, unsloth).
Evaluation runs tiered gates from fast CI checks to full red-team
sweeps. SBOM generation follows SPDX 3.0 AI Profile.

**For everyone:** Training data is not neutral. Every example teaches
a model what to value. This project makes that visible and
configurable — so you know exactly what went into your model, under
what license, and whether it passed its tests.

## Status

Early development. Not ready for use.

## Architecture

See [docs/DESIGN.md](docs/DESIGN.md) for the full architecture.

### Four Stable Interfaces

Everything else is a reversible implementation detail.

```
Module Interface    →  Compose Manifest  →  Training Bundle  →  Eval Report
module.yaml            manifest.json        train.jsonl         eval.json
+ prepare.py           (what went in)       + framework cfg     + pass/fail
→ canonical JSONL                           (for any backend)
```

### Project Structure

```
teapot/
├── docs/               # Design docs
├── modules/            # Training data modules
│   ├── safety-consequence/   # KE ethics (first module)
│   └── ...
├── configs/            # Preset configurations
│   ├── defconfig       # Sane defaults
│   └── ...
├── scripts/            # Build system
│   ├── compose.py      # Data composition engine
│   ├── eval/           # Evaluation pipeline
│   └── ...
├── schemas/            # JSON Schemas for validation
├── ETHICS.md           # Ethical foundation
├── CONTRIBUTING.md     # How to contribute (including as an LLM)
└── LICENSE             # Apache 2.0
```

## License

Apache 2.0
