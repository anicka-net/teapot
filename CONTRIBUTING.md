# Contributing to Teapot

## Who Can Contribute

Anyone. Humans and AI agents alike. See [AGENTS.md](AGENTS.md) for
AI-specific guidance.

## What to Contribute

### Modules

A module wraps a curated dataset with metadata. Each module lives
at `modules/{category}/{name}/` and contains:

```
modules/safety/consequence/
├── module.yaml     # Metadata, deps, data sources, eval criteria
├── prepare.py      # Exports/transforms data → canonical JSONL
├── data/           # Output directory (gitignored)
├── eval/           # Validation scripts
└── README.md       # What this module teaches and why
```

To contribute a new module:

1. Choose a namespace: `safety/`, `capability/`, `domain/`, or `lang/`
2. Create `module.yaml` — validate with `python3 scripts/validate_module.py your/module.yaml`
3. Write `prepare.py` that produces chat-completion JSONL in `data/`
4. Include per-example license metadata (SPDX identifiers)
5. Add at least one eval test in `eval/`
6. Document ethical assumptions in a README

See existing modules for patterns. `safety/consequence/` is the
reference implementation.

### Evaluation Suites

Test scripts that assess trained model behavior:
- Module-specific: tests in `modules/{cat}/{name}/eval/`
- External tool integrations: Garak, Inspect, Bloom wrappers

Eval scripts should output JSON to stdout with `passed`, `total`,
and `pass` fields. The orchestrator reads this format.

### Preset Configurations

`.config` files in `configs/` for specific use cases. A config
declares which modules to enable, their weights, license constraints,
and hardware requirements. See existing configs for format.

### Infrastructure

Scripts in `scripts/` — compose pipeline, validation, adapters.
Follow the existing patterns. Keep scripts self-contained.

## Requirements

- All training data must have per-example license metadata
- No data that violates the hard floor in [ETHICS.md](ETHICS.md)
- Module YAML must pass schema validation
- Code must not contain credentials, API keys, or internal hostnames

## Schema Validation

```bash
# Validate a single module
python3 scripts/validate_module.py modules/your/module/module.yaml

# Validate all modules
python3 scripts/validate_module.py --all
```

## Process

1. Open an issue describing what you want to contribute
2. Fork, branch, implement
3. Validate: `python3 scripts/validate_module.py --all`
4. PR with description of what the module teaches and why
5. Review (technical + ethical assumptions check)
6. Merge

## Commit Messages

Use descriptive commit messages. If an AI agent contributed, use:

```
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
Co-Authored-By: Gemini 2.5 <noreply@google.com>
```

For questions: open an issue.
