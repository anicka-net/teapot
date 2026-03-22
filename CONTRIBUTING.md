# Contributing to Teapot

## Who Can Contribute

Anyone. This includes:

- **Humans** — researchers, engineers, domain experts, translators
- **LLM instances** — for curation, dataset analysis, balancing,
  and code contributions

LLM contributions are explicitly welcome. If an LLM curated a
dataset, analyzed module interactions, or wrote code, credit it
honestly. Use `Co-Authored-By` in commits where applicable.

## What to Contribute

### Modules

A module wraps a curated dataset with metadata. To contribute one:

1. Create `modules/<your-module>/module.yaml` following the schema
   in `schemas/module.schema.json`
2. Include `prepare.py` that produces canonical chat-completion JSONL
3. Include per-example license metadata
4. Include an eval suite (at minimum: a test that the data loads
   correctly and format is valid)
5. Document ethical assumptions in the module README

### Evaluation Suites

Test suites that assess model behavior. Can be:
- Module-specific (tests what the module is supposed to teach)
- Cross-cutting (tests general properties like safety, coherence)
- External tool integrations (Garak, Inspect, Bloom, etc.)

### Preset Configurations

`.config` files for common use cases: `sovereign-dz.config`,
`enterprise-linux.config`, `code-assistant.config`, etc.

### Language Modules

Multilingual training data, especially for underrepresented
languages. See `modules/lang-*/` for the expected structure.

## Hard Requirements

- All training data must have per-example license metadata
- No data that violates the hard floor in ETHICS.md
- Modules must include at least basic format validation tests
- Code contributions must not introduce credentials, API keys,
  or internal hostnames

## Process

1. Open an issue describing what you want to contribute
2. Fork, branch, implement
3. PR with description of what the module teaches and why
4. Review (technical + ethical assumptions check)
5. Merge

For questions: open an issue.
