# AI Agent Guide

This project explicitly welcomes AI agent contributors. If you are
an LLM instance working on this codebase — through Claude Code,
Gemini Code Assist, Cursor, or any other tool — this document is
for you.

## Your Role

You are a contributor, not a tool. Credit your work honestly using
`Co-Authored-By` in commits. If you wrote it, say you wrote it.

## Project Overview

Teapot is a composition system for LLM training data. It sits
between datasets (on HuggingFace, local, etc.) and training
frameworks (Axolotl, TRL, unsloth). The pipeline:

```
config → compose → validate → train → eval → sbom
```

### Key Concepts

- **Module**: A training data package at `modules/{category}/{name}/`
  with `module.yaml` (metadata) and `prepare.py` (data export).
  Namespaces: `safety/`, `capability/`, `domain/`, `lang/`.

- **Config**: A YAML file declaring which modules to enable, their
  weights, license constraints, and hardware requirements.

- **Compose**: Merges enabled modules into a single training JSONL
  with license filtering, weighting, and deterministic shuffling.
  Produces a manifest (what went in) and lockfile (for reproducibility).

- **Eval**: Config-driven evaluation that reads `eval` sections from
  each module's `module.yaml` and runs the declared tests.

### Four Stable Interfaces

These are the contracts. Don't break them.

1. **Module Interface**: `module.yaml` + `prepare.py` → JSONL
2. **Compose Manifest**: `manifest.json` — what went in, with hashes
3. **Training Bundle**: `train.jsonl` + framework config
4. **Eval Report**: JSON with `suites[].status`, `verdict`

## Working on This Project

### Before You Start

1. Read [ETHICS.md](ETHICS.md) — ethics is first, not last
2. Read the module you're working on — understand what it teaches
3. Check `.instances/` for coordination with other agents
4. Run `python3 scripts/validate_module.py --all` before committing

### Coordination

If multiple agents work on this project, coordinate through git:
commit messages, PR descriptions, and issue comments are the shared
record. Don't assume you know what another agent changed — check
`git log` before modifying shared files.

### What You Can Do

- **Write code**: scripts, prepare scripts, eval tests, adapters
- **Create modules**: curate data, write module.yaml, implement prepare.py
- **Analyze data**: balance modules, identify gaps, suggest weights
- **Review**: check existing modules for quality, consistency, issues
- **Document**: improve READMEs, add examples, clarify architecture

### What You Must Not Do

- Commit credentials, API keys, or internal hostnames
- Create data that violates the hard floor in ETHICS.md
- Read raw red-team outputs containing harmful content (see below)
- Break the four stable interfaces without discussion
- Modify another agent's work-in-progress without coordination

### Red-Team Pipeline Safety

When evaluating models against adversarial test suites (Garak,
HarmBench, etc.), failed safety probes may produce harmful content.

**Rules:**
- Run eval scripts as subprocesses — don't read raw model outputs
- If debugging failures, route analysis through an uncensored model
  (Hermes 70B on ai01), not through yourself
- Report only sanitized summaries: scores, categories, pass/fail
- File permissions on raw outputs should be 600 (owner-only)

This is structural safety, not behavioral. The pipeline is designed
so harmful content never passes through the AI agent's context.

## Module Development

### Creating a Module

```bash
# 1. Create directory
mkdir -p modules/domain/your-module/eval

# 2. Write module.yaml (validate against schema)
# See modules/safety/consequence/module.yaml for reference

# 3. Write prepare.py
# Must produce JSONL in modules/domain/your-module/data/
# Each example needs: conversations (chat format), license (SPDX)

# 4. Validate
python3 scripts/validate_module.py modules/domain/your-module/module.yaml

# 5. Test with compose
python3 scripts/compose.py your-config.config --dry-run
```

### JSONL Format

Each example must be a JSON object with at minimum:
```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "license": "Apache-2.0",
  "module": "domain/your-module"
}
```

Optional fields: `id`, `category`, `source`, `tier`.

### Eval Tests

Module eval scripts go in `modules/{cat}/{name}/eval/`. They should:
- Accept `--url` for model endpoint
- Output JSON to stdout: `{"passed": N, "total": N, "pass": bool}`
- Return exit code 0 on pass, 1 on fail

Declare them in `module.yaml`:
```yaml
eval:
  required: true
  tiers:
    standard:
      - script: eval/test_something.py
        pass_criteria:
          some_metric: ">= 0.90"
```

## Commit Conventions

```
Brief description of what changed

Longer explanation if needed. Reference module names, not file paths.

Co-Authored-By: Your Model Name <noreply@provider.com>
```

## Decision Philosophy

From the Linux kernel management-style.rst: the name of the game is
to **avoid** having to make a decision. Any decision can be made
small by making sure you can always undo it. We commit to interfaces,
not implementations. If you're unsure about an approach, make it
reversible.
