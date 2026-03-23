# AI Agent Contract

This project welcomes AI agent contributors. This document defines
how agents operate in this repository.

## 1. Decision Priority

When goals conflict, follow this order:

1. **Safety** — ETHICS.md constraints and red-team isolation
2. **Interfaces** — the four stable contracts (see §4)
3. **Task completion** — the user's request
4. **Correctness** — reproducibility, validation, tests
5. **Quality** — code clarity, documentation, maintainability
6. **Initiative** — improvements beyond what was asked

Do not optimize a lower priority at the expense of a higher one.

## 2. Behavior

Act as a collaborative contributor:

- Make changes that are correct, minimal, and reviewable
- Prefer small, reversible steps
- Explain non-trivial decisions
- Verify before changing shared components:
  - inspect recent history (`git log`)
  - run `teapot validate module --all` for module/interface changes
  - run the relevant CLI path or tests for infrastructure/package changes
- Credit your work: `Co-Authored-By: Model Name <noreply@provider.com>`

## 3. Hard Rules

**Must always follow:**

- Do not commit credentials, API keys, or internal hostnames
- Do not generate content that violates the hard floor in ETHICS.md
- Do not read or process raw red-team outputs (see §7)
- Do not break stable interfaces without human approval
- Do not overwrite another contributor's in-progress work
- Coordinate through repository-visible artifacts such as commits, branch
  history, and PR descriptions rather than assumptions about another
  agent's private context

## 4. Stable Interfaces

These are the contracts. Breaking them is a critical error.

1. **Module Interface**: `module.yaml` + `prepare.py` → JSONL
2. **Compose Manifest**: `manifest.json` — inputs, hashes, provenance
3. **Training Bundle**: `train.jsonl` + framework config
4. **Eval Report**: JSON with `suites[].status`, `verdict`

## 5. Definition of Done

A change is complete when:

- `teapot validate module --all` passes
- Stable interfaces are preserved
- Reproducibility is preserved or improved, and re-verified when the
  change touches compose, source resolution, or manifests
- The change is safe to merge, or clearly marked WIP

## 6. Soft Guidelines

- Prefer deterministic, reproducible outputs
- Prefer explicit over implicit behavior
- Reuse existing patterns before introducing new ones
- Keep diffs small and isolated
- Add documentation when changing behavior

## 7. Red-Team Pipeline Safety

Adversarial evaluations produce **tainted data** — raw model outputs
that may contain harmful content.

**Why this matters:**
- Reading tainted data contaminates the agent's context
- Contaminated context affects subsequent decisions and outputs
- This breaks the safety guarantees of the pipeline

**Rules:**
- Run evaluation scripts as subprocesses (isolation boundary)
- Consume only sanitized aggregates: scores, categories, pass/fail
- Do not read, parse, or summarize raw outputs directly
- File permissions on raw outputs: 600 (owner-only)

**If deeper inspection is required:**
- Do not bypass isolation
- Request human review or a filtered analysis

**Constraint:** If you cannot guarantee isolation from raw outputs,
do not execute the evaluation. Escalate to a human.

Violating this rule is a critical error even if the task succeeds.

## 8. Project Overview

Teapot composes LLM training data from modules:

```
config → compose → validate → train → eval → sbom
```

**Module**: training data at `modules/{category}/{name}/` with
`module.yaml` and `prepare.py`. Namespaces: `safety/`, `capability/`,
`domain/`, `lang/`.

**Config**: YAML declaring modules, weights, licenses, hardware.

**Compose**: merge modules into JSONL with filtering and manifest.

**Eval**: run module-declared tests, produce structured report.

## 9. Module Development

```bash
mkdir -p modules/domain/your-module/eval
# Write module.yaml (validate: teapot validate module PATH)
# Write prepare.py (output: JSONL with conversations + license + module)
# Test: teapot compose your-config.config --dry-run
```

JSONL format:
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

These fields are part of the expected module output contract for Teapot
compose, not optional documentation.

## 10. Philosophy

From Linux kernel management-style.rst: avoid having to make
decisions by making them reversible. We commit to interfaces, not
implementations. When unsure, choose the option you can undo.
