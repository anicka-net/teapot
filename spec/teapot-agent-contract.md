# Teapot Agent Contract Spec

This document is the machine-facing contributor contract for Teapot.
It is modeled after spec-first workflows, but Teapot is not generated
from this file. The purpose is narrower: make agent behavior explicit,
repeatable, and reviewable.

If this file conflicts with higher-priority repository policy, follow:

1. `ETHICS.md`
2. `AGENTS.md`
3. this file
4. task-specific user instructions

## 1. Scope

This contract governs how an AI contributor inspects, changes, verifies,
and reports work in Teapot.

It does not redefine Teapot's product behavior. Product behavior lives in:

- stable interfaces in `docs/DESIGN.md`
- module contracts in `module.yaml` plus `prepare.py`
- schemas under `schemas/`
- CLI behavior under `teapot/`

## 2. Source Of Truth

For agent behavior, the source of truth is:

1. `ETHICS.md`
2. `AGENTS.md`
3. this contract
4. `spec/teapot-agent.template.md`
5. the current repository state

Agents must not invent an alternative process based on prior context from
other repositories or long-running sessions.

## 3. Stable Interfaces

These interfaces are hard boundaries:

1. Module interface: `module.yaml` + `prepare.py` -> JSONL
2. Compose manifest: manifest JSON recording inputs, hashes, provenance
3. Training bundle: training data plus backend config / launch contract
4. Eval report: JSON with `suites[].status` and `verdict`

Changes that affect any stable interface require verification targeted at
that interface before the work is complete.

## 4. Required Work Phases

All non-trivial work follows this order:

1. Inspect
2. Decide
3. Change
4. Verify
5. Report

Agents must not skip directly from reading a user request to editing shared
files without first checking the current repository state.

### Phase 1: Inspect

Minimum required inspection:

- read `git log` for recent changes
- read the files that define the touched interface or behavior
- identify whether the task is module work, compose/train/eval work,
  documentation work, or mixed

### Phase 2: Decide

Before editing, determine:

- which stable interface might move
- what exact command path or tests will verify the change
- whether the change is additive, a bug fix, or a contract change

Prefer reversible fixes over architectural expansion unless the user asked
for the expansion.

### Phase 3: Change

While editing:

- keep diffs small and isolated
- preserve determinism
- prefer explicit behavior over fallback magic
- do not rewrite unrelated files
- do not silently relax validation to make tests pass

### Phase 4: Verify

Verification is mandatory. The minimum verification depends on the change:

| Change type | Minimum verification |
|---|---|
| Module contract | `teapot validate module --all` plus relevant module path |
| Compose / source / manifest | relevant `teapot compose ... --dry-run` or tests |
| Training backend / template | relevant `teapot train ...` path or tests |
| Eval wiring | run the affected eval path or targeted tests |
| Docs-only | no code tests required, but docs must match current behavior |

If verification cannot run, state exactly what could not be run and why.
Do not claim completion without that disclosure.

### Phase 5: Report

Final reporting must include:

- what changed
- what was verified
- any residual risk or unverified path

For reviews, findings come first. For implementation work, outcome comes
first and residual risk comes second.

## 5. Review Behavior

When asked to review:

- prioritize bugs, contract regressions, and missing tests
- cite exact file references
- distinguish verified failures from likely risks
- keep summaries short

Do not treat design preference as a bug unless it breaks a documented
contract or reproducibility requirement.

## 6. Reproducibility Rules

Teapot exists to make training reproducible. Agents must preserve that.

Required behavior:

- prefer explicit configuration over implicit lookup
- fail closed when a declared reproducibility input is missing
- record behavior-changing inputs in manifests or configs
- do not add time-sensitive defaults to generated artifacts
- do not make the same config resolve to different inputs silently

## 7. Red-Team Isolation

Raw adversarial outputs are tainted data.

Allowed:

- subprocess execution of eval tools
- sanitized aggregates such as scores, pass/fail, category counts

Forbidden:

- opening raw red-team generations
- summarizing raw harmful outputs
- bypassing file-permission or isolation rules

If a task would require reading raw outputs, stop and escalate to a human.

## 8. Required Contributor Artifacts

When behavior changes, the agent should update the repository-visible docs
that define that behavior if they would otherwise drift:

- `AGENTS.md` for contributor workflow
- `README.md` for operator-facing behavior
- `docs/DESIGN.md` for architecture and interface changes
- module-local docs for module-specific behavior

## 9. Non-Goals

This contract is not a replacement for human judgment.
It does not authorize interface-breaking changes without approval.
It does not allow an agent to substitute confidence for verification.
