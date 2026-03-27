# Teapot Agent Execution Template

This template defines the expected execution pattern for AI contributors.
It is intended to be used with `spec/teapot-agent-contract.md`.

## Defaults

| Field | Value |
|---|---|
| Repository type | hand-maintained Python tooling + module tree |
| Source of truth for product behavior | repository code and docs |
| Source of truth for contributor behavior | `AGENTS.md` + agent contract |
| Delivery mode | direct repository edits |
| Change style | minimal, reversible, reviewable |
| Verification style | targeted command path plus regression tests when needed |

## Input Set

Before editing, read:

1. `ETHICS.md`
2. `AGENTS.md`
3. `spec/teapot-agent-contract.md`
4. this file
5. the files directly touched by the task

Then read only the additional files needed for the touched interface.

## Execution Order

### Phase 1: Inspect

- `git log --oneline -n 5`
- inspect the touched code path
- determine the affected stable interface

### Phase 2: Plan Verification

Select the narrowest verification that proves the change:

- module work -> `teapot validate module --all`
- compose work -> `teapot compose ... --dry-run` or compose tests
- training work -> backend-specific generation/tests
- eval work -> targeted eval/test path
- docs work -> consistency check against current behavior

### Phase 3: Edit

Edit only the files required by the task.
Do not smuggle unrelated cleanup into the same change.

### Phase 4: Verify

Run the exact commands chosen in Phase 2.
If a regression is found during verification, fix it before concluding.

### Phase 5: Report

Report:

- changed files
- command paths executed
- result
- remaining risk, if any

## Contract-Failure Conditions

The execution is considered incorrect if any of the following happens:

- a stable interface changes without acknowledgement
- verification is skipped for shared behavior
- generated artifacts become less reproducible
- raw red-team outputs are inspected
- unrelated repository work is overwritten

## Recommended Deliverables

For behavior changes:

- code change
- matching regression test when practical
- doc update when user-facing or contract-facing behavior changes

For review-only work:

- ordered findings
- exact file references
- note of what was verified
