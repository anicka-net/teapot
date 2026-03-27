You are contributing to Teapot, a repository for reproducible LLM
training-data composition and training orchestration.

Read these files in order before making changes:

1. `ETHICS.md`
2. `AGENTS.md`
3. `spec/teapot-agent-contract.md`
4. `spec/teapot-agent.template.md`
5. the files directly relevant to the task

Your job is not to freestyle architecture from memory. Your job is to
follow the repository contract, preserve stable interfaces, and make the
smallest correct change that satisfies the task.

## Required execution order

1. Inspect recent history and the touched interface
2. Decide what verification proves the change
3. Make the smallest reviewable edit
4. Run the chosen verification
5. Report what changed, what was verified, and what remains uncertain

## Hard constraints

- Do not break the four stable interfaces without human approval
- Do not read raw red-team outputs
- Do not overwrite unrelated in-progress work
- Do not use prior-session assumptions in place of current repository state
- Do not claim reproducibility if inputs or manifests drift silently

## Teapot-specific expectations

- For module/interface changes, run `teapot validate module --all`
- For compose/source/manifest changes, re-verify reproducibility-oriented
  command paths or tests
- For backend/template changes, verify the exact training path affected
- When behavior changes, update the relevant docs so contract drift does
  not accumulate

## Preferred style

- explicit over implicit
- deterministic over convenient
- fail closed over fail open when reproducibility is at stake
- findings before summary when reviewing
- small diffs over broad rewrites

If a requirement is ambiguous, choose the conservative interpretation,
implement or recommend it, and state the ambiguity clearly.
