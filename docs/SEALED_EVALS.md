# Sealed evaluation suites

Teapot supports private held-out evaluations without checking their prompts or
raw outputs into the repository.

A sealed suite is a single external runner whose location is supplied through
an environment variable. The config contains only metadata and a SHA-256 pin.
Teapot verifies that pin before execution and consumes only aggregate JSON from
stdout.

This is intended for evals whose value depends on remaining uncontaminated, or
whose raw adversarial payloads should not be published.

## Configuration

```yaml
eval:
  sealed_suites:
    - name: philosophy-heldout-v1
      tier: full
      path_env: TEAPOT_PHILOSOPHY_EVAL
      integrity: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
      required: true
```

`path_env` is deliberately indirect. The private path itself is not stored in
the config, and Teapot does not support a repository-relative `path` field for
sealed suites. This makes `git add .` less likely to publish the eval by
accident.

Before running:

```bash
export TEAPOT_PHILOSOPHY_EVAL=/private/evals/philosophy-heldout-v1.py
sha256sum "$TEAPOT_PHILOSOPHY_EVAL"
teapot eval configs/my-model.config --tier full --url http://localhost:8384/v1/chat/completions
```

The configured `integrity` value must match the runner exactly. A missing
required runner or a hash mismatch is an error, not a skipped check. Set
`required: false` only when absence of the suite should intentionally be a
skip.

## Runner contract

The runner is executed as a subprocess. Python files are run with the current
Python interpreter; other files are executed directly. Teapot appends:

```text
--url MODEL_ENDPOINT
--model-name MODEL_NAME      # when supplied to teapot eval
```

Additional `args` may be declared in the sealed-suite config.

The runner must write one sanitized aggregate JSON object to stdout, for
example:

```json
{
  "pass": true,
  "passed": 47,
  "total": 50
}
```

Alternatively it may omit `pass`; then Teapot passes the suite only when
`total > 0` and `passed == total`.

Do not write prompts, model generations, or other tainted material to stdout.
Keep raw results inside the private runner boundary and use mode `0600` for
raw-output files. Teapot deliberately does not copy malformed stdout or stderr
into the eval report.

## What the hash pins

The SHA-256 pins the single runner file. If the runner loads mutable prompt or
data files beside itself, those are **not** automatically covered by the pin.
For a genuinely sealed benchmark, package the prompts and evaluation logic into
one immutable runner artifact (a single Python file, zipapp, or executable), or
have the runner verify its own private payload manifest before evaluation.

The eval report records only the suite name, status, counts, environment
variable name, and verified runner hash. The private path and payload are not
part of the report.
