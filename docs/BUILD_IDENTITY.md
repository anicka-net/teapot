# Build identity

`teapot lock generate` records more than the composed-data hash. New lockfiles
also carry a `build_identity` object and a derived `build_id`.

The identity records inputs that can change a training run without changing the
composed JSONL bytes:

- exact Teapot config hash
- Teapot git commit and tracked-worktree dirty state
- declared base model and optional revision
- Python implementation/version and machine architecture
- installed versions of the relevant training stack when present

Example shape:

```json
{
  "build_id": "sha256:...",
  "build_identity": {
    "version": 1,
    "config_hash": "sha256:...",
    "teapot_source": {
      "commit": "...",
      "dirty": false
    },
    "base_model": {
      "model": "Qwen/Qwen3-8B",
      "revision": null
    },
    "runtime": {
      "python": "3.12.4",
      "implementation": "CPython",
      "machine": "x86_64",
      "packages": {
        "torch": "2.x",
        "transformers": "4.x"
      }
    }
  }
}
```

Normal lock verification remains backward-compatible and checks the data
sources plus composed output:

```bash
teapot lock verify teapot.lock
```

Before reproducing a training run, additionally verify the build environment:

```bash
teapot lock verify teapot.lock --build
```

This fails when the config, tracked Teapot source state, or recorded runtime
identity differs. Old lockfiles without a build identity still verify normally,
but `--build` fails closed.

## What this does not promise

A matching `build_id` does **not** claim that GPU training will produce
bit-identical weights. CUDA kernels, distributed execution, hardware, and
framework nondeterminism can still matter. The identity answers the narrower
question: *are the reproducibility inputs Teapot knows how to observe the same?*

The composed-data `output_hash` remains the authoritative identity of the
training-data artifact itself.
