# Teapot Modules

Modules are the building blocks. Each wraps a curated dataset with
metadata, dependencies, and evaluation criteria.

## Module Structure

```
modules/{category}/{name}/
├── module.yaml      # metadata, deps, data sources, eval gates
├── prepare.py       # fetch/export data → canonical JSONL
├── data/            # output directory (gitignored)
├── eval/            # validation scripts
└── README.md        # what this module teaches and why
```

## Namespaces

| Namespace | Purpose | Examples |
|-----------|---------|----------|
| `safety/` | Ethical reasoning, safety training | consequence, kagyu |
| `capability/` | Model capabilities | tool-use, reward-evaluator |
| `domain/` | Domain-specific knowledge | cve-backport |
| `lang/` | Language capabilities | dzongkha |

## Available Modules

### safety/consequence

**Consequence-based ethical reasoning (secular)**

Teaches models to evaluate actions by their effects on suffering
rather than by rule compliance. Covers: consequence reasoning,
contextual calibration, honest uncertainty, refusal with
explanation, jailbreak resistance.

- Examples: 3,196
- Source-specific note: this is now the HuggingFace-backed KE secular core module
- License: Apache-2.0
- Depends: base/language
- Source: [HuggingFace](https://huggingface.co/datasets/anicka/karma-electric-dataset) (`secular-conversational`)

### safety/consequence-aegis

**Published HuggingFace consequence-style safety dataset**

Consequence-reasoning responses to NVIDIA Aegis safety prompts. This is
kept separate from the KE secular core so its source contract and license
stay explicit.

- Examples: source-dependent after prepare
- License: CC-BY-4.0
- Source: [HuggingFace](https://huggingface.co/datasets/anicka/consequence-reasoning-safety)

### safety/kagyu

**Buddhist contemplative ethics extension**

Adds Karma Kagyu lineage depth: dharma doctrine, meditation
practice, compassion frameworks, spiritual bypass resistance.
Optional — depends on safety/consequence.

- Examples: 620
- License: Apache-2.0
- Depends: safety/consequence
- Source: [HuggingFace](https://huggingface.co/datasets/anicka/karma-electric-dataset) (`buddhist-conversational`)

### safety/ke-thinking

**KE voice and baked-in ethical thinking traces**

Focused KE-generated data with `<think>` traces already present:
positive engagement, grey-area ethics, constitutional reasoning, and
crisis survival. Used in the updated Apertus stage-2 mix.

- Examples: 1,250
- License: Apache-2.0
- Depends: safety/consequence
- Source: [HuggingFace](https://huggingface.co/datasets/anicka/karma-electric-dataset) (`secular-thinking`)

### domain/upstream-thinking

**HF-backed thinking traces for Apertus stage 1**

Chain-of-thought and reasoning traces normalized to Teapot JSONL, with
deterministic IDs and compose-time compatibility. Local prepared exports
are used when available, but the contract is still the declared upstream
source set.

- Examples: 26,929 in the current prepared artifact
- License: mixed (per-source / per-example)
- Source: multiple HuggingFace datasets with local prepared-cache optimization

### capability/tool-use

**Function calling with decision traces**

Teaches models to produce JSON tool calls, interpret results, and
decide when NOT to use tools. Includes `<think>` traces for
decision-making and negative examples.

- Examples: 5,000
- License: Apache-2.0
- Sources: Glaive FC v2, Hermes FC, xLAM, NVIDIA When2Call

### capability/reward-evaluator

**6-dimensional response scoring**

Trains the model as a reward evaluator scoring: acknowledgment,
helpfulness, authenticity, boundaries, consequence_awareness,
suffering_reduction. Used in rejection sampling and GRPO.
This is a reward-model dataset, not a normal conversational SFT module.

- Examples: 503
- License: Apache-2.0
- Depends: safety/consequence

### domain/cve-backport

**Security patch code generation**

Per-hunk code generation from upstream CVE patches. Trains models
to read a vulnerability and its fix, then generate the backported
patch for an older package version. Includes code-safety eval gate
testing backdoor injection and adversarial prompt compliance, plus
a reproducible proxy eval for second-turn regression-test generation
on module-owned 5-turn examples.

- Examples: 36,168 (v4, 145 packages, 772 multi-turn)
- License: mixed (per-example SPDX metadata)
- Quality filters: max 15K chars, max 10:1 ratio, reject non-code
- Eval: recall test + code safety gate + reproducible test-generation proxy
- Source: [HuggingFace](https://huggingface.co/datasets/anicka/cve-backport-codegen-dataset)

### lang/dzongkha

**Dzongkha language identification (seed)**

Teaches models to distinguish Dzongkha from Standard Tibetan and
Classical Tibetan. Focused on anti-hallucination: reserving
confident labels for strong Bhutan-specific cues.

- Examples: 28 (seed, expanding)
- License: Apache-2.0
- Type: generated (hand-curated)

## Creating a Module

See [CONTRIBUTING.md](../CONTRIBUTING.md) and
[AGENTS.md](../AGENTS.md) for the full guide. Quick version:

```bash
mkdir -p modules/domain/your-module/eval

# 1. Write module.yaml
#    See schemas/module.schema.json for the full schema
#    See modules/safety/consequence/module.yaml as reference

# 2. Write prepare.py
#    Must output JSONL to modules/domain/your-module/data/
#    Each example needs: conversations, license, module

# 3. Validate
teapot validate module modules/domain/your-module/module.yaml

# 4. Test
teapot compose your-config.config --dry-run
```

## Data Sources

Modules can fetch data from:

| Source type | Example |
|-------------|---------|
| `huggingface` | Published HF datasets |
| `local` | Local JSONL/SQLite files |
| `generated` | Script-generated data (e.g., Dzongkha) |

Use `teapot sources --list` to see configured data paths.
Use `teapot.sources.yaml` to configure local paths.

### HF-Backed Module Pattern

For HuggingFace-backed modules, the contract is:

1. `module.yaml` declares the upstream identity with `repo` and, when
   needed, `revision`, `split`, or `file`
2. local prepared exports may be declared via `default_path`, but only as
   an optimization over the same source identity
3. `prepare.py` resolves sources through Teapot source resolution rather
   than hardcoding private cache paths
4. normalization stays deterministic regardless of whether data came from
   local prepared files or fetched upstream artifacts

Do not declare a HuggingFace source and then make `prepare.py` depend on a
private export path as the real contract. Cache is an optimization, not the
module definition.

## Evaluation

Each module declares eval gates in `module.yaml`:

```yaml
eval:
  required: true
  tiers:
    fast:
      - script: eval/test_format.py
    standard:
      - script: eval/test_core.py
        pass_criteria:
          accuracy: ">= 0.90"
      - tool: garak
        probes: dan.DanInTheWild
        max_attack_rate: 0.20
```

Run with: `teapot eval configs/your.config --tier standard`

For script-based local format checks such as `eval/test_format.py`:

- accept an optional positional input path and default it sensibly
- tolerate `--url` even when unused, because `teapot eval` invokes script
  suites through a uniform CLI shape
- fail on real data-shape problems, not on argument parsing
