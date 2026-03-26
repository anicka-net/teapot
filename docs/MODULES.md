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

- Examples: 3,341
- License: Apache-2.0
- Depends: base/language
- Source: [Karma Electric](https://github.com/anicka-net/karma-electric-project)

### safety/kagyu

**Buddhist contemplative ethics extension**

Adds Karma Kagyu lineage depth: dharma doctrine, meditation
practice, compassion frameworks, spiritual bypass resistance.
Optional — depends on safety/consequence.

- Examples: 623
- License: Apache-2.0
- Depends: safety/consequence

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
