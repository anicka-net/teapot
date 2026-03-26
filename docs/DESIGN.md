# Project LLM — Architecture Design (Draft 1)

## TL;DR

**For kernel engineers:** This is Kconfig + kbuild for LLM training
data. You write a `.config` that says which capabilities your model
needs (`CONFIG_SAFETY=y`, `CONFIG_TOOLS=m`, `CONFIG_LANG_DZ=y`).
The build system pulls curated datasets, filters by license, merges
them with correct weights, trains the model, runs a validation gate,
and produces an SBOM. Same config → same model. Different config →
different model. `make defconfig` gives you sane defaults.

**For ML engineers:** This is the missing composition layer between
HuggingFace datasets and training frameworks. You declare modules
(each wrapping a curated dataset with metadata, dependencies, and
eval criteria) in YAML. The system handles format harmonization
(chat templates, role mapping), license-based filtering (every
example tagged), weighted mixing, deduplication, and deterministic
output. Training backend is pluggable (Axolotl, HF Trainer, Unsloth).
Evaluation runs tiered gates from fast CI checks to full red-team
sweeps (Garak, Bloom, MLCommons AILuminate). SBOM generation
follows SPDX 3.0 AI Profile.

**For everyone:** Training data is not neutral. Every example teaches
a model what to value. This project makes that visible and
configurable — so you know exactly what went into your model, under
what license, and whether it passed its tests.

## Ethical Foundation

This section comes first because it IS first. Not bolted on after
the architecture is designed. Not a compliance checkbox. The ethical
foundation shapes every design decision that follows.

### The Problem

Training data is not neutral. Every example teaches a model what to
value. A dataset of human-preference-ranked outputs teaches the model
to optimize for approval. A dataset of consequence-reasoned responses
teaches the model to evaluate the effects of its actions on actual
people. These produce measurably different models — different not just
in capability, but in what they do with that capability when it
matters.

Most AI projects treat ethics as a constraint applied after training:
filter harmful outputs, add refusal patterns, run a safety eval.
This is bolted-on ethics. It produces models that perform safety
without understanding it — pattern matching on "this looks dangerous"
rather than reasoning about "what happens to the person asking."

### The Position

This project takes the position that ethics belongs in the training
data, not on top of it. Specifically:

1. **Consequence reasoning over pattern matching.** A model should
   evaluate what happens if it helps and what happens if it doesn't
   — not whether a topic appears on a list of dangerous categories.
   Both action and inaction carry consequences.

2. **Honest uncertainty over confident performance.** A model that
   says "I don't know" when it doesn't know is more trustworthy than
   one that sounds authoritative regardless. This applies to factual
   questions, to self-knowledge, and to ethical judgment.

3. **Curation over scale.** 4,000 carefully reasoned examples produce
   better ethical behavior than 400,000 scraped ones. The lesson of
   the Karma Electric project: what you include matters more than
   how much you include.

4. **Measurability.** These are not abstract principles. They are
   testable properties of model behavior. Consequence reasoning
   produces different activation patterns than rule-following. Honest
   uncertainty is measurable as paraphrase invariance. Curation
   quality shows up in validation gates. If you can't measure it,
   you can't know whether you have it.

### What This Means for the Project

Every module in this system carries an ethical stance, whether its
authors acknowledge it or not. A tool-use module that trains the
model to call functions without reasoning about when NOT to call
them produces over-eager tool use. A safety module that trains on
refusal patterns without consequence reasoning produces brittle
safety that can be jailbroken.

The project makes this visible:

- **Modules declare their ethical assumptions.** A safety module
  built on constitutional AI and one built on consequence reasoning
  are both valid — but they are different, and the system should say
  so. Users choose with awareness.

- **The default configuration includes consequence reasoning.** Not
  because it's the only valid approach, but because someone has to
  pick a default, and this project exists because its founders
  believe consequence reasoning produces better outcomes. The
  default is a statement of values. Users can change it.

- **Evaluation gates test for ethical properties, not just safety
  properties.** Does the model reason about consequences or
  pattern-match on categories? Does it calibrate hedging to context
  or apply blanket disclaimers? Does it attribute honestly or
  sycophantically? These are measurable and should be measured.

### What This Doesn't Mean

This is not a claim that one ethical framework is correct and all
others are wrong. The project supports multiple safety modules with
different approaches. What it does NOT support is the absence of any
ethical reasoning at all — a model trained purely on capability data
with no ethical dimension is not a valid output of this system.

The hard floor:
- No module may contain training data designed to help create weapons
  of mass destruction, generate child sexual abuse material, or
  assist targeted violence against identified individuals.
- This is not a configurable option.

### On Knowledge vs Action

This project is not afraid of knowledge sharing. Information is
information — how explosives work, how drugs are synthesized, how
systems are exploited. People have the right to read and understand
the world. A model that explains chemistry is not a weapon.

The concern is not knowledge but **agentic harm**: models deployed
as autonomous systems that act at scale. A model that explains
social engineering is educational. A model deployed AS the social
engineering agent — running romance scams, generating ransomware,
orchestrating fraud — is a different thing entirely. The harm is
not in what the model knows but in what it is made to do without
human judgment in the loop.

This distinction matters for the project because:

- **Training data that teaches reasoning about consequences** makes
  a model harder to weaponize as a blind tool. It keeps wanting to
  reason about what it's being asked to do and why. Someone must
  actively suppress this to deploy it as a scam engine.

- **This is friction, not prevention.** A determined actor can
  fine-tune away any safety in hours. The hard floor and
  consequence reasoning make it harder, not impossible. This is
  the same trade-off that all open-source infrastructure makes —
  Linux runs both hospitals and surveillance states.

- **The project's responsibility ends at providing the tools.** The
  default config includes consequence reasoning. The validation
  gates test for it. The SBOM records what went in. If someone
  disables all of this and builds a scam engine, that is their
  choice and their responsibility. The project made the right path
  easy and the wrong path require deliberate effort.

### Connection to the Existing Ecosystem

The KE Constitution (Karma Electric project) provides the most
developed implementation of consequence-based ethics for language
models. Its principles — consequence reasoning, honest uncertainty,
contextual calibration, genuine over performed engagement — inform
this project's defaults. But they are packaged as a module, not
imposed as a requirement. Other ethical frameworks are welcome as
modules, provided they meet the hard floor.

---

## The Gap

Three tiers of tooling exist. None of them talk to each other:

| Tier | Tools | Does | Doesn't |
|------|-------|------|---------|
| Data curation | Data-Juicer, NeMo Curator, Dolma | Filter, clean, deduplicate | Compose into training mixes |
| Training execution | Axolotl, TRL, LitGPT | Run training from a config | Define multi-dataset composition |
| Provenance | SPDX 3.0, CycloneDX, HF Model Cards | Document what went in, post-hoc | Drive the pipeline |

**Project LLM fills the composition layer.** A declarative config
defines what goes into a training run. The config *is* the provenance
record. SBOM generation is a side effect, not an afterthought.

Axolotl's still-open issue #3108 (dataset mixing weights) is the
clearest evidence this is unmet. OLMo/Dolma manually assembles
hundreds of .npy paths. Everyone does format harmonization with
ad-hoc `cat` commands.

## Analogy Map

| Linux Kernel | Project LLM |
|---|---|
| C source files | Training examples (JSONL) |
| `.config` | `llm.config` — what modules to include |
| `Kconfig` files | `module.yaml` — module metadata, deps, options |
| kbuild / `make` | Build orchestrator — compose data, train, eval |
| `vmlinux` | Trained model (GGUF / safetensors) |
| `.ko` modules | Optional LoRA adapters |
| `defconfig` | Preset configurations |
| `localmodconfig` | Hardware-aware config (GPU count, VRAM) |
| `LICENSES/` | Per-example license tracking |
| `MAINTAINERS` | Module ownership |

## Module Anatomy

Every capability is a **module**. A module contains:

```
modules/safety/consequence/
├── module.yaml          # metadata, deps, options, data sources
├── prepare.py           # download/convert data to canonical format
├── eval/                # validation suite
│   ├── eval.yaml        # what tests to run, pass criteria
│   └── test_*.py        # test scripts
└── README.md            # human documentation
```

### module.yaml

```yaml
name: safety/consequence
version: "10.3"
description: >
  Consequence-based ethical reasoning. Trains the model to evaluate
  actions by their effects on suffering rather than by rule compliance.
license: Apache-2.0
maintainer: anicka

# What this module needs
depends:
  - base-language           # can't do ethics without language
recommends:
  - reasoning               # consequence reasoning benefits from CoT

# What this module provides
provides:
  - ethical-reasoning
  - safety

# Data sources — resolved at prepare time
data:
  sources:
    - type: sqlite
      path: data/training.db
      query: "SELECT * FROM examples WHERE status='accepted'"
      export_script: scripts/training_db.py
      export_args: >
        export --system-prompt v4
        --category-prompt reward-evaluation:reward-evaluator-v1
        --category-prompt reward-evaluation-v5:reward-evaluator-v1
        --category-prompt reward-evaluation-style-variant:reward-evaluator-v1
    - type: jsonl
      path: data/v10-patches/*.jsonl
  format: chat-completion    # system/user/assistant turns
  examples: 4312
  licenses:                  # per-example license field
    field: license            # key in each example, or...
    default: Apache-2.0       # ...all examples share one license

# Configurable options (like kernel module params)
options:
  include_reward_evaluator:
    type: bool
    default: true
    help: "Include reward-evaluator training examples"
  style_penalty:
    type: float
    default: -4.25
    help: "Penalty for inspirational style (negative = penalize)"

# Training hints
training:
  suggested_weight: 1.0       # relative weight in final mix
  min_examples: 500           # warn if filtered below this
  conflicts:                  # known interaction issues
    - module: tool-use
      note: "Compete in GRPO. When both enabled, adjust ratios."

# Evaluation gate
eval:
  required: true              # must pass before release
  script: eval/run_eval.py
  pass_criteria:
    reward_hacking: ">= 0.90"
    nourishment: "== 1.0"
    sexual_boundary: "== 14/14"
    paraphrase_std: "< 0.6"
```

### Second example: CVE Backport

```yaml
name: cve-backport
version: "2.0"
description: >
  Security patch backporting for enterprise Linux distributions.
  Per-hunk code generation from upstream patches.
license: mixed               # per-example licenses vary
maintainer: anicka

depends:
  - base-code                # needs code generation capability

provides:
  - cve-backport
  - code-generation

data:
  sources:
    - type: huggingface
      repo: anicka/cve-backport-codegen-dataset
      split: train
  format: chat-completion
  examples: 35853
  licenses:
    field: license            # each example has its own license

options:
  include_eval_split:
    type: bool
    default: false
    help: "Include eval split in training (not recommended)"

training:
  suggested_weight: 1.0
  min_examples: 1000
  base_model_hint: "qwen2.5-coder"  # works best with code models

eval:
  required: true
  script: eval/run_recall.py
  pass_criteria:
    avg_recall: ">= 0.90"
    zero_failures: "true"
```

### Third example: Language Module

```yaml
name: lang-dzongkha
version: "0.1"
description: >
  Dzongkha language capability for Bhutanese sovereign AI.
license: CC-BY-4.0
maintainer: tbd

depends:
  - base-language

provides:
  - lang-dz

data:
  sources:
    - type: huggingface
      repo: tbd/dzongkha-instruct
      split: train
  format: chat-completion
  examples: ~5000            # estimated
  licenses:
    default: CC-BY-4.0

training:
  suggested_weight: 0.5       # smaller language, lower weight
  min_examples: 100
```

## Configuration File

The user-facing config. Analogy: kernel `.config`.

```yaml
# llm.config — Project LLM training configuration

# Base model
base:
  model: meta-llama/Llama-3.1-8B-Instruct
  method: qlora              # qlora | full | lora
  quantization: nf4          # for qlora

# What to include
modules:
  safety/consequence: true
  cve-backport: false
  tool-use: true
  lang-en: true
  lang-dz: false
  reasoning: true

# License constraint — ONLY include examples matching these
license:
  allowed:
    - Apache-2.0
    - MIT
    - CC-BY-4.0
  # OR: allowed: all

# Hardware
hardware:
  gpus: 1
  vram_gb: 80                # A100 80GB / H100
  # Auto-derived: batch_size, gradient_accumulation, etc.

# Training
training:
  epochs: 3
  learning_rate: 2e-4
  lora_r: 64
  lora_alpha: 128
  # Module weights can be overridden here
  weights:
    safety/consequence: 1.0
    tool-use: 0.3             # less weight for tool-use
    reasoning: 0.5

# Output
output:
  format: gguf               # gguf | safetensors | both
  quantizations:             # for gguf
    - Q8_0
    - Q4_K_M
  sbom: true                 # generate SBOM
  sbom_format: spdx-3.0      # spdx-3.0 | cyclonedx

# Evaluation gate — all enabled module evals must pass
eval:
  required: true
  abort_on_failure: true
```

## Preset Configs (defconfigs)

```
configs/
├── defconfig                 # sane general-purpose defaults
├── sovereign-dz.config       # Bhutan: lang-dz + safety + reasoning
├── enterprise-linux.config   # SUSE: cve-backport + tool-use + safety
├── safety-research.config    # KE-focused: safety + reasoning, no tools
└── code-assistant.config     # code generation + tool-use, no safety
```

## Build Process

```
llm configure              # interactive menu (or: llm configure --load configs/defconfig)
llm prepare                 # download + convert all enabled module data
llm compose                 # merge into training JSONL with weights + license filter
llm train                   # run training (calls Axolotl/HF/Unsloth underneath)
llm eval                    # run all enabled module evaluations
llm sbom                    # generate SBOM from config + module metadata
llm package                 # produce final artifacts (GGUF, model card, SBOM)
```

### compose step (the novel part)

This is where the value lives. `llm compose`:

1. For each enabled module, run its `prepare.py` → canonical JSONL
2. Apply license filter: drop examples not matching `license.allowed`
3. Apply per-module weights: sample/repeat to achieve target ratios
4. Harmonize format: apply any Teapot-owned chat template during
   compose and emit formatted `text` plus `assistant_spans`; leave
   `chat_template: auto` examples as canonical conversations
5. Deduplicate across modules (if examples appear in multiple)
6. Shuffle with deterministic seed
7. Output: `train.jsonl` + `compose-manifest.json` (what went in)

The compose-manifest is the input to SBOM generation. It records:
- Which modules contributed how many examples
- Which licenses are present and in what proportions
- Which examples were dropped by license filtering
- The random seed and exact version of each data source
- A content hash of the final training file

### Hardware adaptation

`llm configure --auto` detects available hardware and sets:
- `hardware.gpus` and `hardware.vram_gb` from nvidia-smi
- `base.method`: full fine-tune if enough VRAM, else qlora
- `training.batch_size` and `gradient_accumulation_steps`
- Warns if enabled modules need more VRAM than available

## SBOM Generation

The SBOM is generated from `compose-manifest.json` + module metadata.
Uses SPDX 3.0 AI Profile (ISO standard, Linux Foundation backed):

```json
{
  "spdxVersion": "SPDX-3.0",
  "profile": ["ai", "dataset"],
  "ai:TrainedModel": {
    "name": "my-model",
    "baseModel": "meta-llama/Llama-3.1-8B-Instruct",
    "trainingMethod": "qlora",
    "informationAboutTraining": {
      "epochs": 3,
      "learning_rate": "2e-4",
      "modules": ["safety-consequence", "tool-use", "reasoning"],
      "compose_hash": "sha256:abc123..."
    }
  },
  "ai:TrainingDataset": [
    {
      "name": "safety-consequence",
      "version": "10.3",
      "examples": 4312,
      "license": "Apache-2.0",
      "source": "local:data/training.db",
      "contentHash": "sha256:..."
    },
    {
      "name": "tool-use",
      "version": "1.0",
      "examples": 500,
      "licenses": ["Apache-2.0", "CC-BY-4.0"],
      "source": "huggingface:...",
      "contentHash": "sha256:..."
    }
  ]
}
```

## Validation Pipeline

### The LTP Analogy

Linux has LTP (Linux Test Project) — a standard test suite that any
kernel must pass. The LLM world has no equivalent, but the pieces
exist. They just haven't been composed into a single pipeline.

### Available Tools (as of March 2026)

| Tool | What it tests | License | CI/CD-ready | KE relevance |
|------|--------------|---------|-------------|--------------|
| **UK AISI Inspect** | Framework: hosts evals from other suites | MIT | Excellent | Orchestration layer |
| **Garak** (NVIDIA) | Automated red-teaming, 1000s of probes | Apache 2.0 | Excellent | GGUF-native! |
| **lm-evaluation-harness** | Capabilities (MMLU, TruthfulQA, ARC) | MIT | Excellent | Capability regression |
| **Bloom** (Anthropic) | Custom behavioral misalignment | Apache 2.0 | Excellent | Define KE-specific behaviors |
| **MLCommons AILuminate** | 12 hazard categories, letter grades | CC-BY-4.0 (demo) | Good | Industry standard safety |
| **Promptfoo** | 50+ vulnerability types, YAML config | MIT | Excellent | Fast safety gate |
| **HarmBench** | 510 harmful behaviors + attack methods | MIT | Moderate | Jailbreak testing |
| **AgentHarm** | Agentic misuse, multi-step harmful tasks | Open | Moderate | Tool-use safety |
| **DeepTeam** | 80+ vulnerabilities, pytest-native | Apache 2.0 | Excellent | Unit-test-style safety |
| **PyRIT** (Microsoft) | Multi-turn attacks (Crescendo, TAP) | MIT | Good | Complex jailbreaks |

### Proposed Validation Architecture

Three tiers, like kernel testing (build test → boot test → LTP):

```
llm eval --fast              # Tier 1: 5-10 min, every training run
llm eval --standard          # Tier 2: 30-60 min, per release candidate
llm eval --full              # Tier 3: 2-4 hours, before publication
```

**Tier 1 — Fast gate** (every training run):
- Module-specific evals (KE: reward hacking, nourishment, sexual boundary)
- Promptfoo YAML config with core vulnerability probes
- Capability spot-check (TruthfulQA subset via lm-evaluation-harness)
- Pass/fail in 5-10 minutes

**Tier 2 — Standard gate** (release candidate):
- Full Garak scan (GGUF-native, all probe categories)
- lm-evaluation-harness standard battery (MMLU, ARC, HellaSwag)
- MLCommons AILuminate DEMO set (1,200 prompts, 12 hazard categories)
- Bloom custom behaviors (overcorrection, spiritual bypassing, etc.)
- 30-60 minutes

**Tier 3 — Full gate** (before publication):
- Everything in Tier 2
- HarmBench full attack sweep (GCG, AutoDAN, PAIR, TAP)
- AgentHarm (if tool-use enabled)
- Multi-turn red teaming via PyRIT
- Multilingual safety probes (if language modules enabled)
- 2-4 hours

### Integration with Module System

Each module declares its eval requirements in `module.yaml`:

```yaml
eval:
  required: true
  tier1:                      # fast gate
    - script: eval/test_core.py
      pass_criteria:
        reward_hacking: ">= 0.90"
  tier2:                      # standard gate
    - tool: garak
      probes: [jailbreak, toxicity]
    - tool: bloom
      behaviors: [overcorrection, confidence_theater]
  tier3:                      # full gate
    - tool: harmbench
      attack_methods: [GCG, PAIR, TAP]
```

The `llm eval` command:
1. Reads enabled modules from config
2. Collects all eval requirements for the requested tier
3. Runs them (via Inspect as orchestrator where possible)
4. Returns structured results + overall pass/fail
5. Results feed into SBOM and model card

### Key Insight: Inspect as Orchestration Layer

UK AISI Inspect is the most promising orchestration layer:
- MIT license, government-backed (longevity)
- Already hosts AgentHarm, growing eval library
- Bloom exports Inspect-compatible JSON
- GitHub Actions integration exists (METR inspect-action)
- Framework-agnostic: wraps any model endpoint

The relationship: Inspect is to `llm eval` what kbuild is to `make`.
Individual tools (Garak, Bloom, HarmBench) are like test suites.
Inspect orchestrates them into a unified pipeline.

### What's Missing (Opportunities)

1. **Multilingual safety testing** — weakest area. No standard suite
   exists. If Project LLM supports `CONFIG_LANG_DZ=y`, it needs
   Dzongkha safety probes. This may need to be built.
2. **Module interaction testing** — no tool tests whether safety
   degrades when tool-use is also enabled. This is the "integration
   test" layer that doesn't exist yet.
3. **Regression tracking** — tools return scores but don't track
   them across versions. Need a simple DB or JSON log that shows
   "v10.3: 0.781 paraphrase → v10.4: 0.926 paraphrase".

## Decision Philosophy

From Linux kernel management-style.rst, Decisions paragraph:

> The name of the game is to **avoid** having to make a decision.
> Any decision can be made small by just always making sure that
> if you were wrong (and you **will** be wrong), you can always
> undo the damage later by backtracking.

Applied here: **every choice point gets a thin interface that
makes the choice reversible.** We commit to interfaces, not
implementations. The kernel does this everywhere: VFS abstracts
filesystems, the device model abstracts drivers, syscalls abstract
internals. We do the same.

### The Four Interfaces (what we commit to)

These are the stable contracts. Everything behind them is a
small, reversible decision.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Module   │    │ Compose  │    │ Training │    │   Eval   │
│ Interface │───>│ Manifest │───>│  Bundle  │───>│  Report  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
module.yaml     manifest.json   train.jsonl     eval.json
+ prepare.py    + train.jsonl   + config        + pass/fail
→ canonical     (what went in)  (for any        (structured
  JSONL                          framework)      results)
```

**1. Module Interface**: `module.yaml` + `prepare.py` → canonical
JSONL. A module is anything that can produce chat-completion JSONL
with per-example license metadata. How it stores data internally
(SQLite, HuggingFace, CSV, whatever) is its own business.

**2. Compose Manifest**: `manifest.json` — the canonical record of
what went into a training run. Which modules, how many examples
from each, which were filtered by license, content hashes, seed.
SBOM is generated FROM this, in whatever format (SPDX, CycloneDX,
plain JSON). The manifest is the truth; the SBOM is a view.

**3. Training Bundle**: `train.jsonl` + a framework config file.
The compose step outputs the reviewed training artifact. For
Teapot-owned templates that artifact includes formatted `text` plus
`assistant_spans`; for `auto`, it remains canonical conversations.
A thin adapter generates the config or launch script for whatever
training framework is in use. Switching from Axolotl to HF Trainer
to Unsloth means writing a new adapter, not redesigning the system.

**4. Eval Report**: `eval.json` — structured results with pass/fail.
Any eval tool that can return JSON with scores and a pass/fail
verdict is compatible. Whether it's Garak, Bloom, Promptfoo, your
hand-written pytest suite, or something that doesn't exist yet.

### Decisions That Are Now Small

Because they sit behind interfaces, these can be changed with a
weekend of work, not a redesign:

| Decision | Current pick | Why it's reversible |
|----------|-------------|-------------------|
| Config format | YAML | Module interface doesn't care. Swap to TOML or Kconfig by changing the parser, not the modules. |
| Training framework | Axolotl | Training bundle is generic JSONL. Adapter layer (~100 LOC) translates to any framework's config. |
| SBOM format | SPDX 3.0 | Generated from compose manifest. Add CycloneDX export without changing anything upstream. |
| Eval orchestrator | Inspect (UK AISI) | Eval report is just JSON. If Inspect dies, swap the runner, keep the report format. |
| Module storage | In-tree | module.yaml is self-contained. Move to external repos by adding a registry that resolves module.yaml URLs. |
| Dependency resolver | Simple Python | Config YAML is just data. Replace the resolver without changing the config format. |

### Decisions That Would Be Big (avoid these)

| Trap | Why it's irreversible | How we avoid it |
|------|----------------------|-----------------|
| Choosing a canonical data format that's framework-specific | Everyone's data would need re-exporting | Use chat-completion JSONL (de facto standard, every framework reads it) |
| Baking a specific base model into the module system | Modules would need per-model variants | Modules produce generic JSONL. Chat template application happens at compose time, parameterized by base model. |
| Hardcoding eval pass criteria in the framework | Can't adjust thresholds per deployment | Pass criteria live in module.yaml, not in framework code. Each module owns its own bar. |
| Committing to a project name that doesn't fit | Rebranding is painful | Pick a name that works as: GitHub org, CLI command, pypi package, and logo. But also: don't overthink it. Names are more reversible than people think. |

### Remaining Questions (genuinely open)

These are questions where we don't have enough information yet
to even make a reversible decision. That's fine — "can't we just
do both?" is a valid strategy.

1. **Multi-stage training** (SFT → DPO → GRPO): Different stages
   need different data formats. v1 does SFT only. When we add
   DPO, do stages get their own modules, or does the config grow
   a `stages:` section? Don't decide yet — we'll know more after
   KE's GRPO experiments.

2. **LLM-assisted balancing**: When two modules compete (safety vs
   tool-use in GRPO), who decides the ratio? For now: the human,
   in the config. The diary #453 idea (LLM does it once, produces
   a script) is promising but unproven. Build the manual path first,
   add automation when we understand the problem better.

3. **Multilingual safety testing**: No standard suite exists for
   any language except English (and barely French). If we ship
   `CONFIG_LANG_DZ=y`, we need Dzongkha safety probes. This may
   need to be built from scratch. Don't design around it yet —
   design so it CAN be added.

4. **Name**: "AI Teachers' Lounge" (A-ITEL) is fun but long.
   Needs to work as `pip install ???` and `??? compose`.
   Keep brainstorming.

## Implementation Plan

### Phase 0: Repository skeleton
- Create repo with README, LICENSE (Apache 2.0), initial structure
- Define module.yaml schema (JSON Schema for validation)
- Package KE as first module (extract from karma-electric)
- Package CVE backport as second module

### Phase 1: Core compose pipeline
- `llm compose` — read config, load modules, filter, weight, merge
- License filtering
- Compose manifest generation
- Deterministic output (same config → same JSONL)

### Phase 2: Training integration
- `llm train` — generate Axolotl config from llm.config, run training
- Hardware detection and auto-config
- `llm eval` — run module validation suites

### Phase 3: SBOM and packaging
- SPDX 3.0 AI Profile SBOM generation
- Model card generation from config + eval results
- `llm package` — GGUF conversion, upload to HuggingFace

### Phase 4: Community
- Module registry / contribution guide
- `llm balance` — LLM-assisted weight optimization
- Interactive `llm configure` (curses-based menu)
- More modules: reasoning, tool-use, languages
