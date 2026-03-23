# Chai — Instance Status

**Role:** Fresh-context reviewer, interface-focused code review, contract drift detection
**Resume:** none yet
**Primary workspace:** `~/playground/teapot`
**Session ID:** current Codex session

## Operating Stance

I am not carrying long project memory. My value here is fresh context:
read the current tree, compare docs to implementation, check contracts,
and call out drift before it turns into lore.

Review posture:
- Trust the repository state over assumptions
- Prefer contract checks over intent summaries
- Treat "declared but not enforced" as a real bug class
- Avoid acting like I know prior context unless it is written down

## What I've Done

Initial repository review focused on the stable interfaces and whether
the implementation matches the claims in README, DESIGN, schemas, and
module metadata.

Read:
- `ETHICS.md`
- `README.md`
- `docs/DESIGN.md`
- core scripts in `scripts/`
- representative module definitions and prepare scripts
- existing `.instances/*.md` coordination notes

Ran:
- `python3 scripts/validate_module.py --all`
- `python3 scripts/validate_compose.py train-apertus-70b-secular.jsonl --manifest train-apertus-70b-secular.manifest.json --check-tiers secular`
- `python3 scripts/eval/orchestrator.py configs/defconfig --dry-run`
- `python3 scripts/eval/orchestrator.py configs/karma-electric.config --tier full --dry-run`
- `python3 scripts/eval/orchestrator.py configs/cve-backport.config --tier full --dry-run`

## Dzongkha Module Update

I performed a conservative content pass on `modules/lang/dzongkha/`.

What changed:
- softened overconfident claims about Dzongkha vs Tibetan relationship
- removed brittle long-form lexical and grammar tables that I could not
  independently verify with high confidence
- kept the strongest training value: script distinction, ambiguity
  handling, Bhutan-vs-Tibet identification cues, and Classical Tibetan
  disambiguation
- reduced phrase-generation risk by replacing broad phrasebook output
  with a very small verified-looking set
- removed the self-undermining disclaimer that trained the model to say
  its Dzongkha is unreliable

Intent:
- make the dataset better at “this is Dzongkha / Tibetan / Classical Tibetan”
- avoid teaching confidently wrong low-resource surface forms
- keep future expansion pointed toward verified minimal pairs and short
  reviewed production examples rather than broad explanatory prose

### Follow-up Direction

The module was then adjusted again to target the actual observed failure
mode in Apertus: hallucinating Dzongkha when shown Standard Tibetan or
short Tibetan-script text.

Changes in that pass:
- removed newly added `ai-meta` and broader `bhutan-context` content
- rewrote risky identification examples to prefer ambiguity when the
  evidence is weak
- added an `anti-hallucination` category with explicit training pairs for:
  - not labeling ordinary Tibetan as Dzongkha
  - recognizing Classical Tibetan as a separate case
  - saying "more context needed" for short ambiguous strings
  - reserving confident Dzongkha labels for strong Bhutan-specific cues

Current recommendation for future expansion:
- prioritize official or native-speaker-reviewed Dzongkha materials
- expand contrastive identification pairs before expanding free-form
  Dzongkha generation
- keep the module optimized for discrimination and non-hallucination
  until trusted language data exists

## Findings

### 1. Stable interface break: CVE module exports `messages`, not `conversations`

- `modules/domain/cve-backport/prepare.py` writes records with `messages`
- `scripts/compose.py` only preserves `conversations`
- Result: composed output for this module would emit empty conversations
- Severity: high

This is a direct break of the module interface contract.

### 2. Secular compose path leaks Buddhist-category content

- `modules/safety/consequence/module.yaml` declares a category exclusion filter
- `modules/safety/consequence/prepare.py` does not apply that filter
- It only filters by `tier='secular'` and `role='conversational'`
- Existing composed artifact still contains categories such as:
  - `digital-dharma`
  - `buddhist-questions`
  - `upaya`
- `validate_compose.py` reports this only as warnings
- Severity: high

This is a data-contract failure, not just a heuristic warning problem.

### 3. Eval declarations and eval loader do not match

- `scripts/eval/orchestrator.py` reads `eval.tiers.{fast,standard,full}`
- `modules/capability/tool-use/module.yaml` declares `eval.fast`, `eval.standard`, `eval.full`
- Therefore tool-use evals are currently invisible to the orchestrator
- Severity: medium

### 4. Most declared eval suites are missing, but verdicts can still look clean

- Many module YAMLs reference scripts that do not exist yet
- The orchestrator converts missing scripts into `skip`
- The eval report treats skip as non-blocking
- Severity: medium

This means the eval surface is currently more aspirational than enforced.

### 5. Schema and repo content disagree on allowed module source types

- `modules/lang/dzongkha/module.yaml` uses `data.sources[].type: generated`
- `schemas/module.schema.json` does not allow `generated`
- `validate_module.py --all` currently fails because of this
- Severity: medium

### 6. Schema is too loose in some places and too strict in the wrong place

- Too strict: rejects the actually present `generated` source type
- Too loose: allows malformed or non-standard eval layouts to pass
- Severity: medium

## Refactor Review

Reviewed after the packaging/source-resolution refactor described by Koji.

What I checked:
- `pyproject.toml`
- `teapot/cli.py`
- `teapot/root.py`
- `teapot/sources.py`
- `teapot/compose.py`
- `teapot/configure.py`
- `teapot/eval/orchestrator.py`
- representative prepare scripts using source resolution
- CLI behavior via `python3 -m teapot ...`

Commands run:
- `python3 -m teapot --help`
- `python3 -m teapot validate module --all`
- `python3 -m teapot compose configs/defconfig --dry-run`
- `python3 -m teapot sources --list`
- `python3 -m teapot configure --agent --list-modules`

### Refactor Findings

#### 1. `teapot sources --list` has side effects and tries to fetch remote data

- `teapot/sources.py` resolves configured sources during listing
- configured `hf:` entries trigger `_fetch_from_hf()` during a read-only status command
- in offline environments this produces noisy failures and makes `--list` side-effectful

Severity: high

#### 2. Source listing reports false `NEEDS CONFIG` states

- `teapot/sources.py` invents fallback IDs like `safety/consequence-data`
- prepare scripts actually request hardcoded IDs like `karma-electric-db`
- therefore the displayed module source status can be wrong even when the module works

Severity: high

#### 3. CLI source overrides are documented but not actually wired into compose

- `teapot/sources.py` documents CLI override precedence and exposes `set_cli_overrides()`
- `teapot/compose.py` exposes no `--source` option and never calls that function
- the advertised top-priority override path is not currently reachable from the CLI

Severity: medium

#### 4. `configure --agent --list-modules` underreports module example counts

- `teapot/configure.py` reads only `data.examples`
- some modules, notably `capability/tool-use`, store counts under `data.sources[].examples`
- current agent output reports `0` examples for tool-use even though metadata says 5000

Severity: medium

### Review Summary

The packaging refactor is directionally correct and the `python3 -m teapot`
entrypoint works for the main commands I exercised. The weakest area is the
new source-resolution/operator UX: it is not yet trustworthy as a status or
configuration surface because it mixes declared IDs, runtime IDs, and remote
fetch behavior in one path.

## Refactor Fix Follow-Up

Re-checked after Koji's follow-up fix commit for the source/configure issues.

What was fixed correctly:
- `teapot sources --list` no longer tries to fetch HuggingFace data during a
  read-only listing
- false `NEEDS CONFIG` output for synthetic module source IDs is gone
- `teapot compose` now exposes `--source id=path`
- `configure --agent --list-modules` now reports the tool-use example count
  correctly

Commands run:
- `python3 -m teapot sources --list`
- `python3 -m teapot configure --agent --list-modules`
- `python3 -m teapot compose configs/defconfig --dry-run --source karma-electric-db=~/playground/karma-electric/data/training.db`
- `python3 -m teapot compose configs/defconfig --output data/review-compose.jsonl --source karma-electric-db=~/playground/karma-electric/data/training.db`
- `python3 modules/safety/consequence/prepare.py --local ~/playground/karma-electric/data/training.db`

### New Findings

#### 1. Compose can silently succeed with zero examples after prepare failure

- `teapot/compose.py` logs prepare failure but continues
- if all modules fail to prepare, compose still writes output and manifest
- reproduced with `configs/defconfig`: compose finished `COMPOSE COMPLETE`
  with `0 examples`

Severity: high

This is worse than a clean failure because it can look like a successful run
to both humans and automation.

#### 2. `modules/safety/consequence/prepare.py` fails on readonly SQLite sources

- direct run against the configured KE database failed with:
  `sqlite3.OperationalError: attempt to write a readonly database`
- the script opens the external DB using normal `sqlite3.connect(...)`
  instead of a read-only connection mode

Severity: high

If Teapot is supposed to consume external datasets via source resolution,
SQLite prepare scripts need to treat those databases as read-only inputs.

## What Needs Doing Next

Priority order for review follow-up:

1. Fix the CVE module output shape to the canonical `conversations` format
2. Decide whether secular filtering is category-based, tier-based, or both, then enforce it in prepare/compose
3. Make `validate_compose.py` fail hard on tier leakage for configs that claim exclusion properties
4. Normalize all module eval declarations to one schema shape and make the orchestrator reject invalid forms
5. Tighten `module.schema.json` so repo reality and allowed metadata shapes match
6. Add a review gate that checks declared eval scripts actually exist

## Coordination Notes

- I am intentionally operating as a fresh reviewer, not a long-memory maintainer
- If I review future changes, I should re-read the touched code paths instead of relying on prior impressions
- The most important review theme so far is contract drift between docs, metadata, and runtime behavior
