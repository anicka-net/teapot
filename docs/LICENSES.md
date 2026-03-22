# License Handling in Teapot

## The Problem

Training data comes under many licenses. When you build a model,
you need to know:
1. What licenses are in the training data?
2. Are they compatible with each other?
3. Are they compatible with your intended use?
4. Can you prove this?

Nobody else in the ML ecosystem solves this systematically.

## How Teapot Handles Licenses

### Per-Example Metadata

Every training example must have a license. Modules declare this
in `module.yaml`:

```yaml
data:
  licenses:
    default: Apache-2.0           # all examples share one license
    # OR
    field: license                 # per-example license in a JSON field
```

If a module has `field: license`, each example in the JSONL must
have a `"license": "SPDX-ID"` field. Examples without a license
field are rejected during `teapot compose`.

### License Filtering

The user's config declares allowed licenses:

```yaml
license:
  allowed:
    - Apache-2.0
    - MIT
    - CC-BY-4.0
  # OR
  allowed: all                    # no filtering
```

During `teapot compose`:
1. Each example's license is checked against `allowed`
2. Non-matching examples are dropped
3. The compose manifest records how many were dropped and why
4. If a module falls below `min_examples` after filtering, compose
   warns (or fails, if the module is required)

### SPDX Identifiers

All licenses use [SPDX identifiers](https://spdx.org/licenses/).
Common ones in training data:

| SPDX ID | Name | Permissive? |
|---------|------|-------------|
| Apache-2.0 | Apache License 2.0 | Yes |
| MIT | MIT License | Yes |
| CC-BY-4.0 | Creative Commons Attribution 4.0 | Yes |
| CC-BY-SA-4.0 | Creative Commons Attribution ShareAlike 4.0 | Copyleft |
| GPL-2.0-only | GNU GPL v2 only | Copyleft |
| GPL-2.0-or-later | GNU GPL v2 or later | Copyleft |
| GPL-3.0-only | GNU GPL v3 only | Copyleft |
| LGPL-2.1-only | GNU LGPL v2.1 only | Weak copyleft |
| BSD-2-Clause | BSD 2-Clause | Yes |
| BSD-3-Clause | BSD 3-Clause | Yes |
| MPL-2.0 | Mozilla Public License 2.0 | Weak copyleft |
| CC0-1.0 | Creative Commons Zero | Public domain |
| Unlicense | The Unlicense | Public domain |

### License Compatibility

Some license combinations are incompatible. Teapot checks for
known conflicts during compose:

- **GPL-2.0-only + Apache-2.0**: Incompatible (GPL-2.0 is not
  compatible with Apache-2.0 patent grant)
- **GPL-3.0-only + Apache-2.0**: Compatible (FSF says so)
- **CC-BY-SA-4.0 + Apache-2.0**: Check required (ShareAlike
  may impose constraints)

When incompatible licenses appear in the same training run,
`teapot compose` warns. Whether mixed-license training data
creates a derivative work is an open legal question — Teapot
documents what went in and lets users make the legal judgment.

### Preset License Profiles

```yaml
# Permissive only — safe for commercial use
license:
  allowed: [Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, CC0-1.0, Unlicense, CC-BY-4.0]

# GPL-compatible — for open-source models
license:
  allowed: [Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, GPL-2.0-or-later, GPL-3.0-only, LGPL-2.1-only, CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0]

# Everything — you handle the legal analysis
license:
  allowed: all
```

### SBOM Output

The compose manifest (and generated SBOM) records:
- Total examples per license
- Which modules contributed which licenses
- Which examples were filtered out and why
- The final license composition of the training data

This feeds into SPDX 3.0 AI Profile generation, providing the
`ai:TrainingDataset` entries with accurate license information.

## Open Questions

1. **Is training data a derivative work?** If a model is trained on
   GPL-licensed code examples, is the model itself GPL? Current legal
   consensus is unclear. Teapot documents the inputs accurately;
   the legal interpretation is the user's responsibility.

2. **License of model outputs?** If a model trained on Apache-2.0
   data generates code, what license applies to that code? Again,
   unresolved. Teapot provides the provenance; users decide.

3. **Fair use / research exceptions?** Some jurisdictions allow
   training on copyrighted data for research. Teapot's license
   filtering lets users be conservative (filter strictly) or
   liberal (allow all, claim fair use) based on their jurisdiction.
