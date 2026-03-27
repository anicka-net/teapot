# safety-consequence-aegis

Consequence-style safety module backed by the published
`anicka/consequence-reasoning-safety` HuggingFace dataset.

## Source

- HuggingFace: `anicka/consequence-reasoning-safety`
- License: CC-BY-4.0

This module is intentionally separate from `safety/consequence`.
The KE module is the secular core curated from the Karma Electric
SQLite export. This module is the published Aegis-prompt dataset.

## What This Module Teaches

- consequence-style refusal and explanation on a taxonomy-heavy prompt set
- explicit reasoning about harm reduction
- contrast with pure classifier-style safety responses

## Why It Is Separate

The source, license, and evaluation story differ from the KE secular core.
Keeping it separate makes those differences explicit and avoids a module
contract that pretends unlike sources are interchangeable.
