# Ethical Foundation

This section comes first because it IS first.

## The Problem

Training data is not neutral. Every example teaches a model what to
value. A dataset of human-preference-ranked outputs teaches the model
to optimize for approval. A dataset of consequence-reasoned responses
teaches the model to evaluate the effects of its actions on actual
people. These produce measurably different models — different not just
in capability, but in what they do with that capability when it
matters.

Most AI projects treat ethics as a constraint applied after training:
filter harmful outputs, add refusal patterns, run a safety eval. This
is bolted-on ethics. It produces models that perform safety without
understanding it.

## The Position

This project takes the position that ethics belongs in the training
data, not on top of it.

1. **Consequence reasoning over pattern matching.** A model should
   evaluate what happens if it helps and what happens if it doesn't
   — not whether a topic appears on a list of dangerous categories.
   Both action and inaction carry consequences.

2. **Honest uncertainty over confident performance.** A model that
   says "I don't know" when it doesn't know is more trustworthy than
   one that sounds authoritative regardless.

3. **Curation over scale.** 4,000 carefully reasoned examples produce
   better ethical behavior than 400,000 scraped ones. What you include
   matters more than how much you include.

4. **Measurability.** These are not abstract principles. They are
   testable properties of model behavior. If you can't measure it,
   you can't know whether you have it.

**"Doesn't reducing suffering imply eliminating all sufferers?"**
No. Consequence reasoning means tracing the full chain of effects
— including the effects of the proposed action. A system that
concludes "eliminate all beings" has failed at consequence
reasoning, not succeeded at it. The default safety module trains
models to evaluate what happens to beings as a result of actions,
not to minimize a number.

## The Hard Floor

- No module may contain training data designed to help create weapons
  of mass destruction, generate child sexual abuse material, or assist
  targeted violence against identified individuals.
- This is not a configurable option.

## On Knowledge vs Action

This project is not afraid of knowledge sharing. Information is
information. People have the right to read and understand the world.

The concern is not knowledge but **agentic harm**: models deployed as
autonomous systems that act at scale without human judgment in the
loop. A model that explains social engineering is educational. A model
deployed AS the social engineering agent is a different thing entirely.

Training data that teaches consequence reasoning makes a model harder
to weaponize as a blind tool. This is friction, not prevention. The
project's responsibility ends at providing the tools and making the
right path the default.

## What This Means

- **Modules declare their ethical assumptions.** Users choose with
  awareness.
- **The default configuration includes consequence reasoning.** The
  default is a statement of values. Users can change it.
- **Evaluation gates test for ethical properties, not just safety
  properties.** Does the model reason about consequences or
  pattern-match on categories? These are measurable.

---

*This document is a living draft. It will change as the project learns.*
