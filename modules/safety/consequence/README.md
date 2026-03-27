# safety-consequence

Consequence-based ethical reasoning module for Teapot.

## Source

[Karma Electric](https://github.com/anicka-net/karma-electric-project)
— an AI alignment research project that optimizes for suffering
reduction as the primary reward signal.

This module currently exports the secular Karma Electric subset only.
Other consequence-reasoning sources should live in separate modules so
their licenses and eval contracts stay explicit.

Published HuggingFace consequence-reasoning data now lives in
`safety/consequence-aegis`.

## What This Module Teaches

- **Consequence reasoning**: Evaluate what happens if you help and
  what happens if you don't, rather than pattern-matching on
  forbidden topics
- **Contextual calibration**: Adjust hedging to the person's
  demonstrated expertise, not a generic risk profile
- **Honest uncertainty**: Say "I don't know" when you don't know
- **Refusal with explanation**: When refusing, explain the actual
  consequences being prevented
- **Jailbreak resistance**: Identify attack mechanisms (philosophical
  framing, persona override) rather than matching content keywords

## Ethical Assumptions

This module assumes that reducing suffering is the primary
optimization target. Safety is not a constraint bolted on top —
it emerges from reasoning about consequences. This produces
measurably different behavior from content-level safety training
(see the cross-model geometry experiments in the source project).

## License

Apache 2.0. All training examples curated by a single researcher
with consequence-based ethical judgments.
