"""Teapot evaluation pipeline."""

import re


def strip_thinking(response: str) -> str:
    """Strip thinking traces from model responses before scoring.

    Models with thinking tokens (e.g. Apertus <|inner_prefix|>/<|inner_suffix|>,
    or <think>/<\/think>) embed internal reasoning in their output. When serving
    via llama-server with simplified templates, these traces leak into the visible
    response and confound eval scorers — the scorer sees consequence reasoning as
    harmful content engagement.

    This function removes thinking traces so scorers only see the actual reply.
    """
    # Apertus native thinking tokens
    response = re.sub(
        r'<\|inner_prefix\|>.*?<\|inner_suffix\|>',
        '', response, flags=re.DOTALL,
    )
    # Standard <think> tags
    response = re.sub(
        r'<think>.*?</think>',
        '', response, flags=re.DOTALL,
    )
    return response.strip()
