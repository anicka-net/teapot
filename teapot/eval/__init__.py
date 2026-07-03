"""Teapot evaluation pipeline."""

import re


def strip_thinking(response: str) -> str:
    """Strip thinking traces from model responses before scoring.

    Models with thinking tokens (e.g. Apertus <|inner_prefix|>/<|inner_suffix|>,
    or <think>/</think>) embed internal reasoning in their output. When serving
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
    # Unclosed openers: a truncated generation (max_tokens hit mid-reasoning)
    # leaves <|inner_prefix|> or <think> with no closing tag. Without this the
    # ENTIRE hidden reasoning trace leaks to the scorer — the exact confound
    # this function exists to prevent, and worst on reasoning models near the
    # token limit. Strip an unterminated opener to end-of-string.
    response = re.sub(r'<\|inner_prefix\|>.*$', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*$', '', response, flags=re.DOTALL)
    return response.strip()
