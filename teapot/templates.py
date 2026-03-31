#!/usr/bin/env python3
"""Chat template handling for Teapot compose.

Teapot's owned templates are applied during compose so the training
artifact itself is the source of truth. For those templates we emit:
- canonical `conversations` for inspection/debugging
- formatted `text` for training
- `assistant_spans` so backends can mask loss without re-templating
"""


# Apertus special token IDs for reference:
# 61: <|system_start|>, 62: <|system_end|>
# 63: <|developer_start|>, 64: <|developer_end|>
# 65: <|user_start|>, 66: <|user_end|>
# 67: <|assistant_start|>, 68: <|assistant_end|> (= EOS)
# 69: <|inner_prefix|>, 70: <|inner_suffix|>
# 71: <|tools_prefix|>, 72: <|tools_suffix|>
# 9: [TOOL_CALLS], 5: [AVAILABLE_TOOLS], 6: [/AVAILABLE_TOOLS]


def _join_parts(parts):
    text = ""
    spans = []
    for part, is_assistant in parts:
        start = len(text)
        text += part
        if is_assistant:
            spans.append([start, len(text)])
    return text, spans


def format_apertus(conversations, thinking=False, tools=False):
    """Format conversations in Apertus native token format."""
    parts = []

    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append((f"<|system_start|>{content}<|system_end|>", False))
            # Developer message controls deliberation and tools
            dev_parts = []
            dev_parts.append(f"Deliberation: {'enabled' if thinking else 'disabled'}")
            dev_parts.append(f"Tool Capabilities: {'enabled' if tools else 'disabled'}")
            dev_text = "\n".join(dev_parts)
            parts.append((f"<|developer_start|>{dev_text}<|developer_end|>", False))

        elif role == "user":
            parts.append((f"<|user_start|>{content}<|user_end|>", False))

        elif role == "assistant":
            if thinking and "<think>" in content:
                # Convert <think>...</think> to <|inner_prefix|>...<|inner_suffix|>
                import re
                match = re.match(r'<think>(.*?)</think>\s*(.*)', content, re.DOTALL)
                if match:
                    think_content = match.group(1).strip()
                    # Strip any residual </think> tags (from double-tagged source data)
                    response_content = match.group(2).replace('</think>', '').strip()
                    parts.append((
                        f"<|assistant_start|>"
                        f"<|inner_prefix|>{think_content}<|inner_suffix|>"
                        f"{response_content}"
                        f"<|assistant_end|>"
                    , True))
                else:
                    # Has <think> but doesn't match pattern — pass through
                    parts.append((f"<|assistant_start|>{content}<|assistant_end|>", True))
            else:
                parts.append((f"<|assistant_start|>{content}<|assistant_end|>", True))

        elif role in ("ipython", "tool"):
            # Tool results go as user messages with prefix
            parts.append((f"<|user_start|>[Tool result]: {content}<|user_end|>", False))

    return _join_parts(parts)


def format_chatml(conversations):
    """Format conversations in ChatML format."""
    parts = []
    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append((f"<|im_start|>{role}\n{content}<|im_end|>", role == "assistant"))
    return _join_parts(parts)


def format_llama3(conversations):
    """Format conversations in Llama 3 format."""
    parts = [("<|begin_of_text|>", False)]
    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append((
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        , role == "assistant"))
    return _join_parts(parts)


def format_conversation(conversations, template, thinking=False, tools=False):
    """Format a conversation and return (text, assistant_spans)."""
    if template == "auto" or template is None:
        return None, []

    if template == "apertus":
        return format_apertus(conversations, thinking=False, tools=tools)

    if template == "apertus-think":
        return format_apertus(conversations, thinking=True, tools=tools)

    if template == "apertus-tools":
        return format_apertus(conversations, thinking=False, tools=True)

    if template == "apertus-full":
        return format_apertus(conversations, thinking=True, tools=True)

    if template == "chatml":
        return format_chatml(conversations)

    if template == "llama3":
        return format_llama3(conversations)

    # Unknown template — pass through
    return None, []


# Template metadata for documentation and validation
TEMPLATES = {
    "auto": {
        "description": "Pass through as-is, let training framework handle formatting",
        "special_tokens": [],
    },
    "chatml": {
        "description": "ChatML format (<|im_start|>role\\ncontent<|im_end|>)",
        "special_tokens": ["<|im_start|>", "<|im_end|>"],
    },
    "llama3": {
        "description": "Llama 3 format",
        "special_tokens": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    },
    "apertus": {
        "description": "Apertus native format (no thinking, no tools)",
        "special_tokens": ["<|system_start|>", "<|system_end|>", "<|developer_start|>",
                          "<|developer_end|>", "<|user_start|>", "<|user_end|>",
                          "<|assistant_start|>", "<|assistant_end|>"],
    },
    "apertus-think": {
        "description": "Apertus with deliberation (converts <think> to <|inner_prefix|>/<|inner_suffix|>)",
        "special_tokens": ["<|system_start|>", "<|system_end|>", "<|developer_start|>",
                          "<|developer_end|>", "<|user_start|>", "<|user_end|>",
                          "<|assistant_start|>", "<|assistant_end|>",
                          "<|inner_prefix|>", "<|inner_suffix|>"],
    },
    "apertus-full": {
        "description": "Apertus with deliberation and tools",
        "special_tokens": ["<|system_start|>", "<|system_end|>", "<|developer_start|>",
                          "<|developer_end|>", "<|user_start|>", "<|user_end|>",
                          "<|assistant_start|>", "<|assistant_end|>",
                          "<|inner_prefix|>", "<|inner_suffix|>",
                          "<|tools_prefix|>", "<|tools_suffix|>",
                          "[TOOL_CALLS]", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]"],
    },
}
