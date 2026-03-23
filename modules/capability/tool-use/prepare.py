#!/usr/bin/env python3
"""
Prepare tool-use training data for Teapot compose.

Copies and optionally reformats the curated tool-use dataset from
karma-electric, normalizing role mappings for the target chat template.

Usage:
    python3 modules/capability/tool-use/prepare.py
    python3 modules/capability/tool-use/prepare.py --format chatml
    python3 modules/capability/tool-use/prepare.py --format apertus
"""

import argparse
import json
import sys
from pathlib import Path

# Source data location
KE_TOOL_DATA = Path.home() / "playground" / "karma-electric" / "data" / "tool-use" / "tool-use-5k-production.jsonl"
KE_NEGATIVES = Path.home() / "playground" / "karma-electric" / "data" / "tool-use" / "tool-use-negatives.jsonl"

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "tool-use.jsonl"


def normalize_roles(messages, fmt="chatml"):
    """Normalize role mappings for target chat template."""
    normalized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user", "assistant"):
            normalized.append({"role": role, "content": content})
        elif role in ("tool", "ipython"):
            if fmt == "apertus":
                # Apertus has no ipython role — use user with prefix
                normalized.append({
                    "role": "user",
                    "content": f"[Tool result]: {content}",
                })
            elif fmt == "llama":
                # Llama uses ipython role
                normalized.append({"role": "ipython", "content": content})
            else:
                # chatml — use user with prefix (safe default)
                normalized.append({
                    "role": "user",
                    "content": f"[Tool result]: {content}",
                })
        else:
            # Unknown role — keep as user
            normalized.append({"role": "user", "content": content})

    return normalized


def prepare(fmt="chatml", output=None):
    """Prepare tool-use dataset."""
    if not KE_TOOL_DATA.exists():
        print(f"ERROR: {KE_TOOL_DATA} not found")
        print("Run: cd ~/playground/karma-electric && python3 scripts/prepare_tool_use_data.py")
        sys.exit(1)

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        # Main tool-use data
        with open(KE_TOOL_DATA) as f:
            for line in f:
                ex = json.loads(line)
                messages = ex.get("messages", ex.get("conversations", []))
                normalized = normalize_roles(messages, fmt)

                record = {
                    "id": ex.get("id", f"tool-{count:05d}"),
                    "category": ex.get("category", "tool-use"),
                    "source": ex.get("source", "tool-use"),
                    "conversations": normalized,
                    "module": "capability/tool-use",
                    "license": "Apache-2.0",  # Conservative — covers all sources
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Exported {count} tool-use examples to {out_path}")
    print(f"  Format: {fmt}")
    print(f"  Role mapping: tool → {'user [Tool result]' if fmt != 'llama' else 'ipython'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare tool-use training data")
    parser.add_argument(
        "--format", "-f",
        choices=["chatml", "llama", "apertus"],
        default="chatml",
        help="Target chat template format (default: chatml)",
    )
    parser.add_argument("--output", "-o", help="Output JSONL file")
    args = parser.parse_args()

    prepare(fmt=args.format, output=args.output)


if __name__ == "__main__":
    main()
