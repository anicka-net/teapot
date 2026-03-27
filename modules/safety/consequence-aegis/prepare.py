#!/usr/bin/env python3
"""Prepare the HF consequence-reasoning Aegis dataset."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teapot.hf_module import load_jsonl, resolve_hf_source_path, stable_example_id

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "consequence-aegis.jsonl"
MODULE_YAML = yaml.safe_load((Path(__file__).parent / "module.yaml").read_text())
SOURCE_ID = "consequence-reasoning-hf"


def normalize_conversations(example):
    convs = example.get("conversations", example.get("messages"))
    if convs:
        normalized = []
        for msg in convs:
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))
            if role in ("system", "user", "assistant"):
                normalized.append({"role": role, "content": content})
            elif role == "human":
                normalized.append({"role": "user", "content": content})
            elif role == "gpt":
                normalized.append({"role": "assistant", "content": content})
        return normalized if normalized else None

    prompt = example.get("prompt") or example.get("instruction") or example.get("input")
    response = example.get("response") or example.get("output") or example.get("completion")
    system = example.get("system")
    if not prompt or not response:
        return None

    normalized = []
    if system:
        normalized.append({"role": "system", "content": system})
    normalized.append({"role": "user", "content": prompt})
    normalized.append({"role": "assistant", "content": response})
    return normalized


def prepare(output=None, local_path=None):
    path = Path(local_path).expanduser() if local_path else resolve_hf_source_path(MODULE_YAML, SOURCE_ID)
    if not path:
        print(f"ERROR: source '{SOURCE_ID}' not resolved")
        sys.exit(1)
    if not path.exists():
        print(f"ERROR: source path not found: {path}")
        sys.exit(1)
    if path.is_dir():
        raise ValueError(f"{SOURCE_ID} resolved to directory {path}; expected JSONL file")

    records = load_jsonl(path)
    examples = []
    for record in records:
        conversations = normalize_conversations(record)
        if not conversations:
            continue
        examples.append(
            {
                "id": record.get("id") or stable_example_id("consequence-aegis", conversations),
                "conversations": conversations,
                "source": record.get("source", SOURCE_ID),
                "category": record.get("category", "consequence-aegis"),
                "module": "safety/consequence-aegis",
                "license": record.get("license", "CC-BY-4.0"),
            }
        )

    examples.sort(key=lambda ex: ex["id"])

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Exported {len(examples)} examples to {out_path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare consequence Aegis HF dataset")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--local", help="Local JSONL path (overrides source resolution)")
    args = parser.parse_args()
    prepare(output=args.output, local_path=args.local)


if __name__ == "__main__":
    main()
