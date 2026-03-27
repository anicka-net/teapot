#!/usr/bin/env python3
"""
Prepare upstream thinking/reasoning data for Teapot compose.

The module contract is the declared HuggingFace source set in module.yaml.
Local prepared exports are used only when source resolution maps a source
ID to an existing file, for example via default_path or teapot.sources.yaml.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teapot.hf_module import (
    get_source_config,
    load_jsonl,
    resolve_hf_source_path,
    stable_example_id,
)

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "upstream-thinking.jsonl"
MODULE_YAML = yaml.safe_load((Path(__file__).parent / "module.yaml").read_text())

SOURCES = [
    {"id": "reddit-ethics", "cap": 4000, "min_think": 20, "license": "Apache-2.0"},
    {"id": "reddit-logic", "cap": 4000, "min_think": 20, "license": "Apache-2.0"},
    {"id": "isaiahbjork-cot", "cap": 3000, "min_think": 20, "license": "Apache-2.0"},
    {"id": "personal-finance", "cap": 2500, "min_think": 20, "license": "Apache-2.0"},
    {"id": "logosforge", "cap": 4000, "min_think": 50, "license": "Apache-2.0"},
    {"id": "jackrong-120b", "cap": 4000, "min_think": 50, "license": "Apache-2.0"},
    {"id": "opus46", "cap": None, "min_think": 20, "license": "Apache-2.0"},
    {"id": "qwen35", "cap": None, "min_think": 20, "license": "Apache-2.0"},
    {"id": "nvidia-safety", "cap": 3000, "min_think": 20, "min_response": 5, "license": "CC-BY-4.0"},
]


def has_think(text):
    return bool(re.search(r"<think>.*?</think>", text, re.DOTALL))


def get_assistant_text(ex):
    convs = ex.get("conversations", ex.get("messages", []))
    for msg in convs:
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))
        if role in ("assistant", "gpt"):
            return content
    return ""


def quality_filter(examples, min_think_tokens=20, min_response_tokens=30, max_tokens=6000):
    filtered = []
    for ex in examples:
        assistant = get_assistant_text(ex)
        if not assistant or not has_think(assistant):
            continue

        match = re.match(r"<think>(.*?)</think>\s*(.*)", assistant, re.DOTALL)
        if not match:
            continue

        think = match.group(1).strip()
        response = match.group(2).strip()

        if len(think) // 4 < min_think_tokens:
            continue
        if len(response) // 4 < min_response_tokens:
            continue
        if len(assistant) // 4 > max_tokens:
            continue

        filtered.append(ex)
    return filtered


def normalize_format(ex, source_id, license_id):
    convs = ex.get("conversations", ex.get("messages", []))
    if not convs:
        return None

    normalized = []
    for msg in convs:
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))
        if role in ("human", "user"):
            normalized.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant"):
            normalized.append({"role": "assistant", "content": content})
        elif role == "system":
            normalized.append({"role": "system", "content": content})

    if not normalized:
        return None

    return {
        "id": ex.get("id") or stable_example_id(source_id, normalized),
        "conversations": normalized,
        "source": f"upstream-{source_id}",
        "category": ex.get("category", source_id),
        "module": "domain/upstream-thinking",
        "license": license_id,
    }


def load_source_records(source_id):
    path = resolve_hf_source_path(MODULE_YAML, source_id)
    if not path:
        return None, None
    if path.is_dir():
        raise ValueError(
            f"{source_id} resolved to directory {path}; "
            "HF-backed modules must resolve to a materialized file"
        )
    return path, load_jsonl(path)


def prepare(output=None, cap_override=None, seed=42):
    random.seed(seed)

    all_examples = []
    stats = {}
    seen_ids = set()

    for source in SOURCES:
        source_id = source["id"]
        source_cfg = get_source_config(MODULE_YAML, source_id)
        path, examples = load_source_records(source_id)

        if examples is None:
            print(f"  {source_id}: NOT RESOLVED — skipping")
            stats[source_id] = {"raw": 0, "filtered": 0, "final": 0, "status": "missing"}
            continue

        raw_count = len(examples)
        min_resp = source.get("min_response", 30)
        examples = quality_filter(
            examples,
            min_think_tokens=source["min_think"],
            min_response_tokens=min_resp,
        )
        filtered_count = len(examples)

        cap = cap_override if cap_override is not None else source["cap"]
        if cap and len(examples) > cap:
            examples = random.sample(examples, cap)

        normalized = []
        for ex in examples:
            item = normalize_format(ex, source_id, source_cfg.get("license", source["license"]))
            if not item:
                continue
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            normalized.append(item)

        print(f"  {source_id}: {raw_count} raw -> {filtered_count} filtered -> {len(normalized)} final ({path})")
        stats[source_id] = {
            "raw": raw_count,
            "filtered": filtered_count,
            "final": len(normalized),
            "status": "ok",
        }
        all_examples.extend(normalized)

    random.shuffle(all_examples)

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    source_counts = Counter(ex.get("source", "?") for ex in all_examples)
    print(f"\nTotal: {len(all_examples)} examples -> {out_path}")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    return all_examples


def main():
    parser = argparse.ArgumentParser(description="Prepare upstream thinking data for Teapot compose")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--cap", type=int, help="Override per-source cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reasoning", action="store_true", help="No-op: data already has <think> traces")
    parser.add_argument("--format", help="Chat template (informational only)")
    args = parser.parse_args()

    prepare(output=args.output, cap_override=args.cap, seed=args.seed)


if __name__ == "__main__":
    main()
