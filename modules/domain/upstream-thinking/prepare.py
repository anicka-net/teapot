#!/usr/bin/env python3
"""
Prepare upstream thinking/reasoning data for Teapot compose.

Cache-first: if converted data exists in the cache directory, use it.
Otherwise, download from HuggingFace, convert to <think> format,
and save to cache for next time.

Usage:
    python3 modules/domain/upstream-thinking/prepare.py
    python3 modules/domain/upstream-thinking/prepare.py --output data/upstream-thinking.jsonl
    python3 modules/domain/upstream-thinking/prepare.py --reasoning  # already has <think>, no-op
    python3 modules/domain/upstream-thinking/prepare.py --cap 4000   # cap per source
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "upstream-thinking.jsonl"

# Source definitions: id, cache filename, subsample cap, min_think_tokens
SOURCES = [
    # Ethics / moral reasoning
    {"id": "reddit-ethics", "cache": "reddit-ethics-think.jsonl",
     "cap": 4000, "min_think": 20, "license": "Apache-2.0"},
    {"id": "reddit-logic", "cache": "reddit-logic-think.jsonl",
     "cap": 4000, "min_think": 20, "license": "Apache-2.0"},

    # General chain-of-thought
    {"id": "isaiahbjork-cot", "cache": "isaiahbjork-think.jsonl",
     "cap": 3000, "min_think": 20, "license": "Apache-2.0"},
    {"id": "personal-finance", "cache": "personal-finance-think.jsonl",
     "cap": 2500, "min_think": 20, "license": "Apache-2.0"},

    # STEM (pre-filtered versions)
    {"id": "logosforge", "cache": "logosforge-filtered.jsonl",
     "filtered": True, "cap": 4000, "min_think": 50, "license": "Apache-2.0"},
    {"id": "jackrong-120b", "cache": "jackrong-120b-filtered.jsonl",
     "filtered": True, "cap": 4000, "min_think": 50, "license": "Apache-2.0"},

    # Smaller sets (keep all)
    {"id": "opus46", "cache": "opus46-filtered.jsonl",
     "filtered": True, "cap": None, "min_think": 20, "license": "Apache-2.0"},
    {"id": "qwen35", "cache": "qwen35-filtered.jsonl",
     "filtered": True, "cap": None, "min_think": 20, "license": "Apache-2.0"},

    # NVIDIA safety reasoning (taxonomy classification — short responses, long think)
    {"id": "nvidia-safety", "cache": "nvidia-tf_efficient_reasoning-think.jsonl",
     "cap": 3000, "min_think": 20, "min_response": 5, "license": "CC-BY-4.0"},
]


def find_cache_dir():
    """Find the cache directory with pre-converted data."""
    module_yaml = Path(__file__).parent / "module.yaml"
    if module_yaml.exists():
        cfg = yaml.safe_load(module_yaml.read_text())
        cache_cfg = cfg.get("data", {}).get("cache", {})
        cache_path = Path(cache_cfg.get("path", "")).expanduser()
        if cache_path.exists():
            return cache_path, cache_cfg.get("converted_dir", "converted"), cache_cfg.get("filtered_dir", "filtered")

    # Fallback: common location
    ke_upstream = Path.home() / "playground/karma-electric/data/upstream-thinking"
    if ke_upstream.exists():
        return ke_upstream, "converted", "filtered"

    return None, None, None


def has_think(text):
    """Check if text has <think>...</think> block."""
    return bool(re.search(r'<think>.*?</think>', text, re.DOTALL))


def get_assistant_text(ex):
    """Extract assistant content from any format."""
    convs = ex.get("conversations", ex.get("messages", []))
    for msg in convs:
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))
        if role in ("assistant", "gpt"):
            return content
    return ""


def quality_filter(examples, min_think_tokens=20, min_response_tokens=30, max_tokens=6000):
    """Filter for quality: sufficient think trace and response."""
    filtered = []
    for ex in examples:
        assistant = get_assistant_text(ex)
        if not assistant:
            continue
        if not has_think(assistant):
            continue

        match = re.match(r'<think>(.*?)</think>\s*(.*)', assistant, re.DOTALL)
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
    """Normalize to Teapot JSONL format."""
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
        "id": ex.get("id", f"{source_id}-{hash(json.dumps(normalized)) & 0xFFFFFFFF:08x}"),
        "conversations": normalized,
        "source": f"upstream-{source_id}",
        "category": ex.get("category", source_id),
        "module": "domain/upstream-thinking",
        "license": license_id,
    }


def load_cached(cache_dir, converted_dir, filtered_dir, source):
    """Load pre-converted/filtered data from cache."""
    source_id = source["id"]
    cache_file = source["cache"]

    # Filtered sources live in filtered/, converted in converted/
    if source.get("filtered"):
        path = Path(cache_dir) / filtered_dir / cache_file
    else:
        path = Path(cache_dir) / converted_dir / cache_file

    if not path.exists():
        return None

    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def prepare(output=None, cap_override=None, seed=42):
    """Prepare upstream thinking data using cache."""
    cache_dir, converted_dir, filtered_dir = find_cache_dir()
    if not cache_dir:
        print("ERROR: No cache directory found.")
        print("  Expected: ~/playground/karma-electric/data/upstream-thinking/")
        print("  Or configure cache.path in module.yaml")
        sys.exit(1)

    print(f"Cache: {cache_dir}")
    random.seed(seed)

    all_examples = []
    stats = {}

    for source in SOURCES:
        source_id = source["id"]
        examples = load_cached(cache_dir, converted_dir, filtered_dir, source)

        if examples is None:
            print(f"  {source_id}: NOT CACHED — skipping")
            print(f"    Expected: {cache_dir}/{converted_dir if not source.get('filtered') else filtered_dir}/{source['cache']}")
            stats[source_id] = {"raw": 0, "filtered": 0, "final": 0, "status": "missing"}
            continue

        raw_count = len(examples)

        # Quality filter
        min_resp = source.get("min_response", 30)
        examples = quality_filter(examples, min_think_tokens=source["min_think"],
                                  min_response_tokens=min_resp)
        filtered_count = len(examples)

        # Subsample
        cap = cap_override or source["cap"]
        if cap and len(examples) > cap:
            examples = random.sample(examples, cap)

        # Normalize format
        examples = [
            normalize_format(ex, source_id, source["license"])
            for ex in examples
        ]
        examples = [ex for ex in examples if ex is not None]

        print(f"  {source_id}: {raw_count} raw → {filtered_count} filtered → {len(examples)} final")
        stats[source_id] = {
            "raw": raw_count,
            "filtered": filtered_count,
            "final": len(examples),
            "status": "ok",
        }

        all_examples.extend(examples)

    # Shuffle
    random.shuffle(all_examples)

    # Write output
    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    source_counts = Counter(ex.get("source", "?") for ex in all_examples)
    print(f"\nTotal: {len(all_examples)} examples → {out_path}")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")

    return all_examples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare upstream thinking data for Teapot compose"
    )
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--cap", type=int, help="Override per-source cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reasoning", action="store_true",
                        help="No-op: data already has <think> traces")
    parser.add_argument("--format", help="Chat template (informational only)")
    args = parser.parse_args()

    prepare(output=args.output, cap_override=args.cap, seed=args.seed)


if __name__ == "__main__":
    main()
