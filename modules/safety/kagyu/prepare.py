#!/usr/bin/env python3
"""
Prepare the Kagyu Buddhist ethics dataset.

Source of truth: anicka/karma-electric-dataset (buddhist-conversational split).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teapot.hf_module import (
    get_source_config,
    load_jsonl,
    resolve_hf_source_path,
)

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "kagyu.jsonl"
MODULE_YAML = yaml.safe_load((Path(__file__).parent / "module.yaml").read_text())


def prepare(output=None, local_path=None):
    source_cfg = get_source_config(MODULE_YAML, "ke-buddhist-conversational")
    path = Path(local_path).expanduser() if local_path else resolve_hf_source_path(MODULE_YAML, "ke-buddhist-conversational")

    if not path:
        print("ERROR: Could not resolve ke-buddhist-conversational source.")
        print("  Set up teapot.sources.yaml or ensure default_path exists.")
        sys.exit(1)
    if not path.exists():
        print(f"ERROR: source path not found: {path}")
        sys.exit(1)

    print(f"Source: {path}")
    raw = load_jsonl(path)

    license_id = source_cfg.get("license", "Apache-2.0")

    examples = []
    for ex in raw:
        ex["module"] = "safety/kagyu"
        ex["license"] = license_id
        examples.append(ex)

    examples.sort(key=lambda x: (x.get("category", ""), x.get("id", "")))

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    cats = Counter(ex.get("category", "?") for ex in examples)
    print(f"Exported {len(examples)} examples to {out_path}")
    print(f"  Top categories: {', '.join(f'{c}({n})' for c, n in cats.most_common(5))}")

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare Kagyu ethics dataset")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--reasoning", action="store_true", help="No-op: data already has <think> traces")
    parser.add_argument("--format", help="Chat template (informational only)")
    parser.add_argument("--local", help="Local JSONL path (overrides source resolution)")
    args = parser.parse_args()
    prepare(output=args.output, local_path=args.local)


if __name__ == "__main__":
    main()
