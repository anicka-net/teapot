#!/usr/bin/env python3
"""Validate KE thinking export shape."""

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "ke-thinking.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Validate KE thinking export shape")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--url", default="", help="Ignored for local format validation")
    args = parser.parse_args()

    total = 0
    with_think = 0
    seen_ids = set()
    duplicate_ids = set()

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            ex = json.loads(line)
            ex_id = ex.get("id")
            if ex_id in seen_ids:
                duplicate_ids.add(ex_id)
            seen_ids.add(ex_id)
            for msg in ex.get("conversations", []):
                if msg.get("role") == "assistant" and re.search(r"<think>.*?</think>", msg.get("content", ""), re.DOTALL):
                    with_think += 1
                    break

    pct = 0 if total == 0 else round((with_think * 100.0) / total, 2)
    report = {
        "module": "safety/ke-thinking",
        "total_examples": total,
        "has_think_pct": pct,
        "duplicate_ids": len(duplicate_ids),
        "pass": total > 0 and pct >= 95.0 and not duplicate_ids,
        "passed": with_think,
        "total": total,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
