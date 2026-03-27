#!/usr/bin/env python3
"""Validate KE thinking export shape."""

import json
import re
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: test_format.py DATA.jsonl")
        return 2

    total = 0
    with_think = 0
    seen_ids = set()
    duplicate_ids = set()

    with open(sys.argv[1], encoding="utf-8") as f:
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
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
