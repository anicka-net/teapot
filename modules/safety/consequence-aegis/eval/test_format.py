#!/usr/bin/env python3
"""Validate consequence-aegis export shape."""

import json
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: test_format.py DATA.jsonl")
        return 2

    total = 0
    seen_ids = set()
    duplicate_ids = set()
    missing_assistant = 0

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
            if not any(msg.get("role") == "assistant" for msg in ex.get("conversations", [])):
                missing_assistant += 1

    report = {
        "module": "safety/consequence-aegis",
        "total_examples": total,
        "duplicate_ids": len(duplicate_ids),
        "missing_assistant": missing_assistant,
        "pass": total > 0 and not duplicate_ids and missing_assistant == 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
