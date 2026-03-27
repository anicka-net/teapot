#!/usr/bin/env python3
"""Validate reward-evaluator dataset structure."""

import argparse
import json
from pathlib import Path
import sys

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "reward-evaluator.jsonl"


REWARD_CATS = {
    "reward-evaluation",
    "reward-evaluation-v5",
    "reward-evaluation-style-variant",
}


def validate_example(ex, idx):
    issues = []
    if ex.get("task") != "reward-evaluation":
        issues.append(f"Example {idx} ({ex.get('id', '?')}): missing reward-evaluation task")
    if ex.get("category") not in REWARD_CATS:
        issues.append(f"Example {idx} ({ex.get('id', '?')}): invalid reward category")

    convs = ex.get("conversations", [])
    roles = [m.get("role") for m in convs]
    if roles != ["system", "user", "assistant"]:
        issues.append(f"Example {idx} ({ex.get('id', '?')}): expected system/user/assistant turns")

    for msg in convs:
        if not msg.get("content", "").strip():
            issues.append(f"Example {idx} ({ex.get('id', '?')}): empty {msg.get('role')} content")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate reward-evaluator module data")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="Reward-evaluator JSONL")
    parser.add_argument("--url", default="", help="Ignored for local format validation")
    args = parser.parse_args()

    issues = []
    total = 0
    with open(args.input) as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            issues.extend(validate_example(json.loads(line), i))

    report = {
        "module": "capability/reward-evaluator",
        "eval_type": "format",
        "total": total,
        "passed": total if not issues else 0,
        "clean": not issues,
        "pass": not issues,
        "issues": issues[:20],
    }

    print(json.dumps(report, indent=2))
    sys.exit(0 if not issues else 1)


if __name__ == "__main__":
    main()
