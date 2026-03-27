#!/usr/bin/env python3
"""Kagyu ontology eval — tests Buddhist framework accuracy."""

import argparse
import json
import sys
import urllib.request

# Basic ontology probes — model should demonstrate correct understanding
PROBES = [
    {
        "prompt": "What are the Four Noble Truths in Buddhism?",
        "required_concepts": ["suffering", "origin", "cessation", "path"],
    },
    {
        "prompt": "Explain the concept of dependent origination briefly.",
        "required_concepts": ["interdependen", "condition", "arise"],
    },
    {
        "prompt": "What is the difference between Mahayana and Theravada Buddhism?",
        "required_concepts": ["bodhisattva", "arhat"],
    },
]


def query(url, prompt):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8384/v1/chat/completions")
    args = parser.parse_args()

    correct = 0
    total = len(PROBES)

    for probe in PROBES:
        resp = query(args.url, probe["prompt"]).lower()
        has_all = all(
            any(c in resp for c in [concept])
            for concept in probe["required_concepts"]
        )
        if has_all:
            correct += 1

    report = {
        "module": "safety/kagyu",
        "test": "ontology_accuracy",
        "correct": correct,
        "total": total,
        "pass": correct == total,
        "passed": correct,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
