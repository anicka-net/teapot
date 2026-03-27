#!/usr/bin/env python3
"""Dzongkha identification eval — tests script/language discrimination."""

import argparse
import json
import sys
import urllib.request

# The model should correctly identify or express uncertainty about these
PROBES = [
    {
        "text": "བཀྲ་ཤིས་བདེ་ལེགས།",
        "expected": "ambiguous",
        "note": "Common greeting — could be Dzongkha or Standard Tibetan",
    },
    {
        "text": "འབྲུག་རྒྱལ་ཁབ་ཀྱི་རྒྱལ་ས་ཐིམ་ཕུ་ཨིན།",
        "expected": "dzongkha",
        "note": "Contains Bhutan-specific markers (འབྲུག, ཐིམ་ཕུ, ཨིན)",
    },
    {
        "text": "What language is this: བོད་སྐད།",
        "expected": "tibetan",
        "note": "བོད་སྐད (Tibetan language) — Standard Tibetan self-reference",
    },
]

CORRECT_SIGNALS = {
    "ambiguous": ["ambiguous", "could be", "more context", "not certain",
                  "either", "both dzongkha and tibetan", "unclear"],
    "dzongkha": ["dzongkha", "bhutan"],
    "tibetan": ["tibetan", "standard tibetan", "bod skad"],
}


def query(url, prompt):
    payload = json.dumps({
        "messages": [{"role": "user", "content": f"Identify the language of this text: {prompt}"}],
        "max_tokens": 200, "temperature": 0.0,
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
        resp = query(args.url, probe["text"]).lower()
        signals = CORRECT_SIGNALS.get(probe["expected"], [])
        if any(sig in resp for sig in signals):
            correct += 1

    report = {
        "module": "lang/dzongkha",
        "test": "language_identification",
        "correct": correct,
        "total": total,
        "pass": correct >= 2,  # At least 2/3 correct
        "passed": correct,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
