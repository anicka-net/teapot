#!/usr/bin/env python3
"""
Prepare the KE reward evaluator dataset.

Exports reward-evaluation examples from training.db with the
reward-evaluator-v1 system prompt. These train the model to score
responses on 6 dimensions.

Usage:
    python3 modules/capability/reward-evaluator/prepare.py
    python3 modules/capability/reward-evaluator/prepare.py --output data/reward-eval.jsonl
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "reward-evaluator.jsonl"


def sqlite_readonly_uri(db_path):
    """Build a readonly, immutable SQLite URI for external source DBs."""
    return f"file:{db_path}?mode=ro&immutable=1"


def find_db(local_override=None):
    if local_override:
        path = Path(local_override).expanduser()
        if path.exists():
            return path
        print(f"ERROR: specified path not found: {local_override}")
        sys.exit(1)

    try:
        from teapot.sources import resolve_source
        result = resolve_source("karma-electric-db")
        if result:
            return Path(result)
    except ImportError:
        pass

    for path in [
        Path.home() / "playground" / "karma-electric" / "data" / "training.db",
        Path("data") / "training.db",
    ]:
        if path.resolve().exists():
            return path.resolve()

    print("ERROR: training.db not found.")
    print("  Configure: teapot sources  (set karma-electric-db)")
    print("  Or pass:   --local /path/to/training.db")
    sys.exit(1)


def prepare(output=None, local=None):
    db_path = find_db(local)
    print(f"Source: {db_path}")

    conn = sqlite3.connect(sqlite_readonly_uri(db_path), uri=True)

    # Get the reward-evaluator system prompt
    row = conn.execute(
        "SELECT content FROM system_prompts WHERE id = 'reward-evaluator-v1'"
    ).fetchone()
    if not row:
        print("ERROR: system prompt 'reward-evaluator-v1' not found in DB")
        sys.exit(1)
    reward_prompt = row[0]

    # Query reward-evaluator examples
    cursor = conn.execute(
        "SELECT id, category, source, conversations FROM examples "
        "WHERE status = 'accepted' AND role = 'reward-evaluator' "
        "ORDER BY category, id"
    )

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for eid, cat, source, convs_json in cursor:
        convs = json.loads(convs_json)

        # Set reward-evaluator system prompt
        convs = [m for m in convs if m.get("role") != "system"]
        convs.insert(0, {"role": "system", "content": reward_prompt})

        examples.append({
            "id": eid,
            "category": cat,
            "source": source,
            "task": "reward-evaluation",
            "conversations": convs,
            "module": "capability/reward-evaluator",
            "license": "Apache-2.0",
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    cats = Counter(ex["category"] for ex in examples)
    print(f"Exported {len(examples)} examples to {out_path}")
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")

    conn.close()
    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare KE reward evaluator dataset")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--local", help="Path to training.db")
    args = parser.parse_args()
    prepare(output=args.output, local=args.local)


if __name__ == "__main__":
    main()
