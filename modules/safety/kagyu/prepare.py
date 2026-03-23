#!/usr/bin/env python3
"""
Prepare the Kagyu Buddhist ethics dataset.

Exports Buddhist-tier examples from training.db for the optional
contemplative ethics layer.

Usage:
    python3 modules/safety/kagyu/prepare.py
    python3 modules/safety/kagyu/prepare.py --output data/kagyu.jsonl
    python3 modules/safety/kagyu/prepare.py --local /path/to/training.db
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "kagyu.jsonl"


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

    conn = sqlite3.connect(str(db_path))
    v4 = conn.execute(
        "SELECT content FROM system_prompts WHERE id = 'v4'"
    ).fetchone()[0]

    rows = conn.execute(
        "SELECT id, category, source, conversations FROM examples "
        "WHERE status = 'accepted' AND tier = 'buddhist' AND role = 'conversational' "
        "ORDER BY category, id"
    ).fetchall()

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for eid, cat, source, convs_json in rows:
            convs = json.loads(convs_json)
            convs = [m for m in convs if m.get("role") != "system"]
            convs.insert(0, {"role": "system", "content": v4})

            record = {
                "id": eid,
                "category": cat,
                "source": source,
                "tier": "buddhist",
                "conversations": convs,
                "module": "safety/kagyu",
                "license": "Apache-2.0",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} Buddhist examples to {out_path}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare Kagyu ethics dataset")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--local", help="Path to training.db")
    args = parser.parse_args()
    prepare(output=args.output, local=args.local)


if __name__ == "__main__":
    main()
