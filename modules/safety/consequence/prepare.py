#!/usr/bin/env python3
"""
Prepare the Karma Electric consequence reasoning dataset.

Exports from training.db with tier/role filtering, outputs JSONL
ready for Teapot compose.

Usage:
    python3 modules/safety/consequence/prepare.py
    python3 modules/safety/consequence/prepare.py --tier secular  # secular only
    python3 modules/safety/consequence/prepare.py --tier buddhist # buddhist only
    python3 modules/safety/consequence/prepare.py --output data/consequence.jsonl
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Source of truth
KE_DB = Path(__file__).resolve().parents[3] / ".." / "karma-electric" / "data" / "training.db"
# Fallback paths
KE_DB_FALLBACKS = [
    Path.home() / "playground" / "karma-electric" / "data" / "training.db",
    Path("data") / "training.db",
]

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "consequence.jsonl"


def find_db():
    """Find training.db from multiple possible locations."""
    for path in [KE_DB] + KE_DB_FALLBACKS:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    print("ERROR: training.db not found. Searched:")
    for p in [KE_DB] + KE_DB_FALLBACKS:
        print(f"  {p.resolve()}")
    sys.exit(1)


def get_system_prompt(conn, prompt_id):
    """Get system prompt content by ID."""
    row = conn.execute(
        "SELECT content FROM system_prompts WHERE id = ?", (prompt_id,)
    ).fetchone()
    if not row:
        print(f"WARNING: system prompt '{prompt_id}' not found")
        return None
    return row[0]


def prepare(tier=None, output=None, include_reward_eval=False, reasoning=False):
    """Export consequence reasoning examples from training.db."""
    db_path = find_db()
    print(f"Source: {db_path}")

    conn = sqlite3.connect(str(db_path))

    # Get system prompts
    v4_prompt = get_system_prompt(conn, "v4")
    reward_prompt = get_system_prompt(conn, "reward-evaluator-v1")

    # Build query
    conditions = ["status = 'accepted'"]
    params = []

    if not include_reward_eval:
        conditions.append("role = 'conversational'")

    # Default to secular tier (use --tier buddhist or --all for others)
    if tier:
        conditions.append("tier = ?")
        params.append(tier)
    else:
        conditions.append("tier = 'secular'")

    query = f"SELECT id, category, source, conversations, tier, reasoning FROM examples WHERE {' AND '.join(conditions)} ORDER BY category, id"
    rows = conn.execute(query, params).fetchall()

    # Reward-eval categories (need different system prompt)
    reward_cats = {
        "reward-evaluation",
        "reward-evaluation-v5",
        "reward-evaluation-style-variant",
    }

    # Prepare output
    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reasoning_count = 0
    reasoning_missing = 0
    examples = []
    for eid, cat, source, convs_json, ex_tier, reasoning_text in rows:
        convs = json.loads(convs_json)

        # Set appropriate system prompt
        if cat in reward_cats:
            prompt = reward_prompt
        else:
            prompt = v4_prompt

        # Replace or insert system prompt
        if prompt:
            convs = [m for m in convs if m.get("role") != "system"]
            convs.insert(0, {"role": "system", "content": prompt})

        # Prepend reasoning trace as <think> block to first assistant message
        if reasoning and reasoning_text:
            for i, msg in enumerate(convs):
                if msg.get("role") == "assistant":
                    convs[i]["content"] = f"<think>{reasoning_text}</think>\n\n{msg['content']}"
                    reasoning_count += 1
                    break
        elif reasoning and not reasoning_text:
            reasoning_missing += 1

        example = {
            "id": eid,
            "category": cat,
            "source": source,
            "tier": ex_tier,
            "conversations": convs,
            # Teapot metadata
            "module": "safety/consequence",
            "license": "MIT",  # KE training data is MIT
        }
        examples.append(example)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter

    tiers = Counter(ex["tier"] for ex in examples)
    print(f"Exported {len(examples)} examples to {out_path}")
    for t, c in sorted(tiers.items()):
        print(f"  {t}: {c}")
    if reasoning:
        print(f"  reasoning traces: {reasoning_count} included, {reasoning_missing} missing")

    conn.close()
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare KE consequence reasoning dataset"
    )
    parser.add_argument(
        "--tier",
        choices=["secular", "buddhist"],
        help="Filter by tier (default: all conversational)",
    )
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument(
        "--include-reward-eval",
        action="store_true",
        help="Include reward-evaluator examples (for 8B training only)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Include reasoning traces as <think> blocks",
    )
    args = parser.parse_args()

    prepare(
        tier=args.tier,
        output=args.output,
        include_reward_eval=args.include_reward_eval,
        reasoning=args.reasoning,
    )


if __name__ == "__main__":
    main()
