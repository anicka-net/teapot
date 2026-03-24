#!/usr/bin/env python3
"""
Teapot Curation Cache — versioned, auditable dataset selection decisions.

When an LLM reviews examples (keep/move/delete), or a human scores
them, the decisions are stored as versioned JSON manifests. Compose
can then use these cached decisions instead of re-running expensive
review passes.

The flow:
    fetch → inspect/index → select (curation) → compose
                               ↓
                        .curations/*.json
                        (versioned, auditable)

Usage:
    teapot curate create --module safety/consequence --scorer human \\
        --input decisions.jsonl --version v1
    teapot curate list
    teapot curate show safety-consequence-v1
    teapot curate apply safety-consequence-v1 --output filtered.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from teapot.root import find_root

CURATIONS_DIR = find_root() / ".curations"


def resolve_curation_path(name, module=None):
    """Resolve a curation from explicit ref or legacy local name."""
    root = find_root()

    if ":" in name:
        tier, version = name.split(":", 1)
        if tier not in {"published", "local"} or not version:
            raise FileNotFoundError(
                f"Invalid curation reference '{name}'. Use published:VERSION or local:VERSION."
            )
        if tier == "local":
            candidates = [CURATIONS_DIR / f"{version}.json", CURATIONS_DIR / version]
        else:
            if not module:
                raise FileNotFoundError(
                    "Published curations require --module MODULE or a manifest with module metadata."
                )
            parts = module.split("/")
            module_dir = root / "modules" / "/".join(parts) / "curations"
            candidates = [module_dir / f"{version}.json", module_dir / f"{parts[-1]}-{version}.json"]
    else:
        candidates = [CURATIONS_DIR / f"{name}.json", CURATIONS_DIR / name]

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Curation not found: {name}")


def create_curation(module, version, scorer, decisions, scorer_version=None,
                     notes=None, publish=False):
    """Create a curation manifest from a list of decisions.

    Each decision: {"id": "example-id", "verdict": "KEEP|MOVE|DELETE|NEEDS_EDIT",
                     "score": float|null, "tags": [...], "reason": "..."}

    If publish=True, writes to the module directory (shared).
    Otherwise writes to .curations/ (local, gitignored).
    """
    if publish:
        parts = module.split("/")
        out_dir = find_root() / "modules" / "/".join(parts) / "curations"
    else:
        out_dir = CURATIONS_DIR

    out_dir.mkdir(exist_ok=True, parents=True)

    slug = module.replace("/", "-")
    if publish:
        filename = f"{version}.json"
    else:
        filename = f"{slug}-{version}.json"

    manifest = {
        "module": module,
        "version": version,
        "scorer": scorer,
        "scorer_version": scorer_version or datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "total_decisions": len(decisions),
        "summary": dict(Counter(d.get("verdict", "UNKNOWN") for d in decisions)),
        "decisions": decisions,
    }

    if notes:
        manifest["notes"] = notes

    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created curation: {path}")
    print(f"  Module: {module}")
    print(f"  Version: {version}")
    print(f"  Scorer: {scorer}")
    print(f"  Decisions: {len(decisions)}")
    for verdict, count in manifest["summary"].items():
        print(f"    {verdict}: {count}")

    return path


def list_curations():
    """List all curation manifests (published + local)."""
    root = find_root()
    manifests = []

    # Tier 1: published curations in module directories
    modules_dir = root / "modules"
    if modules_dir.exists():
        for path in sorted(modules_dir.rglob("curations/*.json")):
            manifests.append(("published", path))

    # Tier 2: local curations
    if CURATIONS_DIR.exists():
        for path in sorted(CURATIONS_DIR.glob("*.json")):
            manifests.append(("local", path))

    if not manifests:
        print("No curations found. Create one with: teapot curate create")
        return []

    print("Curations:")
    print()
    for tier, path in manifests:
        try:
            data = json.loads(path.read_text())
            module = data.get("module", "?")
            version = data.get("version", "?")
            scorer = data.get("scorer", "?")
            total = data.get("total_decisions", 0)
            summary = data.get("summary", {})
            ts = data.get("timestamp", "?")[:10]
            summary_str = ", ".join(f"{k}:{v}" for k, v in summary.items())
            tier_label = "published" if tier == "published" else "local"
            print(f"  [{tier_label}] {path.name}")
            print(f"    {module} {version} by {scorer} ({ts})")
            print(f"    {total} decisions: {summary_str}")
            print()
        except Exception as e:
            print(f"  {path.name} — ERROR: {e}")

    return manifests


def show_curation(name):
    """Show details of a curation manifest."""
    path = resolve_curation_path(name)
    data = json.loads(path.read_text())
    print(json.dumps(data, indent=2))


def apply_curation(name, data_path, output_path, verdict_filter=None, module=None):
    """Apply a curation manifest to filter a JSONL dataset.

    Keeps only examples with verdicts in verdict_filter (default: KEEP*).
    """
    path = resolve_curation_path(name, module=module)
    manifest = json.loads(path.read_text())
    decisions = {d["id"]: d for d in manifest.get("decisions", [])}

    if not verdict_filter:
        verdict_filter = {"KEEP", "KEEP_SECULAR", "KEEP_BUDDHIST"}

    kept = 0
    filtered = 0
    not_in_curation = 0

    with open(data_path) as inp, open(output_path, "w") as out:
        for line in inp:
            if not line.strip():
                continue
            ex = json.loads(line)
            ex_id = ex.get("id", "")

            if ex_id in decisions:
                verdict = decisions[ex_id].get("verdict", "")
                if verdict in verdict_filter:
                    out.write(line)
                    kept += 1
                else:
                    filtered += 1
            else:
                # Example not in curation — keep by default
                out.write(line)
                not_in_curation += 1

    print(f"Applied {name}:")
    print(f"  Kept: {kept}")
    print(f"  Filtered: {filtered}")
    print(f"  Not in curation (kept): {not_in_curation}")
    print(f"  Output: {output_path}")


def import_from_jsonl(input_path):
    """Import decisions from a JSONL file (one decision per line)."""
    decisions = []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            decisions.append(json.loads(line))
    return decisions


def import_from_db_notes(db_path, module):
    """Import curation decisions from training.db notes column.

    Reads examples where the notes column contains curation verdicts
    (e.g., from Sonnet review or manual editing).
    """
    import sqlite3
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(examples)").fetchall()
    }
    if "module" not in columns:
        conn.close()
        raise ValueError(
            "training.db examples table has no 'module' column; "
            "cannot safely import module-scoped curation from notes"
        )

    rows = conn.execute(
        "SELECT id, category, tier, notes FROM examples "
        "WHERE module = ? AND status = 'accepted' AND notes IS NOT NULL AND notes != ''",
        (module,),
    ).fetchall()

    decisions = []
    for eid, cat, tier, notes in rows:
        # Parse notes for verdicts
        verdict = "KEEP"
        tags = []
        if "NEEDS_EDIT" in notes.upper():
            verdict = "NEEDS_EDIT"
        elif "MOVE_BUDDHIST" in notes.upper():
            verdict = "MOVE_BUDDHIST"
        elif "DELETE" in notes.upper():
            verdict = "DELETE"
        elif "KEEP_SECULAR" in notes.upper():
            verdict = "KEEP_SECULAR"

        if cat:
            tags.append(cat)
        if tier:
            tags.append(f"tier:{tier}")

        decisions.append({
            "id": eid,
            "verdict": verdict,
            "score": None,
            "tags": tags,
            "reason": notes[:200] if notes else "",
        })

    conn.close()
    return decisions


def main():
    parser = argparse.ArgumentParser(description="Teapot curation cache")
    sub = parser.add_subparsers(dest="command")

    # create
    create_p = sub.add_parser("create", help="Create a curation manifest")
    create_p.add_argument("--module", required=True, help="Module name (e.g. safety/consequence)")
    create_p.add_argument("--version", required=True, help="Version label (e.g. v1)")
    create_p.add_argument("--scorer", required=True, help="Who scored (human, sonnet-4.6, etc.)")
    create_p.add_argument("--input", required=True, help="Decisions JSONL file")
    create_p.add_argument("--notes", default=None, help="Notes about this curation")
    create_p.add_argument("--publish", action="store_true",
                          help="Save to module directory (shared) instead of .curations/ (local)")

    # create-from-db
    db_p = sub.add_parser("create-from-db", help="Import curation from training.db notes")
    db_p.add_argument("--module", required=True)
    db_p.add_argument("--version", required=True)
    db_p.add_argument("--scorer", required=True)
    db_p.add_argument("--db", required=True, help="Path to training.db")
    db_p.add_argument("--publish", action="store_true",
                      help="Save to module directory (shared)")

    # list
    sub.add_parser("list", help="List all curations")

    # show
    show_p = sub.add_parser("show", help="Show curation details")
    show_p.add_argument("name", help="Curation name (e.g. safety-consequence-v1)")

    # apply
    apply_p = sub.add_parser("apply", help="Apply curation to filter data")
    apply_p.add_argument("name", help="Curation name")
    apply_p.add_argument("--data", required=True, help="Input JSONL")
    apply_p.add_argument("--output", "-o", required=True, help="Output JSONL")
    apply_p.add_argument("--module", help="Module name (required for published:VERSION)")
    apply_p.add_argument("--verdicts", default=None,
                         help="Comma-separated verdicts to keep (default: KEEP*)")

    args = parser.parse_args()

    if args.command == "create":
        decisions = import_from_jsonl(args.input)
        create_curation(args.module, args.version, args.scorer, decisions,
                        notes=args.notes, publish=args.publish)
    elif args.command == "create-from-db":
        decisions = import_from_db_notes(args.db, args.module)
        create_curation(args.module, args.version, args.scorer, decisions,
                        publish=args.publish)
    elif args.command == "list":
        list_curations()
    elif args.command == "show":
        try:
            show_curation(args.name)
        except FileNotFoundError as e:
            print(str(e))
            sys.exit(1)
    elif args.command == "apply":
        verdicts = set(args.verdicts.split(",")) if args.verdicts else None
        try:
            apply_curation(args.name, args.data, args.output, verdicts, module=args.module)
        except FileNotFoundError as e:
            print(str(e))
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
