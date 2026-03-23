#!/usr/bin/env python3
"""
Teapot lockfile — pin data sources for reproducible builds.

The lockfile records full SHA-256 hashes of every data source and
the composed output. Same lockfile + same sources = same training data.

Usage:
    python3 scripts/lockfile.py generate manifest.json
    python3 scripts/lockfile.py generate manifest.json --output teapot.lock
    python3 scripts/lockfile.py verify teapot.lock
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

TEAPOT_ROOT = Path(__file__).resolve().parents[1]


def hash_file(path):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def generate_lock(manifest_path, lock_path=None):
    """Generate a lockfile from a compose manifest."""
    manifest = json.loads(Path(manifest_path).read_text())

    lock = {
        "version": 1,
        "generated": datetime.now().isoformat(),
        "config": manifest.get("config", ""),
        "seed": manifest.get("seed", 42),
        "sources": {},
        "output_hash": manifest.get("output_hash", ""),
    }

    for module_name, module_info in manifest.get("modules", {}).items():
        source_path = Path(module_info["source"])
        entry = {
            "prepared_hash": module_info.get("integrity", ""),
            "examples_raw": module_info.get("examples_raw", 0),
            "examples_weighted": module_info.get("examples_weighted", 0),
            "weight": module_info.get("weight", 1.0),
        }

        # Record the source type and path
        if source_path.exists():
            entry["source_path"] = str(source_path)
            entry["source_type"] = "local"
        else:
            entry["source_path"] = str(source_path)
            entry["source_type"] = "missing"

        lock["sources"][module_name] = entry

    out = Path(lock_path) if lock_path else Path("teapot.lock")
    with open(out, "w") as f:
        json.dump(lock, f, indent=2)

    print(f"Generated {out} ({len(lock['sources'])} sources)")
    return lock


def verify_lock(lock_path):
    """Verify current data against a lockfile. Returns True if all match."""
    lock = json.loads(Path(lock_path).read_text())
    all_ok = True

    print(f"Verifying {lock_path} (generated {lock['generated']})")
    print(f"Config: {lock['config']}")
    print()

    for module_name, entry in lock.get("sources", {}).items():
        source_path = Path(entry.get("source_path", ""))
        expected_hash = entry.get("prepared_hash", "")

        if not source_path.exists():
            print(f"  [!] {module_name}: source file missing ({source_path})")
            all_ok = False
            continue

        if expected_hash:
            current_hash = hash_file(source_path)
            if current_hash == expected_hash:
                print(f"  [+] {module_name}: OK ({entry['examples_raw']} examples)")
            else:
                print(f"  [X] {module_name}: CHANGED")
                print(f"      expected: {expected_hash[:32]}...")
                print(f"      current:  {current_hash[:32]}...")
                all_ok = False
        else:
            print(f"  [-] {module_name}: no hash to verify")

    # Check output hash if we can find the output file
    output_hash = lock.get("output_hash", "")
    if output_hash:
        # Try to find the output file from the config
        config_path = lock.get("config", "")
        if config_path:
            try:
                import yaml
                cfg = yaml.safe_load(open(config_path))
                output_file = cfg.get("output", {})
                if isinstance(output_file, dict):
                    output_file = output_file.get("file", "train.jsonl")
                if Path(output_file).exists():
                    current = hash_file(output_file)
                    if current == output_hash:
                        print(f"\n  [+] Output: OK")
                    else:
                        print(f"\n  [X] Output: CHANGED")
                        all_ok = False
            except Exception:
                pass

    print()
    if all_ok:
        print("RESULT: All sources match lockfile")
    else:
        print("RESULT: Sources have changed — re-run compose or update lock")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Teapot lockfile management")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate lockfile from manifest")
    gen.add_argument("manifest", help="Manifest JSON path")
    gen.add_argument("--output", "-o", default="teapot.lock", help="Lockfile path")

    ver = sub.add_parser("verify", help="Verify sources against lockfile")
    ver.add_argument("lockfile", help="Lockfile path")

    args = parser.parse_args()

    if args.command == "generate":
        generate_lock(args.manifest, args.output)
    elif args.command == "verify":
        ok = verify_lock(args.lockfile)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
