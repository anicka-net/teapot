#!/usr/bin/env python3
"""
Teapot lockfile — pin data sources for reproducible builds.

The lockfile records full SHA-256 hashes of every data source and
the composed output. Same lockfile + same sources = same training data.
It also records a build identity for the config, Teapot source revision,
and relevant runtime package versions.

Usage:
    python3 scripts/lockfile.py generate manifest.json
    python3 scripts/lockfile.py generate manifest.json --output teapot.lock
    python3 scripts/lockfile.py verify teapot.lock
    python3 scripts/lockfile.py verify teapot.lock --build
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from teapot.provenance import collect_build_identity, identity_hash
from teapot.root import find_root
TEAPOT_ROOT = find_root()


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

    config_path = manifest.get("config", "")
    build_identity = collect_build_identity(config_path, TEAPOT_ROOT)

    lock = {
        "version": 1,
        "generated": datetime.now().isoformat(),
        "config": config_path,
        "seed": manifest.get("seed", 42),
        "sources": {},
        "output_file": manifest.get("output_file", ""),
        "output_hash": manifest.get("output_hash", ""),
        "build_identity": build_identity,
        "build_id": identity_hash(build_identity),
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
    print(f"Build ID: {lock['build_id']}")
    return lock


def verify_lock(lock_path, verify_build=False):
    """Verify current data against a lockfile. Returns True if all match.

    ``verify_build`` additionally checks the config, Teapot source revision,
    Python runtime, and installed training-stack package versions recorded at
    lock generation time. It is opt-in so existing source-only verification
    workflows remain backward compatible.
    """
    lock = json.loads(Path(lock_path).read_text())
    all_ok = True

    print(f"Verifying {lock_path} (generated {lock['generated']})")
    print(f"Config: {lock['config']}")
    if lock.get("build_id"):
        print(f"Build ID: {lock['build_id']}")
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

    # Check the output-artifact hash. Any failure to verify is a FAILURE, not a
    # silent skip — the old `except: pass` let a run still report "all match".
    output_hash = lock.get("output_hash", "")
    if output_hash:
        # Prefer the actual output path recorded at compose time (robust to
        # --output overrides); fall back to deriving it from the config only for
        # old locks that predate output_file.
        output_file = lock.get("output_file", "")
        if not output_file:
            config_path = lock.get("config", "")
            try:
                import yaml
                cfg = yaml.safe_load(open(config_path)) if config_path else {}
                of = cfg.get("output", {})
                output_file = of.get("file", "") if isinstance(of, dict) else (of or "")
            except Exception as e:
                print(f"\n  [X] Output: cannot read config to locate output ({e})")
                all_ok = False

        if output_file and Path(output_file).exists():
            if hash_file(output_file) == output_hash:
                print(f"\n  [+] Output: OK")
            else:
                print(f"\n  [X] Output: CHANGED")
                all_ok = False
        elif output_file:
            print(f"\n  [X] Output: file not found ({output_file}) — cannot verify")
            all_ok = False
        else:
            print(f"\n  [X] Output: could not resolve output file — cannot verify")
            all_ok = False

    if verify_build:
        expected_identity = lock.get("build_identity")
        if not expected_identity:
            print("\n  [X] Build: lockfile has no build identity")
            all_ok = False
        else:
            current_identity = collect_build_identity(lock.get("config", ""), TEAPOT_ROOT)
            current_id = identity_hash(current_identity)
            expected_id = lock.get("build_id") or identity_hash(expected_identity)
            if current_id == expected_id and current_identity == expected_identity:
                print("\n  [+] Build: identity matches")
            else:
                print("\n  [X] Build: identity changed")
                print(f"      expected: {expected_id}")
                print(f"      current:  {current_id}")
                all_ok = False

    print()
    if all_ok:
        if verify_build:
            print("RESULT: Sources, output, and build identity match lockfile")
        else:
            print("RESULT: All sources match lockfile")
    else:
        print("RESULT: Lock verification failed")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Teapot lockfile management")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate lockfile from manifest")
    gen.add_argument("manifest", help="Manifest JSON path")
    gen.add_argument("--output", "-o", default="teapot.lock", help="Lockfile path")

    ver = sub.add_parser("verify", help="Verify sources against lockfile")
    ver.add_argument("lockfile", help="Lockfile path")
    ver.add_argument("--build", action="store_true",
                     help="Also verify config, Teapot source, and runtime identity")

    args = parser.parse_args()

    if args.command == "generate":
        generate_lock(args.manifest, args.output)
    elif args.command == "verify":
        ok = verify_lock(args.lockfile, verify_build=args.build)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
