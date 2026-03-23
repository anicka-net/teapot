#!/usr/bin/env python3
"""
Validate module.yaml files against the Teapot schema.

Usage:
    python3 scripts/validate_module.py modules/safety/consequence/module.yaml
    python3 scripts/validate_module.py --all
"""

import argparse
import json
import sys
from pathlib import Path

TEAPOT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TEAPOT_ROOT / "schemas" / "module.schema.json"


def validate_one(yaml_path):
    """Validate a single module.yaml. Returns (ok, errors)."""
    import yaml
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())

    try:
        data = yaml.safe_load(open(yaml_path))
    except Exception as e:
        return False, [f"YAML parse error: {e}"]

    if not data:
        return False, ["Empty YAML"]

    errors = []
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        errors.append(f"{e.json_path}: {e.message}")
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")

    return len(errors) == 0, errors


def find_all_modules():
    """Find all module.yaml files in the modules/ tree."""
    modules_dir = TEAPOT_ROOT / "modules"
    return sorted(modules_dir.rglob("module.yaml"))


def main():
    parser = argparse.ArgumentParser(description="Validate module.yaml files")
    parser.add_argument("path", nargs="?", help="Path to module.yaml")
    parser.add_argument("--all", action="store_true", help="Validate all modules")
    args = parser.parse_args()

    if not args.all and not args.path:
        parser.print_help()
        sys.exit(1)

    if args.all:
        paths = find_all_modules()
    else:
        paths = [Path(args.path)]

    all_ok = True
    for path in paths:
        ok, errors = validate_one(path)
        name = str(path.relative_to(TEAPOT_ROOT))
        if ok:
            print(f"  [+] {name}")
        else:
            print(f"  [X] {name}")
            for e in errors:
                print(f"      {e}")
            all_ok = False

    print()
    if all_ok:
        print(f"OK: {len(paths)} module(s) valid")
    else:
        print("FAIL: validation errors found")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
