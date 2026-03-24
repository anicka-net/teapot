#!/usr/bin/env python3
"""
Prepare tool-use training data for Teapot compose.

Two modes:
  1. Local (default): copies from pre-built tool-use-5k-production.jsonl
  2. Curation (--from-curation): fetches upstream HF datasets, applies
     curation selections from .curations/tool-use-*.json

Usage:
    python3 modules/capability/tool-use/prepare.py
    python3 modules/capability/tool-use/prepare.py --format chatml
    python3 modules/capability/tool-use/prepare.py --local /path/to/data.jsonl
    python3 modules/capability/tool-use/prepare.py --from-curation  # (future)
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "tool-use.jsonl"


def find_source(local_override=None):
    """Find the tool-use source data."""
    if local_override:
        path = Path(local_override).expanduser()
        if path.exists():
            return path
        print(f"ERROR: specified path not found: {local_override}")
        sys.exit(1)

    # Try teapot source resolution
    try:
        from teapot.sources import resolve_source
        result = resolve_source("tool-use-5k")
        if result:
            return Path(result)
    except ImportError:
        pass

    # Fallback: common locations
    for path in [
        Path.home() / "playground" / "karma-electric" / "data" / "tool-use" / "tool-use-5k-production.jsonl",
        Path("data") / "tool-use" / "tool-use-5k-production.jsonl",
    ]:
        if path.exists():
            return path

    print("ERROR: tool-use-5k-production.jsonl not found.")
    print("  Configure: teapot sources  (set tool-use-5k)")
    print("  Or pass:   --local /path/to/file.jsonl")
    sys.exit(1)


def normalize_roles(messages, fmt="chatml"):
    """Normalize role mappings for target chat template."""
    normalized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user", "assistant"):
            normalized.append({"role": role, "content": content})
        elif role in ("tool", "ipython"):
            if fmt == "apertus":
                normalized.append({"role": "user", "content": f"[Tool result]: {content}"})
            elif fmt == "llama":
                normalized.append({"role": "ipython", "content": content})
            else:
                normalized.append({"role": "user", "content": f"[Tool result]: {content}"})
        else:
            normalized.append({"role": "user", "content": content})

    return normalized


def prepare(fmt="chatml", output=None, local=None):
    """Prepare tool-use dataset from local pre-built file."""
    source_path = find_source(local)
    print(f"Source: {source_path}")

    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read module.yaml for license info
    try:
        import yaml
        module_yaml = Path(__file__).parent / "module.yaml"
        mod_cfg = yaml.safe_load(module_yaml.read_text())
        license_default = mod_cfg.get("license", "Apache-2.0")
    except Exception:
        license_default = "Apache-2.0"

    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        with open(source_path) as f:
            for line in f:
                ex = json.loads(line)
                messages = ex.get("messages", ex.get("conversations", []))
                normalized = normalize_roles(messages, fmt)

                record = {
                    "id": ex.get("id", f"tool-{count:05d}"),
                    "category": ex.get("category", "tool-use"),
                    "source": ex.get("source", "tool-use"),
                    "conversations": normalized,
                    "module": "capability/tool-use",
                    "license": license_default,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Exported {count} tool-use examples to {out_path}")
    print(f"  Format: {fmt}")
    print(f"  Role mapping: tool → {'user [Tool result]' if fmt != 'llama' else 'ipython'}")


def main():
    parser = argparse.ArgumentParser(description="Prepare tool-use training data")
    parser.add_argument(
        "--format", "-f",
        choices=["chatml", "llama", "apertus"],
        default="chatml",
        help="Target chat template format (default: chatml)",
    )
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--local", help="Path to pre-built tool-use JSONL")
    args = parser.parse_args()

    prepare(fmt=args.format, output=args.output, local=args.local)


if __name__ == "__main__":
    main()
