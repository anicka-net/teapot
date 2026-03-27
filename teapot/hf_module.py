#!/usr/bin/env python3
"""
Helpers for HuggingFace-backed Teapot modules.

Cache is an optimization, not the module contract. A module still declares
its upstream source identity in module.yaml and resolves it through Teapot's
source system.
"""

import hashlib
import json
from pathlib import Path

from teapot.sources import resolve_source


def get_source_config(module_yaml: dict, source_id: str) -> dict:
    """Return a declared source config by ID."""
    for src in module_yaml.get("data", {}).get("sources", []):
        if src.get("id") == source_id:
            return src
    raise KeyError(f"Unknown source '{source_id}'")


def resolve_hf_source_path(module_yaml: dict, source_id: str) -> Path | None:
    """Resolve a declared HF-backed source to a local path."""
    resolved = resolve_source(source_id, module_yaml)
    return Path(resolved) if resolved else None


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stable_example_id(prefix: str, payload) -> str:
    """Build a deterministic ID from normalized payload content."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"
