#!/usr/bin/env python3
"""
Teapot source resolution — decouple modules from data locations.

Modules declare what data they need (source IDs). Users declare where
that data lives (source map). The prepare script asks for the resolved
path without knowing the location.

Resolution order:
1. CLI override (--source id=path)
2. Environment variable (TEAPOT_SOURCE_<ID>=path)
3. Source map file (teapot.sources.yaml)
4. Module.yaml defaults (default_path, default_repo)

Usage in prepare scripts:
    from teapot.sources import resolve_source
    path = resolve_source("karma-electric-db")
"""

import os
from pathlib import Path

import yaml

from teapot.root import find_root

_source_map = None
_cli_overrides = {}


def set_cli_overrides(overrides: dict):
    """Set CLI-provided source overrides (called from compose)."""
    global _cli_overrides
    _cli_overrides = overrides or {}


def _load_source_map():
    """Load teapot.sources.yaml from project root."""
    global _source_map
    if _source_map is not None:
        return _source_map

    root = find_root()
    sources_path = root / "teapot.sources.yaml"

    if sources_path.exists():
        with open(sources_path) as f:
            _source_map = yaml.safe_load(f) or {}
    else:
        _source_map = {}

    return _source_map


def _env_key(source_id: str) -> str:
    """Convert source ID to environment variable name."""
    return "TEAPOT_SOURCE_" + source_id.upper().replace("-", "_").replace("/", "_")


def resolve_source(source_id: str, module_yaml: dict = None) -> str:
    """Resolve a source ID to a local path.

    Args:
        source_id: Declared source identifier (e.g. "karma-electric-db")
        module_yaml: Optional module.yaml dict for fallback defaults

    Returns:
        Resolved path as string, or None if not found.

    Resolution order:
        1. CLI override
        2. Environment variable
        3. Source map file
        4. Module.yaml default_path
        5. Module.yaml default_repo (HuggingFace — fetch if needed)
    """
    # 1. CLI override
    if source_id in _cli_overrides:
        path = Path(_cli_overrides[source_id]).expanduser()
        if path.exists():
            return str(path)

    # 2. Environment variable
    env_val = os.environ.get(_env_key(source_id))
    if env_val:
        path = Path(env_val).expanduser()
        if path.exists():
            return str(path)

    # 3. Source map file
    source_map = _load_source_map()
    if source_id in source_map:
        val = source_map[source_id]
        if isinstance(val, str):
            # Could be a path or "hf:repo/name"
            if val.startswith("hf:"):
                return _fetch_from_hf(val[3:], source_id)
            path = Path(val).expanduser()
            if path.exists():
                return str(path)
        elif isinstance(val, dict):
            if "path" in val:
                path = Path(val["path"]).expanduser()
                if path.exists():
                    return str(path)
            if "repo" in val:
                return _fetch_from_hf(val["repo"], source_id, val.get("file"))

    # 4. Module.yaml defaults
    if module_yaml:
        sources = module_yaml.get("data", {}).get("sources", [])
        for src in sources:
            if src.get("id") == source_id or len(sources) == 1:
                # Try default_path
                default_path = src.get("default_path")
                if default_path:
                    path = Path(default_path).expanduser()
                    if path.exists():
                        return str(path)
                    # Try relative to teapot root
                    root_path = find_root() / default_path
                    if root_path.exists():
                        return str(root_path)

                # Try default_repo (HuggingFace)
                default_repo = src.get("default_repo")
                if default_repo:
                    return _fetch_from_hf(
                        default_repo, source_id, src.get("default_file")
                    )

    return None


def _fetch_from_hf(repo: str, source_id: str, filename: str = None) -> str:
    """Fetch a source from HuggingFace, cache locally."""
    try:
        from teapot.data_fetch import fetch_source
        source_spec = {"type": "hf", "repo": repo}
        if filename:
            source_spec["file"] = filename
        return fetch_source(source_spec, module_name=source_id)
    except ImportError:
        print(f"WARNING: huggingface_hub not installed. Cannot fetch {repo}")
        print(f"  pip install teapot-ai[fetch]")
        print(f"  Or set {_env_key(source_id)}=/path/to/local/data")
        return None
    except Exception as e:
        print(f"WARNING: Failed to fetch {repo}: {e}")
        return None


def list_sources(module_dir: Path = None):
    """List all declared sources and their resolution status."""
    root = find_root()
    source_map = _load_source_map()

    print("Source resolution:")
    print()

    if source_map:
        print("From teapot.sources.yaml:")
        for sid, val in source_map.items():
            resolved = resolve_source(sid)
            status = "OK" if resolved else "NOT FOUND"
            print(f"  [{'+' if resolved else 'X'}] {sid}: {val} → {status}")
        print()

    # Check env vars
    env_sources = {k: v for k, v in os.environ.items()
                   if k.startswith("TEAPOT_SOURCE_")}
    if env_sources:
        print("From environment:")
        for k, v in env_sources.items():
            print(f"  {k}={v}")
        print()

    # Scan modules for declared sources
    modules_dir = root / "modules"
    if modules_dir.exists():
        print("Declared in modules:")
        for yaml_path in sorted(modules_dir.rglob("module.yaml")):
            with open(yaml_path) as f:
                mod = yaml.safe_load(f)
            mod_name = mod.get("name", "?")
            sources = mod.get("data", {}).get("sources", [])
            for src in sources:
                sid = src.get("id", f"{mod_name}-data")
                resolved = resolve_source(sid, mod)
                status = "OK" if resolved else "NEEDS CONFIG"
                print(f"  [{'+' if resolved else '?'}] {mod_name} → {sid}: {status}")


def main():
    """CLI: teapot sources"""
    import argparse
    parser = argparse.ArgumentParser(description="Manage data sources")
    parser.add_argument("--list", action="store_true", help="List all sources")
    parser.add_argument("--check", metavar="ID", help="Check one source")
    args = parser.parse_args()

    if args.list or not args.check:
        list_sources()
    elif args.check:
        result = resolve_source(args.check)
        if result:
            print(f"{args.check} → {result}")
        else:
            print(f"{args.check} → NOT FOUND")
            print(f"Set {_env_key(args.check)}=/path/to/data")


if __name__ == "__main__":
    main()
