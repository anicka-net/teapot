#!/usr/bin/env python3
"""
Teapot Data Fetcher — unified data retrieval for module prepare scripts.

Handles three source types:
  1. local  — path on disk (absolute or relative to module dir)
  2. hf     — HuggingFace dataset (repo/name, optional split)
  3. url    — direct download

All fetched data is cached in .cache/<module-name>/ to avoid re-downloading.
Integrity is verified via SHA-256 when specified.

Usage as library:
    from scripts.data_fetch import fetch_source
    path = fetch_source(source_config, module_name="safety/consequence")

Usage as CLI:
    python3 scripts/data_fetch.py --type hf --repo anicka/karma-electric-dataset --file sft/train-8b-v10.3.jsonl
    python3 scripts/data_fetch.py --type local --path ~/playground/karma-electric/data/training.db
    python3 scripts/data_fetch.py --type url --url https://example.com/data.jsonl
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from teapot.root import find_root
CACHE_DIR = find_root() / ".cache"


def _cache_path(module_name, filename):
    """Get cache path for a module's data file."""
    safe_module = module_name.replace("/", "_")
    cache = CACHE_DIR / safe_module
    cache.mkdir(parents=True, exist_ok=True)
    return cache / filename


def _sha256(filepath):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_cached(cache_file, integrity=None):
    """Check if file is already cached and optionally verify integrity."""
    if not cache_file.exists():
        return False
    if integrity:
        actual = _sha256(cache_file)
        if actual != integrity:
            print(f"  Cache integrity mismatch: expected {integrity[:16]}..., got {actual[:16]}...")
            return False
    return True


def fetch_local(path, module_name, filename=None, integrity=None):
    """Fetch from local filesystem. Returns resolved path."""
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        # Try relative to home
        home_path = Path.home() / path
        if home_path.exists():
            resolved = home_path.resolve()
        else:
            print(f"ERROR: Local path not found: {path}")
            print(f"  Tried: {Path(path).resolve()}")
            print(f"  Tried: {home_path}")
            return None

    # For local files, we can either use directly or copy to cache
    # Use directly if no integrity check needed (faster, no duplication)
    if not integrity:
        print(f"  Local: {resolved}")
        return resolved

    # Copy to cache for integrity verification
    fname = filename or resolved.name
    cache_file = _cache_path(module_name, fname)

    if _is_cached(cache_file, integrity):
        print(f"  Cached: {cache_file}")
        return cache_file

    shutil.copy2(resolved, cache_file)
    print(f"  Copied to cache: {cache_file}")

    if integrity:
        actual = _sha256(cache_file)
        if actual != integrity:
            print(f"  WARNING: Integrity check failed!")
            print(f"    Expected: {integrity}")
            print(f"    Got:      {actual}")

    return cache_file


def fetch_hf(repo, module_name, file=None, split=None, integrity=None):
    """Fetch from HuggingFace. Returns path to cached file."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return None

    if file:
        # Download specific file
        fname = file.split("/")[-1]
        cache_file = _cache_path(module_name, fname)

        if _is_cached(cache_file, integrity):
            print(f"  Cached: {cache_file}")
            return cache_file

        print(f"  Downloading {repo}/{file} from HuggingFace...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=file,
                repo_type="dataset",
                local_dir=str(cache_file.parent),
            )
            # hf_hub_download puts it in a subdirectory, move to expected location
            downloaded_path = Path(downloaded)
            if downloaded_path != cache_file:
                shutil.move(str(downloaded_path), str(cache_file))
            print(f"  Downloaded: {cache_file}")
            return cache_file
        except Exception as e:
            print(f"  ERROR downloading from HF: {e}")
            return None

    elif split:
        # Download as dataset and export
        from datasets import load_dataset

        fname = f"{split}.jsonl"
        cache_file = _cache_path(module_name, fname)

        if _is_cached(cache_file, integrity):
            print(f"  Cached: {cache_file}")
            return cache_file

        print(f"  Loading dataset {repo} split={split} from HuggingFace...")
        try:
            ds = load_dataset(repo, split=split)
            ds.to_json(str(cache_file))
            print(f"  Downloaded: {cache_file} ({len(ds)} examples)")
            return cache_file
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            return None

    else:
        # Snapshot download (all files)
        cache_dir = _cache_path(module_name, "")
        print(f"  Downloading {repo} snapshot from HuggingFace...")
        try:
            path = snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=str(cache_dir / repo.replace("/", "_")),
            )
            print(f"  Downloaded: {path}")
            return Path(path)
        except Exception as e:
            print(f"  ERROR: {e}")
            return None


def fetch_url(url, module_name, filename=None, integrity=None):
    """Fetch from URL. Returns path to cached file."""
    import urllib.request

    fname = filename or url.split("/")[-1].split("?")[0]
    cache_file = _cache_path(module_name, fname)

    if _is_cached(cache_file, integrity):
        print(f"  Cached: {cache_file}")
        return cache_file

    print(f"  Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, str(cache_file))
        print(f"  Downloaded: {cache_file}")

        if integrity:
            actual = _sha256(cache_file)
            if actual != integrity:
                print(f"  WARNING: Integrity check failed!")

        return cache_file
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def fetch_source(source_config, module_name):
    """
    Fetch data from a source config dict.

    Expected keys:
      type: "local" | "hf" | "url"
      path: (for local) filesystem path
      repo: (for hf) HuggingFace repo ID
      file: (for hf) specific file within repo
      split: (for hf) dataset split
      url: (for url) download URL
      filename: (optional) override cached filename
      integrity: (optional) SHA-256 hash for verification
    """
    src_type = source_config.get("type", "local")
    integrity = source_config.get("integrity")
    filename = source_config.get("filename")

    print(f"Fetching [{src_type}] for {module_name}:")

    if src_type == "local":
        return fetch_local(
            source_config["path"], module_name, filename, integrity
        )
    elif src_type in ("hf", "huggingface"):
        return fetch_hf(
            source_config["repo"], module_name,
            file=source_config.get("file"),
            split=source_config.get("split"),
            integrity=integrity,
        )
    elif src_type == "url":
        return fetch_url(
            source_config["url"], module_name, filename, integrity
        )
    else:
        print(f"ERROR: Unknown source type '{src_type}'")
        return None


def main():
    parser = argparse.ArgumentParser(description="Teapot data fetcher")
    parser.add_argument("--type", choices=["local", "hf", "url"], required=True)
    parser.add_argument("--path", help="Local path")
    parser.add_argument("--repo", help="HuggingFace repo ID")
    parser.add_argument("--file", help="Specific file in HF repo")
    parser.add_argument("--split", help="Dataset split")
    parser.add_argument("--url", help="Download URL")
    parser.add_argument("--module", default="test", help="Module name for cache")
    parser.add_argument("--integrity", help="SHA-256 hash for verification")
    args = parser.parse_args()

    config = {"type": args.type}
    if args.path:
        config["path"] = args.path
    if args.repo:
        config["repo"] = args.repo
    if args.file:
        config["file"] = args.file
    if args.split:
        config["split"] = args.split
    if args.url:
        config["url"] = args.url
    if args.integrity:
        config["integrity"] = args.integrity

    result = fetch_source(config, args.module)
    if result:
        print(f"\nResult: {result}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
