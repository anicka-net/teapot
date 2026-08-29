"""Reproducibility identity for Teapot builds.

The identity records behavior-changing inputs that are outside the composed
training-data hash: Teapot source revision, the exact config, declared base
model, Python runtime, and installed training-stack package versions.

It is an identity, not a promise of bit-identical GPU training.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml


PACKAGE_NAMES = (
    "teapot-ai",
    "PyYAML",
    "jsonschema",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "unsloth",
    "axolotl",
    "deepspeed",
)


def hash_file(path):
    """Return a Teapot-style sha256:<hex> digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def git_state(root):
    """Return the current source commit and whether the worktree is dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root), capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def package_versions(names=PACKAGE_NAMES):
    """Return versions only for packages installed in this environment."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_build_identity(config_path, root):
    """Collect the reproducibility inputs available before training starts."""
    config_path = Path(config_path)
    config_hash = hash_file(config_path) if config_path.is_file() else None

    raw = {}
    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text()) or {}
    base = raw.get("base", {}) if isinstance(raw, dict) else {}

    return {
        "version": 1,
        "config_hash": config_hash,
        "teapot_source": git_state(root),
        "base_model": {
            "model": base.get("model"),
            "revision": base.get("revision"),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
    }


def identity_hash(identity):
    """Return a stable hash of an identity object."""
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
