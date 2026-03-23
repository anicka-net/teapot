"""Teapot project root discovery."""

import os
from pathlib import Path

_cached_root = None


def find_root():
    """Find the Teapot project root directory.

    Checks in order:
    1. TEAPOT_ROOT environment variable
    2. Walk up from cwd looking for modules/ + configs/
    3. Walk up from package location
    """
    global _cached_root
    if _cached_root is not None:
        return _cached_root

    # 1. Explicit env var
    env = os.environ.get("TEAPOT_ROOT")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            _cached_root = p
            return _cached_root

    # 2. Walk up from cwd
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "modules").is_dir() and (parent / "configs").is_dir():
            _cached_root = parent
            return _cached_root

    # 3. Walk up from package location
    pkg_dir = Path(__file__).resolve().parent.parent
    if (pkg_dir / "modules").is_dir() and (pkg_dir / "configs").is_dir():
        _cached_root = pkg_dir
        return _cached_root

    raise FileNotFoundError(
        "Cannot find Teapot project root (directory with modules/ and configs/). "
        "Set TEAPOT_ROOT environment variable or run from a Teapot project directory."
    )
