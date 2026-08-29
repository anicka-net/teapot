"""Execution support for private, integrity-pinned evaluation suites.

A sealed suite is a single external executable (or Python script) whose path
is supplied only through an environment variable. Teapot hashes the file
before execution and consumes only aggregate JSON from stdout. The suite
payload therefore does not need to live in the Teapot repository or config.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from teapot.eval.schema import SuiteResult


def sha256_file(path):
    """Return a Teapot-style sha256:<hex> digest for *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _flatten_args(args):
    if isinstance(args, dict):
        flat = []
        for key, value in args.items():
            flat.extend([str(key), str(value)])
        return flat
    return [str(value) for value in (args or [])]


def run_sealed_suite(spec, url, model_name="", timeout=600):
    """Run one private suite without opening its prompt/output artifacts.

    The configured environment variable must resolve to a single file. Its
    SHA-256 must match the config before execution. The subprocess must emit a
    sanitized aggregate JSON object on stdout with either an explicit ``pass``
    flag or ``passed``/``total`` counts. Raw generations belong inside the
    sealed runner and must never be written to stdout.
    """
    name = spec.get("name", "sealed")
    env_name = spec.get("path_env", "")
    required = spec.get("required", True)
    expected = spec.get("integrity", "")

    path_value = os.environ.get(env_name, "") if env_name else ""
    if not path_value:
        return SuiteResult(
            name=f"sealed:{name}",
            status="error" if required else "skip",
            passed=0,
            total=0,
            details={"sealed": True, "path_env": env_name},
            error=f"Sealed suite path environment variable is not set: {env_name}",
        )

    path = Path(path_value)
    if not path.is_file():
        return SuiteResult(
            name=f"sealed:{name}",
            status="error" if required else "skip",
            passed=0,
            total=0,
            details={"sealed": True, "path_env": env_name},
            error=f"Sealed suite path is not a file (from {env_name})",
        )

    actual = sha256_file(path)
    if not expected or actual != expected:
        return SuiteResult(
            name=f"sealed:{name}",
            status="error",
            passed=0,
            total=0,
            details={
                "sealed": True,
                "path_env": env_name,
                "integrity": actual,
            },
            error="Sealed suite integrity mismatch or missing integrity pin",
        )

    cmd = [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]
    cmd.extend(_flatten_args(spec.get("args", [])))
    if url:
        cmd.extend(["--url", url])
    if model_name:
        cmd.extend(["--model-name", model_name])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 1)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return SuiteResult(
            name=f"sealed:{name}",
            status="error",
            passed=0,
            total=0,
            duration_seconds=round(time.time() - t0, 1),
            details={"sealed": True, "path_env": env_name, "integrity": actual},
            error=f"Sealed suite execution failed: {type(exc).__name__}",
        )

    # stdout is deliberately parsed but never copied into the report or an
    # error message. A sealed runner must expose aggregates only.
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return SuiteResult(
            name=f"sealed:{name}",
            status="error",
            passed=0,
            total=0,
            duration_seconds=elapsed,
            details={"sealed": True, "path_env": env_name, "integrity": actual},
            error="Sealed suite did not return aggregate JSON",
        )

    if proc.returncode != 0:
        return SuiteResult(
            name=f"sealed:{name}",
            status="error",
            passed=int(data.get("passed", 0) or 0),
            total=int(data.get("total", 0) or 0),
            duration_seconds=elapsed,
            details={"sealed": True, "path_env": env_name, "integrity": actual},
            error=f"Sealed suite exited with status {proc.returncode}",
        )

    passed = int(data.get("passed", 0) or 0)
    total = int(data.get("total", 0) or 0)
    if "pass" in data:
        status = "pass" if bool(data["pass"]) else "fail"
    else:
        status = "pass" if total > 0 and passed == total else "fail"

    return SuiteResult(
        name=f"sealed:{name}",
        status=status,
        passed=passed,
        total=total,
        threshold=str(spec.get("pass_criteria", "")),
        duration_seconds=elapsed,
        details={
            "sealed": True,
            "path_env": env_name,
            "integrity": actual,
        },
    )
