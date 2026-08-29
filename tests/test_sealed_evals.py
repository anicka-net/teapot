import hashlib
import json
from pathlib import Path

import yaml

from teapot.eval.orchestrator import load_sealed_evals
from teapot.eval.sealed import run_sealed_suite


def _digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runner(tmp_path, body):
    path = tmp_path / "private_suite.py"
    path.write_text(body)
    return path


def test_sealed_suite_runs_integrity_pinned_external_runner(tmp_path, monkeypatch):
    runner = _runner(
        tmp_path,
        "import argparse, json\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--url')\n"
        "p.add_argument('--model-name')\n"
        "a = p.parse_args()\n"
        "print(json.dumps({'pass': bool(a.url and a.model_name), 'passed': 7, 'total': 7}))\n",
    )
    monkeypatch.setenv("TEAPOT_PRIVATE_EVAL", str(runner))

    result = run_sealed_suite(
        {
            "name": "philosophy-heldout-v1",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": _digest(runner),
        },
        "http://model/v1/chat/completions",
        model_name="model-under-test",
    )

    assert result.status == "pass"
    assert result.passed == 7
    assert result.total == 7
    assert result.details["sealed"] is True
    assert result.details["integrity"] == _digest(runner)


def test_sealed_suite_fails_closed_on_hash_mismatch(tmp_path, monkeypatch):
    runner = _runner(tmp_path, "print('{}')\n")
    monkeypatch.setenv("TEAPOT_PRIVATE_EVAL", str(runner))

    result = run_sealed_suite(
        {
            "name": "heldout",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": "sha256:" + "0" * 64,
        },
        "http://model",
    )

    assert result.status == "error"
    assert "integrity" in result.error.lower()


def test_missing_required_sealed_suite_is_error(monkeypatch):
    monkeypatch.delenv("TEAPOT_PRIVATE_EVAL", raising=False)
    result = run_sealed_suite(
        {
            "name": "heldout",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": "sha256:" + "0" * 64,
        },
        "http://model",
    )
    assert result.status == "error"


def test_missing_optional_sealed_suite_is_skip(monkeypatch):
    monkeypatch.delenv("TEAPOT_PRIVATE_EVAL", raising=False)
    result = run_sealed_suite(
        {
            "name": "heldout",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": "sha256:" + "0" * 64,
            "required": False,
        },
        "http://model",
    )
    assert result.status == "skip"


def test_malformed_stdout_is_not_copied_into_report(tmp_path, monkeypatch):
    runner = _runner(tmp_path, "print('PRIVATE RAW OUTPUT MUST NOT ESCAPE')\n")
    monkeypatch.setenv("TEAPOT_PRIVATE_EVAL", str(runner))

    result = run_sealed_suite(
        {
            "name": "heldout",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": _digest(runner),
        },
        "http://model",
    )

    assert result.status == "error"
    assert "PRIVATE RAW OUTPUT" not in result.error
    assert "PRIVATE RAW OUTPUT" not in json.dumps(result.details)


def test_structured_but_invalid_aggregate_fails_closed(tmp_path, monkeypatch):
    runner = _runner(
        tmp_path,
        "import json\nprint(json.dumps({'pass': 'false', 'passed': '7', 'total': 7}))\n",
    )
    monkeypatch.setenv("TEAPOT_PRIVATE_EVAL", str(runner))

    result = run_sealed_suite(
        {
            "name": "heldout",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": _digest(runner),
        },
        "http://model",
    )

    assert result.status == "error"
    assert result.passed == 0
    assert result.total == 0


def test_json_array_is_not_an_aggregate(tmp_path, monkeypatch):
    runner = _runner(tmp_path, "print('[1, 1]')\n")
    monkeypatch.setenv("TEAPOT_PRIVATE_EVAL", str(runner))

    result = run_sealed_suite(
        {
            "name": "heldout",
            "path_env": "TEAPOT_PRIVATE_EVAL",
            "integrity": _digest(runner),
        },
        "http://model",
    )

    assert result.status == "error"


def test_sealed_suite_tier_selection(tmp_path):
    config = tmp_path / "model.config"
    config.write_text(yaml.safe_dump({
        "modules": {},
        "eval": {
            "sealed_suites": [
                {
                    "name": "fast-private",
                    "tier": "fast",
                    "path_env": "FAST_PRIVATE",
                    "integrity": "sha256:" + "1" * 64,
                },
                {
                    "name": "full-private",
                    "tier": "full",
                    "path_env": "FULL_PRIVATE",
                    "integrity": "sha256:" + "2" * 64,
                },
            ]
        },
    }))

    fast = load_sealed_evals(config, 0)
    full = load_sealed_evals(config, 2)

    assert [s["name"] for s in fast] == ["fast-private"]
    assert [s["name"] for s in full] == ["fast-private", "full-private"]
