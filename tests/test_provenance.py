"""Regression tests for the provenance-enforcement fixes (2026-07-03).

The whole value of teapot is that the hashes/licenses it pins actually get
verified. These lock in that check_manifest re-hashes the output, verify_lock
fails closed, and the SBOM reads the license pinned in the manifest.
No torch/network needed.
"""
import hashlib
import json
from pathlib import Path

from teapot.validate_compose import check_manifest
from teapot import lockfile
from teapot.sbom import generate_sbom


def _sha256_prefixed(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return "sha256:" + h.hexdigest()


# --- check_manifest now re-hashes the output (was count-only) ---
def test_check_manifest_passes_untampered(tmp_path):
    out = tmp_path / "train.jsonl"
    out.write_text('{"id":"a"}\n{"id":"b"}\n')
    examples = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    man = tmp_path / "train.manifest.json"
    man.write_text(json.dumps(
        {"total_examples": 2, "output_hash": _sha256_prefixed(out), "modules": {}}))
    assert check_manifest(examples, str(man), str(out)) == 0


def test_check_manifest_detects_tampered_output(tmp_path):
    out = tmp_path / "train.jsonl"
    out.write_text('{"id":"a"}\n{"id":"b"}\n')
    examples = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    man = tmp_path / "train.manifest.json"
    man.write_text(json.dumps(
        {"total_examples": 2, "output_hash": _sha256_prefixed(out), "modules": {}}))
    # same count, different bytes -> hash mismatch must be caught
    out.write_text('{"id":"a"}\n{"id":"HACKED"}\n')
    assert check_manifest(examples, str(man), str(out)) >= 1


# --- verify_lock fails closed (was `except: pass` / silent skip) ---
def test_verify_lock_ok_and_detects_change(tmp_path):
    out = tmp_path / "train.jsonl"
    out.write_text("data\n")
    lock = tmp_path / "teapot.lock"
    lock.write_text(json.dumps({
        "version": 1, "generated": "t", "config": "", "sources": {},
        "output_file": str(out), "output_hash": lockfile.hash_file(str(out))}))
    assert lockfile.verify_lock(str(lock)) is True
    out.write_text("TAMPERED\n")
    assert lockfile.verify_lock(str(lock)) is False


def test_verify_lock_fails_when_output_unresolvable(tmp_path):
    # output_hash present but no output_file and no config: must FAIL, not
    # silently report "all match" (the old bug).
    lock = tmp_path / "teapot.lock"
    lock.write_text(json.dumps({
        "version": 1, "generated": "t", "config": "", "sources": {},
        "output_hash": "sha256:deadbeef"}))
    assert lockfile.verify_lock(str(lock)) is False


# --- SBOM reads the license PINNED in the manifest, not the live module.yaml ---
def test_sbom_uses_pinned_manifest_license(tmp_path):
    man = tmp_path / "train.manifest.json"
    man.write_text(json.dumps({
        "timestamp": "t", "config": "c", "base_model": "m", "chat_template": "x",
        "seed": 42, "total_examples": 1, "output_hash": "sha256:abc",
        "modules": {
            "nonexistent/module": {  # no module.yaml on disk -> old code = "unknown"
                "source": "s", "weight": 1.0, "license": "MIT-PINNED",
                "examples_raw": 1, "examples_weighted": 1, "integrity": "sha256:x",
            }
        },
    }))
    sbom_path = tmp_path / "out.sbom.json"
    generate_sbom(str(man), output=str(sbom_path))
    sbom = json.loads(sbom_path.read_text())
    licenses = [e.get("declaredLicense") for e in sbom["elements"] if "declaredLicense" in e]
    assert "MIT-PINNED" in licenses
    assert "unknown" not in licenses
