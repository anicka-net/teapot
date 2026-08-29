import json

from teapot import lockfile
from teapot.provenance import collect_build_identity, identity_hash


def _manifest(tmp_path, config, output):
    manifest = tmp_path / "train.manifest.json"
    manifest.write_text(json.dumps({
        "config": str(config),
        "seed": 42,
        "modules": {},
        "output_file": str(output),
        "output_hash": lockfile.hash_file(output),
    }))
    return manifest


def test_build_identity_is_stable_for_same_inputs(tmp_path, monkeypatch):
    from teapot import provenance

    monkeypatch.setattr(
        provenance,
        "git_state",
        lambda root: {"commit": "abc123", "dirty": False},
    )
    config = tmp_path / "model.config"
    config.write_text("base:\n  model: example/model\ntraining:\n  epochs: 2\n")

    first = collect_build_identity(config, tmp_path)
    second = collect_build_identity(config, tmp_path)

    assert first == second
    assert identity_hash(first) == identity_hash(second)
    assert first["base_model"]["model"] == "example/model"
    assert first["config_hash"].startswith("sha256:")


def test_lock_records_and_verifies_build_identity(tmp_path, monkeypatch):
    from teapot import provenance

    monkeypatch.setattr(
        provenance,
        "git_state",
        lambda root: {"commit": "abc123", "dirty": False},
    )
    config = tmp_path / "model.config"
    config.write_text("base:\n  model: example/model\ntraining:\n  epochs: 2\n")
    output = tmp_path / "train.jsonl"
    output.write_text('{"id":"a"}\n')
    manifest = _manifest(tmp_path, config, output)
    lock_path = tmp_path / "teapot.lock"

    generated = lockfile.generate_lock(manifest, lock_path)

    assert generated["build_id"].startswith("sha256:")
    assert generated["build_identity"]["config_hash"].startswith("sha256:")
    assert lockfile.verify_lock(lock_path, verify_build=True) is True

    config.write_text("base:\n  model: example/model\ntraining:\n  epochs: 3\n")
    assert lockfile.verify_lock(lock_path, verify_build=True) is False


def test_old_lock_remains_valid_without_build_verification(tmp_path):
    output = tmp_path / "train.jsonl"
    output.write_text("data\n")
    lock_path = tmp_path / "teapot.lock"
    lock_path.write_text(json.dumps({
        "version": 1,
        "generated": "t",
        "config": "",
        "sources": {},
        "output_file": str(output),
        "output_hash": lockfile.hash_file(output),
    }))

    assert lockfile.verify_lock(lock_path) is True
    assert lockfile.verify_lock(lock_path, verify_build=True) is False
