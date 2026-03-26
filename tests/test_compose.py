from pathlib import Path

import json
import pytest

from teapot import compose as compose_mod


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "test.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: test/model",
                "modules:",
                "  safety/consequence: true",
                "license:",
                "  allowed: all",
                "output:",
                "  file: out.jsonl",
            ]
        )
        + "\n"
    )
    return config_path


def test_compose_exits_when_no_examples_loaded(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    data_path = tmp_path / "prepared.jsonl"
    data_path.write_text("")

    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: safety/consequence\n")

    monkeypatch.setattr(compose_mod, "TEAPOT_ROOT", tmp_path)
    monkeypatch.setattr(
        compose_mod,
        "find_module",
        lambda module_name: (module_dir, module_dir / "module.yaml"),
    )
    monkeypatch.setattr(compose_mod, "run_prepare", lambda *args, **kwargs: data_path)

    with pytest.raises(SystemExit) as exc:
        compose_mod.compose(str(config_path), output=str(tmp_path / "out.jsonl"))

    assert exc.value.code == 1
    assert not (tmp_path / "out.jsonl").exists()
    assert not (tmp_path / "out.manifest.json").exists()


def test_compose_exits_when_requested_curation_is_missing(tmp_path, monkeypatch):
    config_path = tmp_path / "test.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: test/model",
                "modules:",
                "  safety/consequence: true",
                "license:",
                "  allowed: all",
                "curations:",
                "  safety/consequence: local:v1",
                "output:",
                "  file: out.jsonl",
            ]
        )
        + "\n"
    )

    data_path = tmp_path / "prepared.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": "ex-1",
                "license": "MIT",
                "conversations": [{"role": "user", "content": "hi"}],
            }
        )
        + "\n"
    )

    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: safety/consequence\n")

    monkeypatch.setattr(compose_mod, "TEAPOT_ROOT", tmp_path)
    monkeypatch.setattr(
        compose_mod,
        "find_module",
        lambda module_name: (module_dir, module_dir / "module.yaml"),
    )
    monkeypatch.setattr(compose_mod, "run_prepare", lambda *args, **kwargs: data_path)

    with pytest.raises(FileNotFoundError, match="Curation.*not found"):
        compose_mod.compose(str(config_path), output=str(tmp_path / "out.jsonl"))


def test_compose_rejects_ambiguous_curation_ref(tmp_path, monkeypatch):
    config_path = tmp_path / "test.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: test/model",
                "modules:",
                "  safety/consequence: true",
                "license:",
                "  allowed: all",
                "curations:",
                "  safety/consequence: v1",
                "output:",
                "  file: out.jsonl",
            ]
        )
        + "\n"
    )

    data_path = tmp_path / "prepared.jsonl"
    data_path.write_text(json.dumps({"id": "ex-1", "license": "MIT", "conversations": []}) + "\n")

    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: safety/consequence\n")

    monkeypatch.setattr(compose_mod, "TEAPOT_ROOT", tmp_path)
    monkeypatch.setattr(
        compose_mod,
        "find_module",
        lambda module_name: (module_dir, module_dir / "module.yaml"),
    )
    monkeypatch.setattr(compose_mod, "run_prepare", lambda *args, **kwargs: data_path)

    with pytest.raises(ValueError, match="Ambiguous curation reference"):
        compose_mod.compose(str(config_path), output=str(tmp_path / "out.jsonl"))


def test_compose_manifest_records_curation_details(tmp_path, monkeypatch):
    config_path = tmp_path / "test.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: test/model",
                "modules:",
                "  safety/consequence: true",
                "license:",
                "  allowed: all",
                "curations:",
                "  safety/consequence: local:v1",
                "output:",
                "  file: out.jsonl",
            ]
        )
        + "\n"
    )

    data_path = tmp_path / "prepared.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "keep-1", "license": "MIT", "conversations": []}),
                json.dumps({"id": "drop-1", "license": "MIT", "conversations": []}),
            ]
        )
        + "\n"
    )

    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: safety/consequence\n")

    curations_dir = tmp_path / ".curations"
    curations_dir.mkdir()
    curation_path = curations_dir / "safety-consequence-v1.json"
    curation_path.write_text(
        json.dumps(
            {
                "module": "safety/consequence",
                "version": "v1",
                "decisions": [
                    {"id": "drop-1", "verdict": "DELETE"},
                ],
            }
        )
    )

    monkeypatch.setattr(compose_mod, "TEAPOT_ROOT", tmp_path)
    monkeypatch.setattr(
        compose_mod,
        "find_module",
        lambda module_name: (module_dir, module_dir / "module.yaml"),
    )
    monkeypatch.setattr(compose_mod, "run_prepare", lambda *args, **kwargs: data_path)

    compose_mod.compose(str(config_path), output=str(tmp_path / "out.jsonl"))

    manifest = json.loads((tmp_path / "out.manifest.json").read_text())
    module_info = manifest["modules"]["safety/consequence"]
    assert module_info["examples_filtered_by_curation"] == 1
    assert module_info["examples_filtered_by_license"] == 0
    assert module_info["curation"]["version"] == "v1"
    assert module_info["curation"]["path"].endswith("safety-consequence-v1.json")
    assert module_info["curation"]["integrity"].startswith("sha256:")


def test_compose_applies_apertus_think_template(tmp_path, monkeypatch):
    config_path = tmp_path / "test.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: apertus/test",
                "modules:",
                "  safety/consequence: true",
                "license:",
                "  allowed: all",
                "training:",
                "  chat_template: apertus-think",
                "output:",
                "  file: out.jsonl",
            ]
        )
        + "\n"
    )

    data_path = tmp_path / "prepared.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": "ex-1",
                "license": "MIT",
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "<think>reason</think>\n\nanswer"},
                ],
            }
        )
        + "\n"
    )

    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: safety/consequence\n")

    monkeypatch.setattr(compose_mod, "TEAPOT_ROOT", tmp_path)
    monkeypatch.setattr(compose_mod, "find_module", lambda module_name: (module_dir, module_dir / "module.yaml"))
    monkeypatch.setattr(compose_mod, "run_prepare", lambda *args, **kwargs: data_path)

    compose_mod.compose(str(config_path), output=str(tmp_path / "out.jsonl"))

    row = json.loads((tmp_path / "out.jsonl").read_text().strip())
    assert row["text"].startswith("<|system_start|>sys<|system_end|><|developer_start|>")
    assert "<|inner_prefix|>reason<|inner_suffix|>answer<|assistant_end|>" in row["text"]
    assert row["assistant_spans"]


def test_compose_applies_apertus_full_tool_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "test.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: apertus/test",
                "modules:",
                "  capability/tool-use: true",
                "license:",
                "  allowed: all",
                "training:",
                "  chat_template: apertus-full",
                "output:",
                "  file: out.jsonl",
            ]
        )
        + "\n"
    )

    data_path = tmp_path / "prepared.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": "tool-1",
                "license": "Apache-2.0",
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "weather?"},
                    {"role": "assistant", "content": "<think>use tool</think>\n\ncalling"},
                    {"role": "tool", "content": "{\"temp\": 20}"},
                    {"role": "assistant", "content": "It is 20C."},
                ],
            }
        )
        + "\n"
    )

    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: capability/tool-use\n")

    monkeypatch.setattr(compose_mod, "TEAPOT_ROOT", tmp_path)
    monkeypatch.setattr(compose_mod, "find_module", lambda module_name: (module_dir, module_dir / "module.yaml"))
    monkeypatch.setattr(compose_mod, "run_prepare", lambda *args, **kwargs: data_path)

    compose_mod.compose(str(config_path), output=str(tmp_path / "out.jsonl"))

    row = json.loads((tmp_path / "out.jsonl").read_text().strip())
    assert "Tool Capabilities: enabled" in row["text"]
    assert "[Tool result]: {\"temp\": 20}" in row["text"]
