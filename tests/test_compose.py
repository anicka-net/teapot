from pathlib import Path

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
