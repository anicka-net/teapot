import pytest

from teapot import training_adapter


def test_generate_unsloth_rejects_insufficient_vram(tmp_path, monkeypatch):
    config_path = tmp_path / "train.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: swiss-ai/Apertus-70B-Instruct-2509",
                "  method: qlora",
                "training:",
                "  chat_template: apertus-think",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(training_adapter, "detect_or_default_hardware", lambda *args: (24, 1))

    with pytest.raises(ValueError, match="requires at least"):
        training_adapter.generate_unsloth(str(config_path), "train.jsonl", str(tmp_path / "train.sh"))


def test_generate_unsloth_includes_method_and_template(tmp_path, monkeypatch):
    config_path = tmp_path / "train.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: meta-llama/Llama-3.1-8B-Instruct",
                "  method: full",
                "hardware:",
                "  gpus: 1",
                "  vram_gb: 46",
                "training:",
                "  chat_template: apertus-think",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(training_adapter, "detect_or_default_hardware", lambda *args: (46, 1))

    output_path = tmp_path / "train.sh"
    training_adapter.generate_unsloth(str(config_path), "train.jsonl", str(output_path))

    script = output_path.read_text()
    assert "--method full" in script
    assert "--template apertus-think" in script
    assert "--output output-teapot-train" in script
