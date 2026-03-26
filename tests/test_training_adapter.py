from pathlib import Path

import pytest
import yaml

from teapot import training_adapter
from teapot.train_qlora_hf import tokenize_with_label_masking


def test_generate_qlora_hf_autodetects_hardware_when_missing(tmp_path, monkeypatch):
    config_path = tmp_path / "cve.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: Qwen/Qwen2.5-Coder-32B-Instruct",
                "  method: qlora",
                "modules:",
                "  domain/cve-backport: true",
                "training:",
                "  epochs: 2",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(training_adapter, "detect_or_default_hardware", lambda *args: (24, 1))

    output_path = tmp_path / "train.sh"
    training_adapter.generate_qlora_hf(
        str(config_path),
        "train.jsonl",
        None,
        str(output_path),
    )

    script = output_path.read_text()
    assert "python3 -m teapot.train_qlora_hf" in script
    assert "--eval " not in script
    assert "--batch-size 1" in script
    assert "--grad-accum 16" in script


def test_generate_axolotl_preserves_explicit_hardware(tmp_path, monkeypatch):
    config_path = tmp_path / "train.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: meta-llama/Llama-3.1-8B-Instruct",
                "  method: qlora",
                "hardware:",
                "  gpus: 2",
                "  vram_gb: 48",
                "training:",
                "  epochs: 1",
            ]
        )
        + "\n"
    )

    called = {"count": 0}

    def unexpected_call(*args, **kwargs):
        called["count"] += 1
        return (24, 1)

    monkeypatch.setattr(training_adapter, "detect_or_default_hardware", unexpected_call)

    output_path = tmp_path / "axolotl.yaml"
    training_adapter.generate_axolotl(str(config_path), "train.jsonl", str(output_path))

    data = yaml.safe_load(output_path.read_text())
    assert called["count"] == 0
    assert data["micro_batch_size"] == 4
    assert data["gradient_accumulation_steps"] == 4


def test_generate_qlora_hf_includes_eval_when_provided(tmp_path, monkeypatch):
    config_path = tmp_path / "cve.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: Qwen/Qwen2.5-Coder-32B-Instruct",
                "  method: qlora",
                "training:",
                "  epochs: 2",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(training_adapter, "detect_or_default_hardware", lambda *args: (24, 1))

    output_path = tmp_path / "train.sh"
    training_adapter.generate_qlora_hf(
        str(config_path),
        "train.jsonl",
        "eval.jsonl",
        str(output_path),
    )

    script = output_path.read_text()
    assert "--eval eval.jsonl" in script


def test_generate_axolotl_rejects_teapot_owned_apertus_template(tmp_path):
    config_path = tmp_path / "train.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: apertus/test",
                "  method: qlora",
                "training:",
                "  chat_template: apertus-think",
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="does not support Teapot-owned template"):
        training_adapter.generate_axolotl(str(config_path), "train.jsonl", str(tmp_path / "axolotl.yaml"))


def test_tokenize_with_label_masking_prefers_composed_text():
    class FakeTokenizer:
        chat_template = None

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("should not re-template composed text")

        def __call__(self, text, truncation=True, max_length=4096, padding=False, return_tensors=None, add_special_tokens=False):
            return {"input_ids": list(range(len(text[:max_length])))}

    tok = tokenize_with_label_masking(
        {
            "text": "abc<|assistant_start|>reply<|assistant_end|>xyz",
            "assistant_spans": [[3, 41]],
        },
        FakeTokenizer(),
        4096,
    )

    assert any(label != -100 for label in tok["labels"])
