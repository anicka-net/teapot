import json

from teapot import training_adapter
from teapot.train_full_hf import FormattedDataset, verify_special_tokens


class FakeTokenizer:
    def __init__(self, encodings, special_tokens=None):
        self.encodings = encodings
        self.all_special_tokens = list(special_tokens or [])

    def encode(self, token, add_special_tokens=False):
        return self.encodings[token]

    def __call__(
        self,
        text,
        truncation=True,
        max_length=4096,
        padding=False,
        return_offsets_mapping=False,
    ):
        text = text[:max_length]
        offsets = [[idx, idx + 1] for idx in range(len(text))]
        result = {
            "input_ids": list(range(len(text))),
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_verify_special_tokens_accepts_registered_single_token_controls():
    encodings = {
        token: [idx]
        for idx, token in enumerate(
            [
                "<|system_start|>",
                "<|system_end|>",
                "<|developer_start|>",
                "<|developer_end|>",
                "<|user_start|>",
                "<|user_end|>",
                "<|assistant_start|>",
                "<|assistant_end|>",
                "<|inner_prefix|>",
                "<|inner_suffix|>",
            ],
            start=1,
        )
    }
    tokenizer = FakeTokenizer(encodings, special_tokens=encodings.keys())

    assert verify_special_tokens(tokenizer, "apertus-think") is True


def test_verify_special_tokens_rejects_split_or_non_special_controls():
    tokenizer = FakeTokenizer(
        {
            "<|system_start|>": [1],
            "<|system_end|>": [2],
            "<|developer_start|>": [3],
            "<|developer_end|>": [4],
            "<|user_start|>": [5],
            "<|user_end|>": [6],
            "<|assistant_start|>": [7],
            "<|assistant_end|>": [8, 9],
            "<|inner_prefix|>": [10],
            "<|inner_suffix|>": [11],
        },
        special_tokens={
            "<|system_start|>",
            "<|system_end|>",
            "<|developer_start|>",
            "<|developer_end|>",
            "<|user_start|>",
            "<|user_end|>",
            "<|assistant_start|>",
            "<|inner_prefix|>",
            "<|inner_suffix|>",
        },
    )

    assert verify_special_tokens(tokenizer, "apertus-think") is False


def test_formatted_dataset_prefers_composed_text(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "text": "abcXYZ",
                "assistant_spans": [[3, 6]],
                "conversations": [
                    {"role": "system", "content": "ignored"},
                    {"role": "assistant", "content": "ignored"},
                ],
            }
        )
        + "\n"
    )

    dataset = FormattedDataset(str(data_path), FakeTokenizer({}), template="apertus-think")
    row = dataset[0]
    assert row["input_ids"].tolist() == [0, 1, 2, 3, 4, 5]
    assert row["labels"].tolist() == [-100, -100, -100, 3, 4, 5]


def test_generate_full_hf_uses_stable_output_dir(tmp_path, monkeypatch):
    config_path = tmp_path / "apertus-70b-full.config"
    config_path.write_text(
        "\n".join(
            [
                "base:",
                "  model: swiss-ai/Apertus-70B-Instruct-2509",
                "  method: full",
                "training:",
                "  chat_template: apertus-think",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(training_adapter, "detect_or_default_hardware", lambda *args: (94, 2))

    output_path = tmp_path / "train.sh"
    training_adapter.generate_full_hf(str(config_path), "train.jsonl", str(output_path))

    script = output_path.read_text()
    assert "python3" not in script
    assert "--output output-teapot-apertus-70b-full" in script
    assert "$(date" not in script
