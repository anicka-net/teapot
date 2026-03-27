import json
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path("modules/safety/consequence-aegis/prepare.py")
SPEC = importlib.util.spec_from_file_location("consequence_aegis_prepare", MODULE_PATH)
aegis_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aegis_prepare)


def test_prepare_normalizes_conversations_and_prompt_response(tmp_path):
    source_path = tmp_path / "aegis.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "conv-1",
                        "conversations": [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "prompt": "question",
                        "response": "answer",
                        "category": "sexual-safety",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "out.jsonl"
    examples = aegis_prepare.prepare(output=str(output_path), local_path=str(source_path))

    assert len(examples) == 2
    assert examples[0]["module"] == "safety/consequence-aegis"
    assert examples[1]["conversations"][0]["role"] == "user"
    assert examples[1]["conversations"][1]["role"] == "assistant"
    assert output_path.exists()


def test_prepare_uses_stable_id_when_missing(tmp_path):
    source_path = tmp_path / "aegis.jsonl"
    source_path.write_text(
        json.dumps({"prompt": "q", "response": "r"}) + "\n",
        encoding="utf-8",
    )

    examples = aegis_prepare.prepare(output=str(tmp_path / "out.jsonl"), local_path=str(source_path))

    assert examples[0]["id"].startswith("consequence-aegis-")


def test_eval_script_exists():
    assert Path("modules/safety/consequence-aegis/eval/test_format.py").exists()


def test_prepare_exits_cleanly_on_missing_local_path(tmp_path):
    with pytest.raises(SystemExit):
        aegis_prepare.prepare(output=str(tmp_path / "out.jsonl"), local_path=str(tmp_path / "missing.jsonl"))
