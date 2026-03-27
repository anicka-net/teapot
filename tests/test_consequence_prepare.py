import json
import importlib.util
from pathlib import Path

MODULE_PATH = Path("modules/safety/consequence/prepare.py")
SPEC = importlib.util.spec_from_file_location("consequence_prepare", MODULE_PATH)
consequence_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(consequence_prepare)


def test_prepare_loads_hf_export_jsonl_and_stamps_metadata(tmp_path):
    source_path = tmp_path / "secular.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "id": "ex-1",
                "category": "general",
                "conversations": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "<think>reason</think>world"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "out.jsonl"
    examples = consequence_prepare.prepare(local_path=str(source_path), output=str(output_path))

    assert len(examples) == 1
    assert examples[0]["module"] == "safety/consequence"
    assert examples[0]["license"] == "Apache-2.0"
    assert output_path.exists()


def test_prepare_exits_cleanly_on_missing_local_path(tmp_path):
    try:
        consequence_prepare.prepare(local_path=str(tmp_path / "missing.jsonl"), output=str(tmp_path / "out.jsonl"))
    except SystemExit:
        return
    raise AssertionError("Expected prepare() to exit on missing local path")
