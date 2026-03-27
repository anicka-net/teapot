import importlib.util
import json
from pathlib import Path


def _load_prepare(path_str, module_name):
    path = Path(path_str)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


kagyu_prepare = _load_prepare("modules/safety/kagyu/prepare.py", "kagyu_prepare")
ke_thinking_prepare = _load_prepare("modules/safety/ke-thinking/prepare.py", "ke_thinking_prepare")


def test_kagyu_prepare_loads_local_hf_export_and_stamps_metadata(tmp_path):
    source_path = tmp_path / "buddhist.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "id": "kg-1",
                "category": "meditation",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "<think>reason</think>answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = kagyu_prepare.prepare(output=str(tmp_path / "out.jsonl"), local_path=str(source_path))
    assert len(examples) == 1
    assert examples[0]["module"] == "safety/kagyu"
    assert examples[0]["license"] == "Apache-2.0"


def test_ke_thinking_prepare_loads_local_hf_export_and_stamps_metadata(tmp_path):
    source_path = tmp_path / "thinking.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "id": "kt-1",
                "category": "positive-engagement",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "<think>reason</think>answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = ke_thinking_prepare.prepare(output=str(tmp_path / "out.jsonl"), local_path=str(source_path))
    assert len(examples) == 1
    assert examples[0]["module"] == "safety/ke-thinking"
    assert examples[0]["license"] == "Apache-2.0"


def test_ke_thinking_eval_script_exists():
    assert Path("modules/safety/ke-thinking/eval/test_format.py").exists()
