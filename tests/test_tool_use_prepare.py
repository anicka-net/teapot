import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path("modules/capability/tool-use/prepare.py")
SPEC = importlib.util.spec_from_file_location("tool_use_prepare", MODULE_PATH)
tool_use_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool_use_prepare)


def test_tool_use_prepare_sets_license_per_source(tmp_path):
    source_path = tmp_path / "tool-use.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "g-1", "source": "glaive", "messages": []}),
                json.dumps({"id": "x-1", "source": "xlam", "messages": []}),
                json.dumps({"id": "u-1", "source": "unknown-source", "messages": []}),
            ]
        )
        + "\n"
    )

    output_path = tmp_path / "out.jsonl"
    tool_use_prepare.prepare(output=str(output_path), local=str(source_path))

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["license"] == "Apache-2.0"
    assert rows[1]["license"] == "CC-BY-4.0"
    assert rows[2]["license"] == "unknown"
