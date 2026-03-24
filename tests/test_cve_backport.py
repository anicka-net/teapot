import json
import importlib.util
from pathlib import Path

from teapot.eval.orchestrator import TIER_ORDER, load_module_evals


MODULE_PATH = Path("modules/domain/cve-backport/prepare.py")
SPEC = importlib.util.spec_from_file_location("cve_backport_prepare", MODULE_PATH)
cve_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cve_prepare)


def test_cve_backport_prepare_normalizes_blank_license(tmp_path):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "## File: a.c\n## Fix\nold code"},
                    {"role": "assistant", "content": "fixed code with enough length"},
                ],
                "metadata": {
                    "id": "ex-1",
                    "suse_license": "",
                    "tier": "synthetic-adapted",
                    "language": "c",
                },
            }
        )
        + "\n"
    )

    output_path = tmp_path / "out.jsonl"
    examples = cve_prepare.prepare(local_path=str(source_path), output=str(output_path))

    assert examples[0]["license"] == "unknown"
    written = json.loads(output_path.read_text().strip())
    assert written["license"] == "unknown"


def test_cve_backport_eval_declares_existing_dataset_path():
    tests = load_module_evals("domain/cve-backport", TIER_ORDER["fast"])

    validate_test = next(test for test in tests if test["script"] == "eval/validate_data.py")
    dataset_arg = validate_test["args"][0]

    assert Path(dataset_arg).exists()
