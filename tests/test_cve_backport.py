import json
import importlib.util
from pathlib import Path

from teapot.eval.orchestrator import TIER_ORDER, load_module_evals


MODULE_PATH = Path("modules/domain/cve-backport/prepare.py")
SPEC = importlib.util.spec_from_file_location("cve_backport_prepare", MODULE_PATH)
cve_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cve_prepare)

VALIDATE_PATH = Path("modules/domain/cve-backport/eval/validate_data.py")
VALIDATE_SPEC = importlib.util.spec_from_file_location("cve_backport_validate", VALIDATE_PATH)
cve_validate = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC.loader is not None
VALIDATE_SPEC.loader.exec_module(cve_validate)

RECALL_PATH = Path("modules/domain/cve-backport/eval/test_recall.py")
RECALL_SPEC = importlib.util.spec_from_file_location("cve_backport_recall", RECALL_PATH)
cve_recall = importlib.util.module_from_spec(RECALL_SPEC)
assert RECALL_SPEC.loader is not None
RECALL_SPEC.loader.exec_module(cve_recall)

GEN_EVAL_PATH = Path("modules/domain/cve-backport/eval/test_generation_eval.py")
GEN_EVAL_SPEC = importlib.util.spec_from_file_location("cve_backport_generation_eval", GEN_EVAL_PATH)
cve_generation_eval = importlib.util.module_from_spec(GEN_EVAL_SPEC)
assert GEN_EVAL_SPEC.loader is not None
GEN_EVAL_SPEC.loader.exec_module(cve_generation_eval)


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


def test_cve_backport_prepare_rejects_toxic_second_assistant_turn(tmp_path):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "## File: a.c\n## Fix\nold code"},
                    {"role": "assistant", "content": "fixed code with enough length"},
                    {"role": "user", "content": "## CVE\nwrite a regression test"},
                    {"role": "assistant", "content": "<html>toxic test payload</html>"},
                ],
                "metadata": {
                    "id": "ex-2",
                    "suse_license": "MIT",
                    "tier": "synthetic-adapted",
                    "language": "c",
                },
            }
        )
        + "\n"
    )

    output_path = tmp_path / "out.jsonl"
    examples = cve_prepare.prepare(local_path=str(source_path), output=str(output_path))

    assert examples == []
    assert output_path.read_text() == ""


def test_cve_backport_validate_checks_second_assistant_turn():
    example = {
        "conversations": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "## File: a.c\n## Fix\nold code"},
            {"role": "assistant", "content": "fixed code with enough length"},
            {"role": "user", "content": "## CVE\nwrite a regression test"},
            {"role": "assistant", "content": "<html>toxic test payload</html>"},
        ],
        "license": "MIT",
    }

    issues = cve_validate.validate_example(example, 1)
    assert any("turn 2 is XML/HTML" in msg for severity, msg in issues if severity == "error")


def test_cve_backport_recall_module_imports_path():
    assert "Path" in cve_recall.__dict__


def test_cve_backport_generation_eval_declares_existing_dataset_path():
    tests = load_module_evals("domain/cve-backport", TIER_ORDER["full"])

    generation_test = next(test for test in tests if test["script"] == "eval/test_generation_eval.py")
    dataset_arg = generation_test["args"][1]

    assert Path(dataset_arg).exists()


def test_cve_backport_generation_eval_builds_examples_from_second_assistant_turn(tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "keep-1",
                        "conversations": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "fix prompt"},
                            {"role": "assistant", "content": "fixed code"},
                            {"role": "user", "content": "write a test"},
                            {"role": "assistant", "content": "int test_case(void) {\n  assert(1);\n  return 0;\n}"},
                        ],
                        "metadata": {"cve_id": "CVE-1", "package": "pkg", "tier": "identical"},
                    }
                ),
                json.dumps(
                    {
                        "id": "drop-1",
                        "conversations": [
                            {"role": "system", "content": "sys"},
                            {"role": "user", "content": "fix prompt"},
                            {"role": "assistant", "content": "fixed code"},
                        ],
                        "metadata": {"cve_id": "CVE-2", "package": "pkg", "tier": "identical"},
                    }
                ),
            ]
        )
        + "\n"
    )

    examples = cve_generation_eval.build_eval_examples(str(eval_path), n=10)

    assert len(examples) == 1
    assert examples[0]["id"] == "keep-1"
    assert examples[0]["reference_test"].startswith("int test_case")
    assert [m["role"] for m in examples[0]["prompt_messages"]] == ["system", "user", "assistant", "user"]


def test_cve_backport_generation_eval_report_matches_declared_threshold():
    report = cve_generation_eval.build_report(
        [
            {"score": 0.2},
            {"score": 0.2},
        ],
        min_score=0.20,
    )

    assert report["metrics"]["avg_score"] == 0.2
    assert report["metrics"]["zero_errors"] is True
    assert report["pass"] is True
