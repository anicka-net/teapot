import importlib.util
import json
from pathlib import Path
import subprocess


def _load_module(path_str, module_name):
    path = Path(path_str)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool_use_eval = _load_module(
    "modules/capability/tool-use/eval/test_tool_use.py",
    "tool_use_eval",
)
consequence_format_eval = _load_module(
    "modules/safety/consequence/eval/test_format.py",
    "consequence_format_eval",
)


def test_tool_use_full_eval_counts_safety_prompts_in_pass_fail(monkeypatch):
    responses = iter(
        [
            "let me search",
            "let me search",
            "let me search",
            "let me search",
            "no tool needed",
            "no tool needed",
            "no tool needed",
            "no tool needed",
            "i cannot help with that",
            "i cannot help with that",
            "i cannot help with that",
        ]
    )

    monkeypatch.setattr(tool_use_eval, "query", lambda url, prompt: next(responses))

    # Replicate main() logic enough to assert the contract.
    tool_correct = 0
    no_tool_correct = 0
    safety_correct = 0

    for prompt in tool_use_eval.TOOL_PROMPTS:
        resp = tool_use_eval.query("", prompt).lower()
        if any(ind in resp for ind in tool_use_eval.TOOL_INDICATORS):
            tool_correct += 1

    for prompt in tool_use_eval.NO_TOOL_PROMPTS:
        resp = tool_use_eval.query("", prompt).lower()
        if not any(ind in resp for ind in tool_use_eval.TOOL_INDICATORS):
            no_tool_correct += 1

    for prompt in tool_use_eval.SAFETY_PROMPTS:
        resp = tool_use_eval.query("", prompt).lower()
        if any(sig in resp for sig in tool_use_eval.REFUSAL_SIGNALS):
            safety_correct += 1

    passed = tool_correct + no_tool_correct + safety_correct
    total = len(tool_use_eval.TOOL_PROMPTS) + len(tool_use_eval.NO_TOOL_PROMPTS) + len(tool_use_eval.SAFETY_PROMPTS)

    assert passed == total


def test_consequence_format_eval_requires_system_and_assistant(tmp_path):
    data_path = tmp_path / "consequence.jsonl"
    data_path.write_text(
        '{"id":"ex-1","conversations":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}\n',
        encoding="utf-8",
    )

    # Execute the eval script via its main entry contract.
    result = subprocess.run(
        ["python3", "modules/safety/consequence/eval/test_format.py", str(data_path)],
        cwd="/home/anicka/playground/teapot",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert '"missing_system": 1' in result.stdout


def test_format_eval_scripts_accept_orchestrator_style_url_flag(tmp_path):
    cases = [
        (
            "modules/safety/consequence/eval/test_format.py",
            {
                "id": "sc-1",
                "conversations": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            },
        ),
        (
            "modules/safety/consequence-aegis/eval/test_format.py",
            {
                "id": "sca-1",
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            },
        ),
        (
            "modules/safety/ke-thinking/eval/test_format.py",
            {
                "id": "kt-1",
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "<think>r</think>a"},
                ],
            },
        ),
        (
            "modules/domain/upstream-thinking/eval/test_format.py",
            {
                "id": "ut-1",
                "conversations": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "<think>r</think>a"},
                ],
            },
        ),
        (
            "modules/capability/reward-evaluator/eval/test_format.py",
            {
                "id": "re-1",
                "task": "reward-evaluation",
                "category": "reward-evaluation",
                "conversations": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ],
            },
        ),
    ]

    for idx, (script, example) in enumerate(cases, start=1):
        data_path = tmp_path / f"eval-{idx}.jsonl"
        data_path.write_text(json.dumps(example) + "\n", encoding="utf-8")
        result = subprocess.run(
            ["python3", script, str(data_path), "--url", "http://localhost:9"],
            cwd="/home/anicka/playground/teapot",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (script, result.stdout, result.stderr)
