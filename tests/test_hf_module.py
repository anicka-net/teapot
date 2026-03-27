import importlib.util
import json
from pathlib import Path

import yaml

from teapot import sources
from teapot.hf_module import stable_example_id


MODULE_PATH = Path("modules/domain/upstream-thinking/prepare.py")
SPEC = importlib.util.spec_from_file_location("upstream_thinking_prepare", MODULE_PATH)
upstream_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(upstream_prepare)


def test_stable_example_id_is_deterministic():
    payload = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]
    assert stable_example_id("x", payload) == stable_example_id("x", payload)


def test_resolve_source_prefers_default_path_before_hf_fetch(tmp_path, monkeypatch):
    local_file = tmp_path / "prepared.jsonl"
    local_file.write_text("{}\n", encoding="utf-8")
    module_yaml = {
        "data": {
            "sources": [
                {
                    "id": "demo",
                    "type": "huggingface",
                    "repo": "demo/repo",
                    "split": "train",
                    "default_path": str(local_file),
                }
            ]
        }
    }

    def fail_fetch(*args, **kwargs):
        raise AssertionError("HF fetch should not run when default_path exists")

    monkeypatch.setattr(sources, "_fetch_from_hf", fail_fetch)
    assert sources.resolve_source("demo", module_yaml) == str(local_file)


def test_resolve_source_uses_declared_repo_split_when_cache_missing(monkeypatch):
    module_yaml = {
        "data": {
            "sources": [
                {
                    "id": "demo",
                    "type": "huggingface",
                    "repo": "demo/repo",
                    "split": "train",
                    "revision": "abc123",
                }
            ]
        }
    }

    def fake_fetch(repo, source_id, filename=None, split=None, revision=None):
        assert repo == "demo/repo"
        assert source_id == "demo"
        assert filename is None
        assert split == "train"
        assert revision == "abc123"
        return "/tmp/demo.jsonl"

    monkeypatch.setattr(sources, "_fetch_from_hf", fake_fetch)
    assert sources.resolve_source("demo", module_yaml) == "/tmp/demo.jsonl"


def test_upstream_thinking_prepare_dedupes_and_uses_stable_ids(tmp_path, monkeypatch):
    source_file = tmp_path / "source.jsonl"
    duplicate = {
        "conversations": [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<think>reasoning words words words words words words words words words words "
                "words words words words words words words words words words words words</think> "
                "final answer words words words words words words words words words words words words "
                "words words words words words words words words",
            },
        ],
        "category": "general-reasoning",
    }
    source_file.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )

    module_yaml = yaml.safe_load(Path("modules/domain/upstream-thinking/module.yaml").read_text())
    monkeypatch.setattr(
        upstream_prepare,
        "MODULE_YAML",
        {
            **module_yaml,
            "data": {
                **module_yaml["data"],
                "sources": [
                    {**src, "default_path": str(source_file)}
                    for src in module_yaml["data"]["sources"]
                ],
            },
        },
    )
    monkeypatch.setattr(
        upstream_prepare,
        "SOURCES",
        [{"id": "reddit-ethics", "cap": None, "min_think": 20, "license": "Apache-2.0"}],
    )

    output_path = tmp_path / "out.jsonl"
    examples = upstream_prepare.prepare(output=str(output_path), cap_override=1, seed=42)

    ids = [ex["id"] for ex in examples]
    assert len(ids) == len(set(ids))
    assert len(examples) == 1
    assert ids[0] == stable_example_id("reddit-ethics", examples[0]["conversations"])


def test_upstream_thinking_eval_script_exists():
    assert Path("modules/domain/upstream-thinking/eval/test_format.py").exists()
