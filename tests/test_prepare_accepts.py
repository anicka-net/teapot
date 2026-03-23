from pathlib import Path

from teapot import compose as compose_mod


def test_run_prepare_passes_declared_format(tmp_path, monkeypatch):
    module_dir = tmp_path / "tool-use"
    module_dir.mkdir()

    (module_dir / "prepare.py").write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import argparse",
                "from pathlib import Path",
                "p = argparse.ArgumentParser()",
                "p.add_argument('--format')",
                "p.add_argument('--output')",
                "args = p.parse_args()",
                "Path('args.txt').write_text(args.format or '')",
                "Path('data').mkdir(exist_ok=True)",
                "Path(args.output).write_text('')",
                "print('ok')",
            ]
        )
        + "\n"
    )

    (module_dir / "module.yaml").write_text(
        "\n".join(
            [
                "name: capability/tool-use",
                "prepare:",
                "  script: prepare.py",
                "  args:",
                "    --output: data/tool-use.jsonl",
                "  accepts:",
                "    - --format",
            ]
        )
        + "\n"
    )

    data_path = compose_mod.run_prepare(module_dir, "capability/tool-use", chat_template="llama")

    assert data_path == module_dir / "data" / "tool-use.jsonl"
    assert (module_dir / "args.txt").read_text() == "llama"
