import json
import sqlite3
from pathlib import Path

import pytest

from teapot.curation import apply_curation, import_from_db_notes, resolve_curation_path


def build_curation_db(db_path: Path, with_module_column: bool = True):
    conn = sqlite3.connect(db_path)
    if with_module_column:
        conn.execute(
            """
            CREATE TABLE examples (
                id TEXT PRIMARY KEY,
                module TEXT,
                category TEXT,
                tier TEXT,
                notes TEXT,
                status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO examples VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("keep-1", "safety/consequence", "general", "secular", "KEEP_SECULAR: good", "accepted"),
                ("drop-1", "domain/cve-backport", "codegen", "synthetic", "DELETE: bad", "accepted"),
            ],
        )
    else:
        conn.execute(
            """
            CREATE TABLE examples (
                id TEXT PRIMARY KEY,
                category TEXT,
                tier TEXT,
                notes TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO examples VALUES (?, ?, ?, ?, ?)",
            ("keep-1", "general", "secular", "KEEP_SECULAR: good", "accepted"),
        )
    conn.commit()
    conn.close()


def test_import_from_db_notes_filters_by_module(tmp_path):
    db_path = tmp_path / "training.db"
    build_curation_db(db_path)

    decisions = import_from_db_notes(db_path, "safety/consequence")

    assert [decision["id"] for decision in decisions] == ["keep-1"]
    assert decisions[0]["verdict"] == "KEEP_SECULAR"


def test_import_from_db_notes_requires_module_column(tmp_path):
    db_path = tmp_path / "training.db"
    build_curation_db(db_path, with_module_column=False)

    with pytest.raises(ValueError, match="no 'module' column"):
        import_from_db_notes(db_path, "safety/consequence")


def test_resolve_published_curation_requires_module(tmp_path, monkeypatch):
    monkeypatch.setattr("teapot.curation.CURATIONS_DIR", tmp_path / ".curations")
    with pytest.raises(FileNotFoundError, match="require --module"):
        resolve_curation_path("published:v1")


def test_apply_curation_reads_published_manifest(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr("teapot.curation.find_root", lambda: root)
    monkeypatch.setattr("teapot.curation.CURATIONS_DIR", root / ".curations")

    curation_dir = root / "modules" / "safety" / "consequence" / "curations"
    curation_dir.mkdir(parents=True)
    (curation_dir / "v1.json").write_text(
        json.dumps(
            {
                "module": "safety/consequence",
                "version": "v1",
                "decisions": [{"id": "drop-1", "verdict": "DELETE"}],
            }
        )
    )

    data_path = root / "data.jsonl"
    data_path.write_text(
        json.dumps({"id": "keep-1"}) + "\n" +
        json.dumps({"id": "drop-1"}) + "\n"
    )
    output_path = root / "filtered.jsonl"

    apply_curation("published:v1", data_path, output_path, module="safety/consequence")

    kept_ids = [json.loads(line)["id"] for line in output_path.read_text().splitlines()]
    assert kept_ids == ["keep-1"]
