import sqlite3
from pathlib import Path

import pytest

from teapot.curation import import_from_db_notes


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
