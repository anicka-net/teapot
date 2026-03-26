import json
import sqlite3
from pathlib import Path

from modules.safety.consequence import prepare as consequence_prepare


def build_test_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE system_prompts (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute(
        """
        CREATE TABLE examples (
            id TEXT PRIMARY KEY,
            category TEXT,
            source TEXT,
            conversations TEXT,
            tier TEXT,
            reasoning TEXT,
            role TEXT,
            status TEXT
        )
        """
    )
    conn.execute("INSERT INTO system_prompts VALUES (?, ?)", ("v4", "system v4"))
    conn.execute(
        "INSERT INTO system_prompts VALUES (?, ?)",
        ("reward-evaluator-v1", "reward prompt"),
    )
    conversations = json.dumps(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    )
    conn.execute(
        """
        INSERT INTO examples
        (id, category, source, conversations, tier, reasoning, role, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("ex-1", "general", "test", conversations, "secular", None, "conversational", "accepted"),
    )
    conn.execute(
        """
        INSERT INTO examples
        (id, category, source, conversations, tier, reasoning, role, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("ex-2", "corporate-vs-dharma", "test", conversations, "secular", None, "conversational", "accepted"),
    )
    conn.execute(
        """
        INSERT INTO examples
        (id, category, source, conversations, tier, reasoning, role, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("ex-3", "reward-evaluation", "test", conversations, "secular", None, "conversational", "accepted"),
    )
    conn.commit()
    conn.close()


def test_prepare_reads_sqlite_in_readonly_mode(tmp_path):
    db_path = tmp_path / "training.db"
    build_test_db(db_path)
    db_path.chmod(0o444)

    output_path = tmp_path / "out.jsonl"
    examples = consequence_prepare.prepare(local=str(db_path), output=str(output_path))

    assert len(examples) == 1
    assert output_path.exists()


def test_prepare_excludes_declared_secular_categories_and_reward_eval(tmp_path):
    db_path = tmp_path / "training.db"
    build_test_db(db_path)

    output_path = tmp_path / "out.jsonl"
    examples = consequence_prepare.prepare(local=str(db_path), output=str(output_path))

    ids = {example["id"] for example in examples}
    assert ids == {"ex-1"}


def test_sqlite_readonly_uri_uses_immutable_mode():
    uri = consequence_prepare.sqlite_readonly_uri("/tmp/training.db")
    assert "mode=ro" in uri
    assert "immutable=1" in uri


def test_sql_string_literal_escapes_single_quotes():
    assert consequence_prepare.sql_string_literal("teacher's note") == "'teacher''s note'"


def test_prepare_skips_examples_with_empty_user_turn(tmp_path):
    db_path = tmp_path / "training.db"
    build_test_db(db_path)

    conn = sqlite3.connect(db_path)
    empty_user_conversations = json.dumps(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "valid answer"},
        ]
    )
    conn.execute(
        """
        INSERT INTO examples
        (id, category, source, conversations, tier, reasoning, role, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("ex-4", "general", "test", empty_user_conversations, "secular", None, "conversational", "accepted"),
    )
    conn.commit()
    conn.close()

    output_path = tmp_path / "out.jsonl"
    examples = consequence_prepare.prepare(local=str(db_path), output=str(output_path))

    ids = {example["id"] for example in examples}
    assert "ex-4" not in ids
