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
