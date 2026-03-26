import json
import importlib.util
import sqlite3
from pathlib import Path

from teapot import validate_compose as validate_compose_mod
from teapot.eval.orchestrator import TIER_ORDER, load_module_evals


MODULE_PATH = Path("modules/capability/reward-evaluator/prepare.py")
SPEC = importlib.util.spec_from_file_location("reward_evaluator_prepare", MODULE_PATH)
reward_prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reward_prepare)


def build_reward_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE system_prompts (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute(
        """
        CREATE TABLE examples (
            id TEXT PRIMARY KEY,
            category TEXT,
            source TEXT,
            conversations TEXT,
            role TEXT,
            status TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO system_prompts VALUES (?, ?)",
        ("reward-evaluator-v1", "reward prompt"),
    )
    conversations = json.dumps(
        [
            {"role": "user", "content": "score this exchange"},
            {"role": "assistant", "content": "evaluation output"},
        ]
    )
    conn.execute(
        """
        INSERT INTO examples
        (id, category, source, conversations, role, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("reward-1", "reward-evaluation", "test", conversations, "reward-evaluator", "accepted"),
    )
    conn.commit()
    conn.close()


def test_reward_prepare_reads_sqlite_in_readonly_mode(tmp_path):
    db_path = tmp_path / "training.db"
    build_reward_db(db_path)
    db_path.chmod(0o444)

    output_path = tmp_path / "reward.jsonl"
    examples = reward_prepare.prepare(output=str(output_path), local=str(db_path))

    assert len(examples) == 1
    assert examples[0]["task"] == "reward-evaluation"
    assert output_path.exists()


def test_validate_compose_accepts_reward_dataset():
    errors, warnings = validate_compose_mod.check_content(
        [
            {
                "id": "reward-1",
                "module": "capability/reward-evaluator",
                "task": "reward-evaluation",
                "category": "reward-evaluation",
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "score"},
                ],
            }
        ]
    )

    assert errors == 0
    assert warnings == 0


def test_validate_compose_rejects_reward_rows_in_conversational_dataset():
    errors, warnings = validate_compose_mod.check_content(
        [
            {
                "id": "reward-1",
                "category": "reward-evaluation",
                "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "score"},
                ],
            }
        ]
    )

    assert errors == 1
    assert warnings == 0


def test_reward_module_eval_declares_existing_format_script():
    tests = load_module_evals("capability/reward-evaluator", TIER_ORDER["fast"])
    eval_test = next(test for test in tests if test["script"] == "eval/test_format.py")
    assert eval_test["pass_criteria"]["clean"] == "true"
