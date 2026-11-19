"""SQLite: схема, подключение, запись аудита."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snippets (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    snippet_text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snippets_doc ON snippets(document_id);
CREATE TABLE IF NOT EXISTS qa_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    error TEXT,
    confidence TEXT,
    provider TEXT,
    raw_output TEXT,
    duration_ms INTEGER
);
CREATE TABLE IF NOT EXISTS audit_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    input TEXT,
    output TEXT,
    status TEXT NOT NULL,
    error TEXT,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_runs(created_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@contextmanager
def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def _clip(value, limit=20_000) -> str | None:
    if value is None:
        return None
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…[обрезано]"


def audit(action: str, inp, output, status: str, error: str | None, duration_ms: int) -> str:
    """Запись в audit_runs идёт отдельным подключением: ошибка основной операции аудит не откатит."""
    rid = new_id("aud")
    with connect() as c:
        c.execute(
            "INSERT INTO audit_runs(id, created_at, action, input, output, status, error, duration_ms)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (rid, now(), action, _clip(inp), _clip(output), status, error, duration_ms),
        )
    return rid
