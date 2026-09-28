"""SQLite index of runs: an embedded single file, not a database server (specification 3.7 excludes a
database server for the core CLI scope; this supplementary layer is a different thing, and even
so stays as light as possible). The source of truth for incident data is always
reports/<run_id>/incidents.csv; this index only speeds up listing runs for the dashboard."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    video_filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    run_id TEXT,
    incident_count INTEGER NOT NULL DEFAULT 0,
    frames_total INTEGER,
    frames_done INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    upload_path TEXT
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS does not add columns to a table that already exists (an
    existing local webapp.db from before upload_path was added); add it if missing so upgrading
    never loses a user's run history."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    if "upload_path" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN upload_path TEXT")
        conn.commit()


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)
    return conn


@contextmanager
def cursor(conn: sqlite3.Connection):
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def create_run(
    conn: sqlite3.Connection,
    job_id: str,
    video_filename: str,
    created_at: str,
    upload_path: str | None = None,
) -> None:
    with cursor(conn) as cur:
        cur.execute(
            "INSERT INTO runs (id, video_filename, created_at, status, upload_path) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (job_id, video_filename, created_at, upload_path),
        )


CANCELLED_MESSAGE = "cancelled: stopped by the user"

INTERRUPTED_MESSAGE = (
    "interrupted: the server was stopped or restarted before this run finished; "
    "upload the video again to re-run it"
)


def fail_interrupted_runs(conn: sqlite3.Connection) -> int:
    """Runs execute in threads inside the server process, so a run still queued or running when
    the server starts was cut off by a stop or restart and can never finish; without this it would
    sit as 'running' forever with a live-progress page that has nothing to connect to."""
    with cursor(conn) as cur:
        cur.execute(
            "UPDATE runs SET status = 'failed', error = ? WHERE status IN ('queued', 'running')",
            (INTERRUPTED_MESSAGE,),
        )
        return cur.rowcount


_UPDATABLE_COLUMNS = {"status", "run_id", "incident_count", "frames_total", "frames_done", "error"}


def update_run(conn: sqlite3.Connection, job_id: str, **fields: object) -> None:
    """``fields`` keys must be column names from ``_UPDATABLE_COLUMNS``, never a caller-supplied
    string, so the generated SQL never carries anything an untrusted caller chose."""
    if not fields:
        return
    unknown = set(fields) - _UPDATABLE_COLUMNS
    if unknown:
        raise ValueError(f"not updatable: {sorted(unknown)}")
    columns = ", ".join(f"{key} = ?" for key in fields)
    with cursor(conn) as cur:
        cur.execute(f"UPDATE runs SET {columns} WHERE id = ?", (*fields.values(), job_id))


def get_run(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    with cursor(conn) as cur:
        cur.execute("SELECT * FROM runs WHERE id = ?", (job_id,))
        return cur.fetchone()


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    with cursor(conn) as cur:
        cur.execute("SELECT * FROM runs ORDER BY created_at DESC")
        return cur.fetchall()
