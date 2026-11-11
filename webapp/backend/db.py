"""SQLite store for the dashboard: an embedded single file, not a database server.
The source of truth for incident data is always reports/<run_id>/incidents.csv; the runs table
only speeds up listing runs. The users table holds dashboard accounts and their roles."""

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
    upload_path TEXT,
    created_by TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'operator', 'analyst')),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    is_bootstrap INTEGER NOT NULL DEFAULT 0 CHECK (is_bootstrap IN (0, 1)),
    created_at TEXT NOT NULL,
    created_by TEXT
);
"""

ROLES = ("admin", "operator", "analyst")


def _migrate(conn: sqlite3.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS does not add columns to a table that already exists (an
    existing local webapp.db from before upload_path was added); add it if missing so upgrading
    never loses a user's run history."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    for column in ("upload_path", "created_by"):
        if column not in columns:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
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
    created_by: str | None = None,
) -> None:
    with cursor(conn) as cur:
        cur.execute(
            "INSERT INTO runs (id, video_filename, created_at, status, upload_path, created_by) "
            "VALUES (?, ?, ?, 'queued', ?, ?)",
            (job_id, video_filename, created_at, upload_path, created_by),
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


class UserExistsError(ValueError):
    pass


def ensure_bootstrap_admin(
    conn: sqlite3.Connection, username: str, password_hash: str, created_at: str
) -> None:
    """The admin named in the environment always exists, is active, and holds the hash from the
    environment, so the dashboard can never be left without an administrator. A previous
    bootstrap account under another name is deactivated, because the environment no longer
    vouches for it."""
    with cursor(conn) as cur:
        cur.execute(
            "UPDATE users SET active = 0, is_bootstrap = 0 "
            "WHERE is_bootstrap = 1 AND username != ? COLLATE NOCASE",
            (username,),
        )
        cur.execute(
            "INSERT INTO users (username, password_hash, role, active, is_bootstrap, created_at) "
            "VALUES (?, ?, 'admin', 1, 1, ?) "
            "ON CONFLICT(username) DO UPDATE SET password_hash = excluded.password_hash, "
            "role = 'admin', active = 1, is_bootstrap = 1",
            (username, password_hash, created_at),
        )


def get_user(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    with cursor(conn) as cur:
        cur.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,))
        return cur.fetchone()


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    with cursor(conn) as cur:
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def list_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    with cursor(conn) as cur:
        cur.execute("SELECT * FROM users ORDER BY username COLLATE NOCASE")
        return cur.fetchall()


def create_user(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    role: str,
    created_at: str,
    created_by: str,
) -> int:
    try:
        with cursor(conn) as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role, created_at, created_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, role, created_at, created_by),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise UserExistsError(f"a user named {username!r} already exists") from exc


_UPDATABLE_USER_COLUMNS = {"role", "active", "password_hash"}


def update_user(conn: sqlite3.Connection, user_id: int, **fields: object) -> None:
    """Same contract as update_run: keys come from a fixed allow-list, never from a caller."""
    if not fields:
        return
    unknown = set(fields) - _UPDATABLE_USER_COLUMNS
    if unknown:
        raise ValueError(f"not updatable: {sorted(unknown)}")
    columns = ", ".join(f"{key} = ?" for key in fields)
    with cursor(conn) as cur:
        cur.execute(f"UPDATE users SET {columns} WHERE id = ?", (*fields.values(), user_id))
