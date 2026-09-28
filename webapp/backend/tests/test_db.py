"""The SQLite run index: an embedded single file, never a second source of truth for incidents."""

from __future__ import annotations

import pytest

from webapp.backend import db


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "webapp.db")
    yield connection
    connection.close()


def test_create_and_get_run(conn):
    db.create_run(conn, "job1", "clip.mp4", "2026-09-23T00:00:00Z")
    row = db.get_run(conn, "job1")
    assert row["status"] == "queued"
    assert row["video_filename"] == "clip.mp4"
    assert row["incident_count"] == 0
    assert row["upload_path"] is None


def test_create_run_stores_the_upload_path(conn):
    db.create_run(conn, "job1", "clip.mp4", "2026-09-23T00:00:00Z", "/data/webapp_uploads/x.mp4")
    row = db.get_run(conn, "job1")
    assert row["upload_path"] == "/data/webapp_uploads/x.mp4"


def test_connecting_to_an_existing_pre_upload_path_database_adds_the_column(tmp_path):
    """A migration test: an older webapp.db created before upload_path existed must still work,
    not lose its run history just because a column was added later."""
    import sqlite3

    path = tmp_path / "old.db"
    old_conn = sqlite3.connect(path)
    old_conn.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, video_filename TEXT NOT NULL, "
        "created_at TEXT NOT NULL, status TEXT NOT NULL, run_id TEXT, "
        "incident_count INTEGER NOT NULL DEFAULT 0, frames_total INTEGER, "
        "frames_done INTEGER NOT NULL DEFAULT 0, error TEXT)"
    )
    old_conn.execute(
        "INSERT INTO runs (id, video_filename, created_at, status) VALUES (?, ?, ?, 'done')",
        ("job1", "old.mp4", "2026-09-23T00:00:00Z"),
    )
    old_conn.commit()
    old_conn.close()

    migrated = db.connect(path)
    row = db.get_run(migrated, "job1")
    assert row["video_filename"] == "old.mp4"
    assert row["upload_path"] is None
    migrated.close()


def test_get_unknown_run_returns_none(conn):
    assert db.get_run(conn, "does-not-exist") is None


def test_update_run_changes_only_named_fields(conn):
    db.create_run(conn, "job1", "clip.mp4", "2026-09-23T00:00:00Z")
    db.update_run(conn, "job1", status="running", frames_done=5)
    row = db.get_run(conn, "job1")
    assert row["status"] == "running"
    assert row["frames_done"] == 5
    assert row["video_filename"] == "clip.mp4"


def test_update_run_rejects_an_unknown_column(conn):
    """The column list is fixed and never built from a caller-supplied string."""
    db.create_run(conn, "job1", "clip.mp4", "2026-09-23T00:00:00Z")
    with pytest.raises(ValueError, match="not updatable"):
        db.update_run(conn, "job1", video_filename="renamed.mp4")


def test_list_runs_orders_newest_first(conn):
    db.create_run(conn, "job1", "a.mp4", "2026-09-23T00:00:00Z")
    db.create_run(conn, "job2", "b.mp4", "2026-09-23T00:01:00Z")
    rows = db.list_runs(conn)
    assert [r["id"] for r in rows] == ["job2", "job1"]


def test_fail_interrupted_runs_only_touches_unfinished_runs(conn):
    db.create_run(conn, "queued1", "a.mp4", "2026-09-23T00:00:00Z")
    db.create_run(conn, "running1", "b.mp4", "2026-09-23T00:01:00Z")
    db.update_run(conn, "running1", status="running")
    db.create_run(conn, "done1", "c.mp4", "2026-09-23T00:02:00Z")
    db.update_run(conn, "done1", status="done", run_id="r1")
    db.create_run(conn, "failed1", "d.mp4", "2026-09-23T00:03:00Z")
    db.update_run(conn, "failed1", status="failed", error="its own reason")

    assert db.fail_interrupted_runs(conn) == 2

    assert db.get_run(conn, "queued1")["status"] == "failed"
    assert db.get_run(conn, "running1")["error"] == db.INTERRUPTED_MESSAGE
    assert db.get_run(conn, "done1")["status"] == "done"
    assert db.get_run(conn, "failed1")["error"] == "its own reason"


def test_status_check_constraint_rejects_an_invalid_status(conn):
    db.create_run(conn, "job1", "a.mp4", "2026-09-23T00:00:00Z")
    with pytest.raises(Exception, match="CHECK constraint failed"):
        db.update_run(conn, "job1", status="not-a-real-status")
