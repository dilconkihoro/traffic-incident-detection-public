"""Role-based access through the HTTP API: what each of administrator, traffic operator and
analyst may and may not do, and how account changes take effect."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webapp.backend import db
from webapp.backend.tests.test_api import configure_env

ADMIN_PASSWORD = "smoke-test-password-123"


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    configure_env(tmp_path, monkeypatch)
    from webapp.backend.main import create_app

    with TestClient(create_app()) as client:
        yield client


def session(app_client, username: str, password: str) -> TestClient:
    """A separate client (own cookie jar) logged in as ``username``, sharing the same app."""
    client = TestClient(app_client.app)
    response = client.post("/api/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def admin(app_client):
    return session(app_client, "admin", ADMIN_PASSWORD)


def make_user(admin, username: str, role: str, password: str = "user-password-1") -> dict:
    response = admin.post(
        "/api/users", json={"username": username, "password": password, "role": role}
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def operator(app_client, admin):
    make_user(admin, "olive", "operator")
    return session(app_client, "olive", "user-password-1")


@pytest.fixture
def analyst(app_client, admin):
    make_user(admin, "andy", "analyst")
    return session(app_client, "andy", "user-password-1")


def test_me_reports_the_logged_in_user_and_role(admin, operator, analyst):
    assert admin.get("/api/me").json() == {"username": "admin", "role": "admin"}
    assert operator.get("/api/me").json() == {"username": "olive", "role": "operator"}
    assert analyst.get("/api/me").json() == {"username": "andy", "role": "analyst"}


def test_me_without_a_session_is_refused(app_client):
    assert TestClient(app_client.app).get("/api/me").status_code == 401


def test_admin_creates_users_and_the_list_never_exposes_password_hashes(admin):
    created = make_user(admin, "olive", "operator")
    assert created["role"] == "operator" and created["active"] is True
    assert created["created_by"] == "admin"

    listed = admin.get("/api/users").json()

    assert {u["username"] for u in listed} == {"admin", "olive"}
    assert all("password_hash" not in u for u in listed)
    assert next(u for u in listed if u["username"] == "admin")["is_bootstrap"] is True


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"username": "x", "password": "short", "role": "operator"}, 422),
        ({"username": "x", "password": "long-enough-1", "role": "superuser"}, 422),
        ({"username": "  ", "password": "long-enough-1", "role": "analyst"}, 422),
        ({"username": "x", "password": "y" * 73, "role": "analyst"}, 422),
    ],
)
def test_invalid_new_users_are_rejected(admin, payload, expected):
    assert admin.post("/api/users", json=payload).status_code == expected


def test_a_duplicate_username_is_rejected_ignoring_case(admin):
    make_user(admin, "Olive", "operator")
    response = admin.post(
        "/api/users", json={"username": "olive", "password": "long-enough-1", "role": "analyst"}
    )
    assert response.status_code == 409


def test_login_is_case_insensitive_on_the_username(app_client, admin):
    make_user(admin, "Olive", "operator")
    assert session(app_client, "OLIVE", "user-password-1").get("/api/me").json()["role"] == (
        "operator"
    )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/users"),
        ("post", "/api/users"),
        ("patch", "/api/users/1"),
    ],
)
def test_only_admins_manage_users(operator, analyst, method, path):
    for client in (operator, analyst):
        response = client.request(method.upper(), path, json=None if method == "get" else {})
        assert response.status_code == 403


def test_analyst_is_read_only(analyst, admin):
    db_conn = analyst.app.state.db
    db.create_run(db_conn, "r1", "clip.mp4", "2026-01-01T00:00:00Z", created_by="admin")
    db.update_run(db_conn, "r1", status="running")

    assert analyst.get("/api/runs").status_code == 200
    assert analyst.get("/api/runs/r1").status_code == 200
    assert analyst.post("/api/uploads", files={"file": ("a.mp4", b"x")}).status_code == 403
    assert analyst.get("/api/models").status_code == 403
    assert (
        analyst.post("/api/runs", json={"upload_id": "u", "path": "p", "filename": "f"}).status_code
        == 403
    )
    assert analyst.post("/api/runs/r1/cancel").status_code == 403


def test_operator_may_upload_and_reaches_the_upload_checks(operator):
    response = operator.post("/api/uploads", files={"file": ("notes.txt", b"x")})
    assert response.status_code == 400  # past the role check, refused by the file-type check


def test_operator_cancels_only_their_own_runs(operator):
    db_conn = operator.app.state.db
    for job_id, owner in (("mine", "olive"), ("theirs", "admin")):
        db.create_run(db_conn, job_id, "clip.mp4", "2026-01-01T00:00:00Z", created_by=owner)
        db.update_run(db_conn, job_id, status="running")

    assert operator.post("/api/runs/theirs/cancel").status_code == 403
    assert operator.post("/api/runs/mine/cancel").status_code == 200


def test_admin_cancels_any_run(admin):
    db_conn = admin.app.state.db
    db.create_run(db_conn, "theirs", "clip.mp4", "2026-01-01T00:00:00Z", created_by="olive")
    db.update_run(db_conn, "theirs", status="running")
    assert admin.post("/api/runs/theirs/cancel").status_code == 200


def test_a_run_records_who_started_it_and_cannot_point_outside_uploads(operator, tmp_path):
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"x")
    response = operator.post(
        "/api/runs",
        json={"upload_id": "u", "path": str(outside), "filename": "elsewhere.mp4"},
    )
    assert response.status_code == 404


def test_disabling_a_user_ends_their_session_and_blocks_login(app_client, admin):
    user = make_user(admin, "olive", "operator")
    olive = session(app_client, "olive", "user-password-1")

    assert admin.patch(f"/api/users/{user['id']}", json={"active": False}).status_code == 200

    assert olive.get("/api/runs").status_code == 401
    refused = TestClient(app_client.app).post(
        "/api/login", json={"username": "olive", "password": "user-password-1"}
    )
    assert refused.status_code == 401


def test_a_role_change_applies_on_the_next_request(app_client, admin):
    user = make_user(admin, "olive", "operator")
    olive = session(app_client, "olive", "user-password-1")
    assert olive.get("/api/models").status_code == 200

    admin.patch(f"/api/users/{user['id']}", json={"role": "analyst"})

    assert olive.get("/api/models").status_code == 403
    assert olive.get("/api/me").json()["role"] == "analyst"


def test_a_password_reset_ends_existing_sessions(app_client, admin):
    user = make_user(admin, "olive", "operator")
    olive = session(app_client, "olive", "user-password-1")

    admin.patch(f"/api/users/{user['id']}", json={"password": "a-new-password-2"})

    assert olive.get("/api/me").status_code == 401
    assert session(app_client, "olive", "a-new-password-2").get("/api/me").status_code == 200


def test_the_bootstrap_admin_cannot_be_demoted_disabled_or_reset_from_the_dashboard(
    app_client, admin
):
    make_user(admin, "second", "admin")
    other_admin = session(app_client, "second", "user-password-1")
    bootstrap_id = next(u["id"] for u in admin.get("/api/users").json() if u["is_bootstrap"])

    for change in ({"role": "analyst"}, {"active": False}, {"password": "new-password-3"}):
        assert other_admin.patch(f"/api/users/{bootstrap_id}", json=change).status_code == 409


def test_an_admin_cannot_demote_or_disable_themselves(app_client, admin):
    me = make_user(admin, "second", "admin")
    second = session(app_client, "second", "user-password-1")
    assert second.patch(f"/api/users/{me['id']}", json={"role": "analyst"}).status_code == 409
    assert second.patch(f"/api/users/{me['id']}", json={"active": False}).status_code == 409


def test_updating_an_unknown_user_is_a_404(admin):
    assert admin.patch("/api/users/9999", json={"role": "analyst"}).status_code == 404


def test_changing_the_env_admin_name_retires_the_previous_bootstrap_account(tmp_path):
    conn = db.connect(tmp_path / "roles.db")
    db.ensure_bootstrap_admin(conn, "first", "hash-1", "2026-01-01")
    db.ensure_bootstrap_admin(conn, "second", "hash-2", "2026-01-02")

    first, second = db.get_user(conn, "first"), db.get_user(conn, "second")

    assert (first["active"], first["is_bootstrap"]) == (0, 0)
    assert (second["active"], second["is_bootstrap"], second["role"]) == (1, 1, "admin")


def test_an_existing_runs_table_gains_the_created_by_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, video_filename TEXT NOT NULL, "
        "created_at TEXT NOT NULL, status TEXT NOT NULL, run_id TEXT, "
        "incident_count INTEGER NOT NULL DEFAULT 0, frames_total INTEGER, "
        "frames_done INTEGER NOT NULL DEFAULT 0, error TEXT)"
    )
    old.commit()
    old.close()

    conn = db.connect(path)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
    assert {"upload_path", "created_by"} <= columns
