"""The HTTP API end to end. Fast tests use no real model; one slow test (marked ``standin``,
matching the repo's existing convention) proves the full stack against the real stand-in model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from synth import write_video  # noqa: E402
from webapp.backend import db  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_SAMPLE = REPO_ROOT / "data" / "samples" / "MVI_20011_short.mp4"
STANDIN_WEIGHTS = REPO_ROOT / "weights" / "coco_standin" / "yolov8n.pt"


def configure_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TW_WEBAPP_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("TW_ADMIN_USERNAME", "admin")
    # bcrypt hash of "smoke-test-password-123", generated once for this fixture
    monkeypatch.setenv(
        "TW_ADMIN_PASSWORD_HASH",
        "$2b$12$cQga.NXHC4S7y0QBzLUfnuz58yclBYNZrvwD6bhQL4YmUAmk88bxa",
    )
    monkeypatch.setenv("TW_WEBAPP_DB", str(tmp_path / "webapp.db"))
    monkeypatch.setenv("TW_WEBAPP_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("TW_RUNS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("TW_LOGIN_RATE_LIMIT_ATTEMPTS", "3")


@pytest.fixture
def client(tmp_path, monkeypatch):
    configure_env(tmp_path, monkeypatch)

    from webapp.backend.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_runs_left_running_by_a_previous_server_are_marked_failed_on_startup(tmp_path, monkeypatch):
    configure_env(tmp_path, monkeypatch)
    earlier = db.connect(tmp_path / "webapp.db")
    db.create_run(earlier, "cut-off", "long.mp4", "2026-09-28T00:00:00Z")
    db.update_run(earlier, "cut-off", status="running", frames_done=4286, frames_total=4425)
    db.create_run(earlier, "finished", "ok.mp4", "2026-09-28T00:01:00Z")
    db.update_run(earlier, "finished", status="done", run_id="r1")
    earlier.close()

    from webapp.backend.main import create_app

    with TestClient(create_app()) as restarted:
        restarted.post(
            "/api/login", json={"username": "admin", "password": "smoke-test-password-123"}
        )
        cut_off = restarted.get("/api/runs/cut-off").json()
        finished = restarted.get("/api/runs/finished").json()

    assert cut_off["status"] == "failed"
    assert "interrupted" in cut_off["error"]
    assert finished["status"] == "done"


def test_protected_route_without_a_session_is_refused(client):
    response = client.get("/api/runs")
    assert response.status_code == 401


def test_login_with_wrong_password_is_refused_and_does_not_leak_which_field(client):
    response = client.post("/api/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 401
    assert "invalid username or password" in response.json()["detail"]


def test_login_with_wrong_username_is_refused_the_same_way(client):
    response = client.post(
        "/api/login", json={"username": "not-admin", "password": "smoke-test-password-123"}
    )
    assert response.status_code == 401


def test_correct_login_sets_a_session_and_unlocks_protected_routes(client):
    response = client.post(
        "/api/login", json={"username": "admin", "password": "smoke-test-password-123"}
    )
    assert response.status_code == 200
    assert client.get("/api/runs").status_code == 200


def test_repeated_bad_logins_are_rate_limited(client):
    for _ in range(3):
        client.post("/api/login", json={"username": "admin", "password": "wrong"})
    response = client.post("/api/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 429


def test_a_correct_login_does_not_spend_rate_limit_budget(client):
    for _ in range(5):
        response = client.post(
            "/api/login", json={"username": "admin", "password": "smoke-test-password-123"}
        )
        assert response.status_code == 200


def login(client) -> None:
    client.post("/api/login", json={"username": "admin", "password": "smoke-test-password-123"})


def test_upload_rejects_an_unsupported_extension(client):
    login(client)
    response = client.post(
        "/api/uploads", files={"file": ("clip.txt", b"not a video", "text/plain")}
    )
    assert response.status_code == 400


def test_upload_rejects_an_empty_file(client):
    login(client)
    response = client.post("/api/uploads", files={"file": ("clip.mp4", b"", "video/mp4")})
    assert response.status_code == 400


def test_upload_rejects_a_video_over_the_frame_limit(client, tmp_path):
    login(client)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(200)]
    clip = write_video(tmp_path / "long.mp4", frames, fps=25.0)
    with open(clip, "rb") as handle:
        response = client.post("/api/uploads", files={"file": ("long.mp4", handle, "video/mp4")})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "200 frames" in detail
    assert "180" in detail


def test_upload_requires_a_session(client):
    response = client.post("/api/uploads", files={"file": ("clip.mp4", b"x", "video/mp4")})
    assert response.status_code == 401


def test_getting_a_run_that_does_not_exist_is_a_404(client):
    login(client)
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.get("/api/runs/does-not-exist/video").status_code == 404
    assert client.get("/api/runs/does-not-exist/incidents.csv").status_code == 404
    assert client.get("/api/runs/does-not-exist/tracks.csv").status_code == 404
    assert client.get("/api/runs/does-not-exist/original").status_code == 404


def probe_codec(path: Path) -> str:
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_an_older_run_still_shows_both_videos_playably(client, tmp_path):
    """A run made before upload_path was stored and before videos were normalised: its original
    is found through the run manifest, and both videos are converted for the browser on demand
    rather than showing the "no supported format" error."""
    login(client)
    settings = client.app.state.settings
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    original = write_video(settings.uploads_dir / "legacy.mp4", frames, fps=10.0)

    run_id = "20260101T000000Z_coco_standin_legacy"
    report_dir = settings.reports_dir / run_id
    report_dir.mkdir(parents=True)
    write_video(report_dir / "annotated.mp4", frames, fps=10.0)
    (report_dir / "run_manifest.json").write_text(
        json.dumps({"video": {"path": str(original)}}), encoding="utf-8"
    )
    db_conn = client.app.state.db
    db.create_run(db_conn, "legacyjob", "old.mp4", "2026-01-01T00:00:00Z")
    db.update_run(db_conn, "legacyjob", status="done", run_id=run_id)

    for route in ("original", "video"):
        response = client.get(f"/api/runs/legacyjob/{route}")
        assert response.status_code == 200, route
        served = tmp_path / f"{route}.mp4"
        served.write_bytes(response.content)
        assert probe_codec(served) == "h264", route


def test_cancelling_an_unknown_or_finished_run_is_refused(client):
    login(client)
    assert client.post("/api/runs/does-not-exist/cancel").status_code == 404

    db_conn = client.app.state.db
    db.create_run(db_conn, "finished", "ok.mp4", "2026-01-01T00:00:00Z")
    db.update_run(db_conn, "finished", status="done", run_id="r1")
    assert client.post("/api/runs/finished/cancel").status_code == 409


def test_cancelling_a_run_nothing_is_executing_clears_it_instead_of_leaving_it_stuck(client):
    """The stuck page from a run whose server was restarted: nothing runs it, so cancel must
    still work and mark it, rather than doing nothing forever."""
    login(client)
    db_conn = client.app.state.db
    db.create_run(db_conn, "orphan", "long.mp4", "2026-01-01T00:00:00Z")
    db.update_run(db_conn, "orphan", status="running", frames_done=4286, frames_total=4425)

    response = client.post("/api/runs/orphan/cancel")

    assert response.status_code == 200 and response.json() == {"status": "cancelled"}
    row = client.get("/api/runs/orphan").json()
    assert row["status"] == "failed" and row["error"] == db.CANCELLED_MESSAGE


def test_cancelling_a_live_run_asks_its_job_to_stop(client):
    login(client)
    db_conn = client.app.state.db
    db.create_run(db_conn, "live", "clip.mp4", "2026-01-01T00:00:00Z")
    db.update_run(db_conn, "live", status="running")
    job = client.app.state.jobs.create("live", Path("unused.mp4"))
    assert not job.cancel_requested

    response = client.post("/api/runs/live/cancel")

    assert response.status_code == 200 and response.json() == {"status": "cancelling"}
    assert job.cancel_requested


def test_cancel_requires_a_session(client):
    assert client.post("/api/runs/anything/cancel").status_code == 401


def test_a_run_with_no_recorded_original_reports_it_missing_instead_of_crashing(client):
    login(client)
    db_conn = client.app.state.db
    db.create_run(db_conn, "nooriginal", "x.mp4", "2026-01-01T00:00:00Z")
    db.update_run(db_conn, "nooriginal", status="done", run_id="no-such-report")
    assert client.get("/api/runs/nooriginal/original").status_code == 404


@pytest.mark.standin
def test_full_flow_upload_run_and_fetch_results_with_the_real_model(client):
    if not STANDIN_WEIGHTS.is_file():
        pytest.skip("run scripts/fetch_standin_weights.py first")
    if not REAL_SAMPLE.is_file():
        pytest.skip("data/samples/MVI_20011_short.mp4 not present on this machine")
    login(client)

    with open(REAL_SAMPLE, "rb") as handle:
        upload = client.post("/api/uploads", files={"file": ("clip.mp4", handle, "video/mp4")})
    assert upload.status_code == 200
    body = upload.json()

    created = client.post(
        "/api/runs",
        json={
            "upload_id": body["upload_id"],
            "path": body["path"],
            "filename": body["filename"],
            "standin": True,
        },
    )
    assert created.status_code == 200
    job_id = created.json()["job_id"]

    with client.stream("GET", f"/api/runs/{job_id}/events") as stream:
        kinds = [
            line.split(": ", 1)[1] for line in stream.iter_lines() if line.startswith("event:")
        ]
    assert kinds[-1] == "done", f"stream ended with {kinds[-1]!r}, full sequence: {kinds}"
    assert kinds.count("frame") == 75

    detail = client.get(f"/api/runs/{job_id}").json()
    assert detail["status"] == "done"
    assert detail["frames_total"] == detail["frames_done"] == 75

    video = client.get(f"/api/runs/{job_id}/video")
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    csv = client.get(f"/api/runs/{job_id}/incidents.csv")
    assert csv.status_code == 200 and csv.text.startswith("incident_id,")
    tracks = client.get(f"/api/runs/{job_id}/tracks.csv")
    assert tracks.status_code == 200 and tracks.text.startswith("video_id,")
    original = client.get(f"/api/runs/{job_id}/original")
    assert original.status_code == 200 and original.headers["content-type"] == "video/mp4"
