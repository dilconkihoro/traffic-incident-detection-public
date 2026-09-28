"""The job runner against a scripted (ground-truth) detector, not the real YOLO model, the same
pattern tests/synth.py already uses for the CLI's own tests. Not instant even so: DeepSORT's
embedder still runs a real network per frame, so these take real seconds, just not the ~1 minute
a real detector run over the same clip needs."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from synth import ScriptedDetector, abrupt_stop_paths, render_scene, write_video  # noqa: E402
from trafficwatch.config import load_config
from webapp.backend import db
from webapp.backend.jobs import Job

SHIPPED_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "pipeline.yaml"


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "webapp.db")
    yield connection
    connection.close()


def wait_for(job: Job, kind: str, timeout: float = 60.0):
    """DeepSORT's embedder runs a real network per frame even with a fake detector, so this
    needs real wall-clock budget, not a short unit-test timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = job.events.get(timeout=max(0.1, deadline - time.monotonic()))
        except queue.Empty:
            break
        if event.kind == kind:
            return event
    raise AssertionError(f"timed out waiting for a {kind!r} event")


def test_job_runs_a_real_pipeline_with_a_fake_detector_and_updates_the_db(tmp_path, conn):
    frames, detections = render_scene(abrupt_stop_paths())
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    db.create_run(conn, "job1", "scene.mp4", "2026-09-23T00:00:00Z")

    job = Job("job1", clip)
    job.start(
        conn,
        SHIPPED_CONFIG,
        tmp_path / "reports",
        standin=True,
        detector=ScriptedDetector(detections),
    )

    done = wait_for(job, "done")
    assert done.data["run_id"]
    row = db.get_run(conn, "job1")
    assert row["status"] == "done"
    assert row["frames_done"] == 51
    assert row["incident_count"] == 1
    video_path = tmp_path / "reports" / row["run_id"] / "annotated.mp4"
    assert video_path.is_file()

    # A browser cannot decode cv2.VideoWriter's raw mp4v output; the job runner must re-encode it
    # to H.264 before marking the run done, or the dashboard's video player will show nothing.
    probe = subprocess.run(
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
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "h264"


def test_job_reports_incident_events_as_they_are_flagged(tmp_path, conn):
    frames, detections = render_scene(abrupt_stop_paths())
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    db.create_run(conn, "job1", "scene.mp4", "2026-09-23T00:00:00Z")

    job = Job("job1", clip)
    job.start(
        conn,
        SHIPPED_CONFIG,
        tmp_path / "reports",
        standin=True,
        detector=ScriptedDetector(detections),
    )

    incident_event = wait_for(job, "incident")
    assert incident_event.data["type"] == "ABRUPT_STOP"
    # Always wait for the job to fully finish before the test returns: its thread keeps writing
    # to `conn` until then, and pytest closes that connection the moment this function returns.
    wait_for(job, "done")


def busy_slot() -> threading.Semaphore:
    slots = threading.Semaphore(1)
    slots.acquire()
    return slots


def test_a_run_waits_as_queued_while_another_holds_the_only_slot_then_runs(tmp_path, conn):
    frames, detections = render_scene(abrupt_stop_paths())
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    db.create_run(conn, "job1", "scene.mp4", "2026-09-23T00:00:00Z")
    slots = busy_slot()

    job = Job("job1", clip, slots=slots)
    job.start(
        conn,
        SHIPPED_CONFIG,
        tmp_path / "reports",
        standin=True,
        detector=ScriptedDetector(detections),
    )

    time.sleep(1.5)
    assert db.get_run(conn, "job1")["status"] == "queued"
    assert job.events.empty()

    slots.release()
    wait_for(job, "done")
    assert db.get_run(conn, "job1")["status"] == "done"


def test_cancelling_a_queued_run_fails_it_without_ever_running(tmp_path, conn):
    frames, detections = render_scene(abrupt_stop_paths())
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    db.create_run(conn, "job1", "scene.mp4", "2026-09-23T00:00:00Z")

    job = Job("job1", clip, slots=busy_slot())
    job.start(
        conn,
        SHIPPED_CONFIG,
        tmp_path / "reports",
        standin=True,
        detector=ScriptedDetector(detections),
    )
    job.cancel()

    failed = wait_for(job, "failed")
    assert failed.data["error"] == db.CANCELLED_MESSAGE
    row = db.get_run(conn, "job1")
    assert row["status"] == "failed" and row["error"] == db.CANCELLED_MESSAGE
    assert not (tmp_path / "reports").exists()


def test_cancelling_a_running_run_stops_it_frees_the_slot_and_removes_partial_output(
    tmp_path, conn
):
    frames, detections = render_scene(abrupt_stop_paths())
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    db.create_run(conn, "job1", "scene.mp4", "2026-09-23T00:00:00Z")
    slots = threading.Semaphore(1)

    job = Job("job1", clip, slots=slots)
    job.start(
        conn,
        SHIPPED_CONFIG,
        tmp_path / "reports",
        standin=True,
        detector=ScriptedDetector(detections),
    )
    wait_for(job, "frame")
    job.cancel()

    failed = wait_for(job, "failed")
    assert failed.data["error"] == db.CANCELLED_MESSAGE
    row = db.get_run(conn, "job1")
    assert row["status"] == "failed" and row["run_id"] is None
    assert row["frames_done"] < 51
    reports = tmp_path / "reports"
    assert not reports.exists() or not list(reports.glob("*_scene"))
    assert slots.acquire(timeout=5), "the cancelled run must give its slot back"


def test_job_reports_failure_without_crashing_the_thread(tmp_path, conn):
    db.create_run(conn, "job1", "missing.mp4", "2026-09-23T00:00:00Z")
    job = Job("job1", tmp_path / "does-not-exist.mp4")
    job.start(conn, SHIPPED_CONFIG, tmp_path / "reports", standin=True, detector=object())

    failed = wait_for(job, "failed")
    assert failed.data["error"]
    row = db.get_run(conn, "job1")
    assert row["status"] == "failed"
    assert row["error"]


def test_shipped_config_loads_in_standin_mode():
    """Sanity check that the fixture path used by every test above is real and current."""
    cfg = load_config(SHIPPED_CONFIG, standin=True)
    assert cfg.detector.weights_kind == "coco_standin"
