"""Background job runner: wraps trafficwatch.pipeline.Pipeline exactly as the CLI does, adding
only a progress queue for the live view. No new queueing system (no Celery, no Redis): this is a
single-user local dashboard, one background thread per job is enough.

A run is CPU-bound and this dashboard runs on a machine with no GPU, so a run must not be able to
take the whole machine down: only ``max_concurrent`` runs execute at once (the rest wait as
"queued"), each run uses a capped number of CPU threads at lowered priority, and a run can be
cancelled at any point."""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import json
import os
import queue
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trafficwatch.config import load_config
from trafficwatch.pipeline import Pipeline
from trafficwatch.types import Incident
from webapp.backend import db
from webapp.backend.video import make_browser_playable

# How much less urgent than the desktop a run is (POSIX only; Windows has no os.nice).
NICE_INCREMENT = 10


@dataclass(frozen=True)
class ProgressEvent:
    kind: str  # "frame" | "incident" | "done" | "failed"
    data: dict[str, Any]


class RunCancelled(Exception):
    """Raised inside the pipeline's per-frame callback to stop a run the user cancelled."""


def _lower_priority_and_cap_threads(cpu_threads: int | None) -> None:
    """Called from the job's own thread: nice values and OpenMP thread counts are per thread on
    Linux, and this is the thread that runs the model."""
    if hasattr(os, "nice"):
        with contextlib.suppress(OSError):
            os.nice(NICE_INCREMENT)
    if cpu_threads is not None:
        import cv2
        import torch

        cv2.setNumThreads(cpu_threads)
        torch.set_num_threads(cpu_threads)


class Job:
    """One video's run: a background thread plus a queue the SSE endpoint reads from."""

    def __init__(
        self,
        job_id: str,
        video_path: Path,
        *,
        slots: threading.Semaphore | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        self.job_id = job_id
        self.video_path = video_path
        self.events: queue.Queue[ProgressEvent] = queue.Queue()
        self._slots = slots
        self._holds_slot = False
        self._cpu_threads = cpu_threads
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def start(
        self,
        conn: sqlite3.Connection,
        pipeline_config_path: Path,
        reports_dir: Path,
        standin: bool,
        *,
        detector: object | None = None,
    ) -> None:
        """``detector`` is a test-only override (see Pipeline's own constructor); production code
        never passes it, so a run always uses the real detector unless a test says otherwise."""
        self._thread = threading.Thread(
            target=self._run,
            args=(conn, pipeline_config_path, reports_dir, standin, detector),
            daemon=True,
        )
        self._thread.start()

    def _acquire_slot(self) -> None:
        if self._slots is None:
            return
        while not self._slots.acquire(timeout=0.5):
            if self._cancel.is_set():
                raise RunCancelled
        self._holds_slot = True

    def _release_slot(self) -> None:
        if self._holds_slot and self._slots is not None:
            self._slots.release()
            self._holds_slot = False

    def _fail(self, conn: sqlite3.Connection, message: str) -> None:
        db.update_run(conn, self.job_id, status="failed", error=message)
        self.events.put(ProgressEvent("failed", {"error": message}))

    def _discard_partial_output(self, reports_dir: Path) -> None:
        """A cancelled run leaves a half-written report folder (it never reaches latency.json)
        that nothing links to; the folder name ends with the upload's unique file stem."""
        if not reports_dir.is_dir():
            return
        for folder in reports_dir.glob(f"*_{glob.escape(self.video_path.stem)}"):
            if folder.is_dir() and not (folder / "latency.json").exists():
                shutil.rmtree(folder, ignore_errors=True)

    def _run(
        self,
        conn: sqlite3.Connection,
        pipeline_config_path: Path,
        reports_dir: Path,
        standin: bool,
        detector: object | None,
    ) -> None:
        try:
            self._acquire_slot()
            try:
                self._execute(conn, pipeline_config_path, reports_dir, standin, detector)
            finally:
                self._release_slot()
        except RunCancelled:
            self._discard_partial_output(reports_dir)
            self._fail(conn, db.CANCELLED_MESSAGE)
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed silently
            self._fail(conn, str(exc))

    def _execute(
        self,
        conn: sqlite3.Connection,
        pipeline_config_path: Path,
        reports_dir: Path,
        standin: bool,
        detector: object | None,
    ) -> None:
        if self._cancel.is_set():
            raise RunCancelled
        db.update_run(conn, self.job_id, status="running")
        _lower_priority_and_cap_threads(self._cpu_threads)
        cfg = load_config(pipeline_config_path, standin=standin)
        # The shipped pipeline.yaml's own output.dir is a relative "reports" path; the
        # webapp's results routes look under settings.reports_dir, which may differ (tests
        # point it at a tmp_path so runs never land in the real repo's reports/ folder).
        cfg = dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, dir=str(reports_dir)))
        pipeline = Pipeline(cfg, detector=detector)

        def on_frame(done: int, total: int | None) -> None:
            if self._cancel.is_set():
                raise RunCancelled
            db.update_run(conn, self.job_id, frames_done=done, frames_total=total)
            self.events.put(ProgressEvent("frame", {"done": done, "total": total}))

        def on_incident(incident: Incident) -> None:
            self.events.put(ProgressEvent("incident", incident.to_dict()))

        result = pipeline.run(self.video_path, on_frame=on_frame, on_incident=on_incident)
        video_path = result.out_dir / "annotated.mp4"
        if video_path.is_file():
            make_browser_playable(video_path)
        db.update_run(
            conn,
            self.job_id,
            status="done",
            run_id=result.run_id,
            incident_count=len(result.incidents),
        )
        self.events.put(ProgressEvent("done", {"run_id": result.run_id}))


class JobRegistry:
    """Holds in-memory Job objects for the lifetime of the process (SSE needs the live queue),
    and the shared slots that limit how many run at once."""

    def __init__(self, max_concurrent: int = 1, cpu_threads: int | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max_concurrent)
        self._cpu_threads = cpu_threads

    def create(self, job_id: str, video_path: Path) -> Job:
        job = Job(job_id, video_path, slots=self._slots, cpu_threads=self._cpu_threads)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


def new_job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def event_to_sse(event: ProgressEvent) -> str:
    return f"event: {event.kind}\ndata: {json.dumps(event.data)}\n\n"
