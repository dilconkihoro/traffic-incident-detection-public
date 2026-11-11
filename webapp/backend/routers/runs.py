from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from webapp.backend import db
from webapp.backend.auth import CurrentUser, require_roles
from webapp.backend.jobs import event_to_sse, new_job_id
from webapp.backend.routers.models import trained_model_status
from webapp.backend.video import ensure_browser_copy

router = APIRouter(prefix="/api/runs")


class CreateRunRequest(BaseModel):
    upload_id: str
    path: str
    filename: str
    standin: bool = True


def _row_to_dict(row) -> dict[str, object]:
    return dict(row)


@router.get("")
def list_runs(request: Request) -> list[dict[str, object]]:
    conn = request.app.state.db
    return [_row_to_dict(r) for r in db.list_runs(conn)]


@router.post("")
def create_run(
    payload: CreateRunRequest,
    request: Request,
    user: CurrentUser = Depends(require_roles("admin", "operator")),
) -> dict[str, str]:
    settings = request.app.state.settings
    # Only a file this dashboard stored as an upload may be run, never an arbitrary local path.
    video_path = Path(payload.path).resolve()
    if video_path.parent != settings.uploads_dir.resolve() or not video_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="uploaded file not found")

    conn = request.app.state.db
    if not payload.standin:
        trained = trained_model_status(settings.pipeline_config_path)
        if not trained["available"]:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail=f"the trained model is not usable: {trained['problem']}",
            )
    job_id = new_job_id()
    db.create_run(
        conn,
        job_id,
        payload.filename,
        datetime.now(timezone.utc).isoformat(),
        str(video_path),
        created_by=user.username,
    )

    job = request.app.state.jobs.create(job_id, video_path)
    job.start(conn, settings.pipeline_config_path, settings.reports_dir, standin=payload.standin)
    return {"job_id": job_id}


@router.get("/{job_id}")
def get_run(job_id: str, request: Request) -> dict[str, object]:
    row = db.get_run(request.app.state.db, job_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found")
    return _row_to_dict(row)


@router.post("/{job_id}/cancel")
def cancel_run(
    job_id: str,
    request: Request,
    user: CurrentUser = Depends(require_roles("admin", "operator")),
) -> dict[str, str]:
    conn = request.app.state.db
    row = db.get_run(conn, job_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found")
    if user.role == "operator" and row["created_by"] != user.username:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail="operators can cancel only their own runs"
        )
    if row["status"] not in ("queued", "running"):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="this run has already finished")
    job = request.app.state.jobs.get(job_id)
    if job is None:
        # Left over from a server that has since stopped: nothing is executing it any more.
        db.update_run(conn, job_id, status="failed", error=db.CANCELLED_MESSAGE)
        return {"status": "cancelled"}
    job.cancel()
    return {"status": "cancelling"}


@router.get("/{job_id}/events")
async def run_events(job_id: str, request: Request) -> StreamingResponse:
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found or already finished")

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, job.events.get)
            yield event_to_sse(event)
            if event.kind in ("done", "failed"):
                return

    return StreamingResponse(stream(), media_type="text/event-stream")


def _report_dir(request: Request, run_id: str) -> Path:
    settings = request.app.state.settings
    reports_dir = (settings.reports_dir / run_id).resolve()
    if reports_dir.parent != settings.reports_dir.resolve() or not reports_dir.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="report not found")
    return reports_dir


@router.get("/{job_id}/video")
def get_video(job_id: str, request: Request) -> FileResponse:
    row = db.get_run(request.app.state.db, job_id)
    if row is None or not row["run_id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found or not finished")
    video_path = _report_dir(request, row["run_id"]) / "annotated.mp4"
    if not video_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="annotated video not found")
    return FileResponse(ensure_browser_copy(video_path), media_type="video/mp4")


@router.get("/{job_id}/incidents.csv")
def get_incidents_csv(job_id: str, request: Request) -> FileResponse:
    row = db.get_run(request.app.state.db, job_id)
    if row is None or not row["run_id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found or not finished")
    csv_path = _report_dir(request, row["run_id"]) / "incidents.csv"
    if not csv_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="incidents.csv not found")
    return FileResponse(csv_path, media_type="text/csv", filename="incidents.csv")


@router.get("/{job_id}/tracks.csv")
def get_tracks_csv(job_id: str, request: Request) -> FileResponse:
    row = db.get_run(request.app.state.db, job_id)
    if row is None or not row["run_id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found or not finished")
    csv_path = _report_dir(request, row["run_id"]) / "tracks.csv"
    if not csv_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="tracks.csv not found")
    return FileResponse(csv_path, media_type="text/csv", filename="tracks.csv")


@router.get("/{job_id}/vehicles.csv")
def get_vehicles_csv(job_id: str, request: Request) -> FileResponse:
    row = db.get_run(request.app.state.db, job_id)
    if row is None or not row["run_id"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="run not found or not finished")
    csv_path = _report_dir(request, row["run_id"]) / "vehicles.csv"
    if not csv_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="vehicles.csv not found")
    return FileResponse(csv_path, media_type="text/csv", filename="vehicles.csv")


def _recorded_upload_path(settings, row) -> Path | None:
    """The upload path stored on the run, or, for a run made before it was stored, the one the
    pipeline's own run manifest recorded for that run."""
    if row["upload_path"]:
        return Path(row["upload_path"])
    if not row["run_id"]:
        return None
    manifest = settings.reports_dir / row["run_id"] / "run_manifest.json"
    try:
        return Path(json.loads(manifest.read_text(encoding="utf-8"))["video"]["path"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


@router.get("/{job_id}/original")
def get_original_video(job_id: str, request: Request) -> FileResponse:
    settings = request.app.state.settings
    row = db.get_run(request.app.state.db, job_id)
    recorded = _recorded_upload_path(settings, row) if row is not None else None
    if recorded is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="original upload not found")
    upload_path = recorded.resolve()
    if upload_path.parent != settings.uploads_dir.resolve() or not upload_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="original upload not found")
    return FileResponse(ensure_browser_copy(upload_path), media_type="video/mp4")
