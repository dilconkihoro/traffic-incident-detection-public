from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from trafficwatch.ingestion import IngestError, VideoIngestor
from webapp.backend.settings import Settings
from webapp.backend.video import make_browser_playable

router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB; a generous cap, not a promise it will be fast


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.post("/uploads")
async def upload_video(
    file: UploadFile, settings: Settings = Depends(get_settings)
) -> dict[str, str]:
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type {suffix!r}; allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    dest = settings.uploads_dir / f"{upload_id}{suffix}"

    written = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1 << 30)} GiB limit",
                )
            out.write(chunk)

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty")

    try:
        probe = VideoIngestor(dest)
    except IngestError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=f"unreadable video file: {exc}"
        ) from exc

    if probe.frame_count > settings.max_upload_frames:
        dest.unlink(missing_ok=True)
        duration_s = probe.frame_count / probe.fps
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                f"video has {probe.frame_count} frames (about {duration_s:.0f}s at "
                f"{probe.fps:.0f} fps); the limit for this demo is {settings.max_upload_frames} "
                "frames, to keep processing fast on a CPU-only machine. Use a shorter clip."
            ),
        )

    # Normalise every upload to a browser-playable .mp4 up front: an uploaded file's own codec
    # or container is not guaranteed to be one a browser can play back later on the results page
    # (this project's own bundled sample clips are a real example of this).
    try:
        dest = make_browser_playable(dest, out_path=dest.with_suffix(".mp4"))
    except RuntimeError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"upload_id": upload_id, "path": str(dest), "filename": file.filename}
