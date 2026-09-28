"""Making a video browser-playable. Shared by the job runner (the annotated output), the upload
route (the original upload, which may just as easily be an old MPEG-4 Part 2 file, like this
repo's own data/samples/MVI_20011_short.mp4) and the results routes (a serve-time fallback for
files created before uploads and runs were normalised)."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

# Two requests for the same not-yet-converted file (a <video> element often opens two
# connections) must not both run ffmpeg into the same temporary file.
_convert_lock = threading.Lock()


def _reencode(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; it is required to make videos playable in a browser "
            "(see docs/08_SETUP_AND_PORTABILITY.md section 2 to install it)"
        )
    browser_copy = dst.with_name(f"{dst.stem}.browser{dst.suffix}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(browser_copy),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        browser_copy.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to re-encode the video: {result.stderr.strip()}")
    browser_copy.replace(dst)


def make_browser_playable(video_path: Path, *, out_path: Path | None = None) -> Path:
    """cv2.VideoWriter writes mp4v (MPEG-4 Part 2), which a desktop player like VLC plays fine
    but which no mainstream browser can decode ("No video with supported format"); many ordinary
    video files (an .avi, an old phone recording) use codecs or containers browsers are equally
    picky about. The pip opencv-python wheel also typically cannot encode H.264 directly (no
    bundled libx264), so re-encoding through the ffmpeg binary is the portable fix.

    Re-encodes in place by default. Pass ``out_path`` (e.g. to normalise a non-.mp4 upload to a
    .mp4 container) to write there instead; the original file at ``video_path`` is then removed
    once the new one exists. Returns the final path.
    """
    target = out_path if out_path is not None else video_path
    _reencode(video_path, target)
    if target != video_path:
        video_path.unlink(missing_ok=True)
    return target


def is_browser_ready(video_path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False
    fields = dict(line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line)
    return fields.get("codec_name") == "h264" and fields.get("pix_fmt") == "yuv420p"


def ensure_browser_copy(video_path: Path) -> Path:
    """Serve-time fallback for files made before uploads and runs were normalised (an older run
    in an existing local database). Returns the file itself when a browser can already play it,
    otherwise a converted ``<name>.web.mp4`` next to it, made once and then reused. The original
    is left untouched: it may be the user's only copy, and a desktop player still plays it."""
    with _convert_lock:
        if is_browser_ready(video_path):
            return video_path
        web_copy = video_path.with_name(f"{video_path.stem}.web.mp4")
        if web_copy.is_file() and web_copy.stat().st_mtime >= video_path.stat().st_mtime:
            return web_copy
        _reencode(video_path, web_copy)
        return web_copy
