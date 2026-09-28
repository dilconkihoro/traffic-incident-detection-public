"""make_browser_playable / ensure_browser_copy: shared by the job runner (annotated output), the
upload route (original upload) and the results routes (serve-time fallback for older files)."""

from __future__ import annotations

import subprocess

import cv2
import numpy as np
import pytest

from webapp.backend.video import ensure_browser_copy, is_browser_ready, make_browser_playable


def write_clip(path, fourcc: str, frames: int = 5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 10.0, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def codec_of(path) -> str:
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
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return probe.stdout.strip()


def test_make_browser_playable_reports_clearly_when_ffmpeg_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("webapp.backend.video.shutil.which", lambda _name: None)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a real video, ffmpeg should never be reached")
    with pytest.raises(RuntimeError, match="ffmpeg not found on PATH"):
        make_browser_playable(video)


def test_make_browser_playable_with_out_path_normalises_container_and_removes_original(tmp_path):
    src = write_clip(tmp_path / "clip.avi", "MJPG")

    out_path = tmp_path / "clip.mp4"
    result = make_browser_playable(src, out_path=out_path)

    assert result == out_path
    assert out_path.is_file()
    assert not src.exists()


def test_make_browser_playable_reencodes_a_real_mpeg4_file_to_h264(tmp_path):
    src = write_clip(tmp_path / "clip.mp4", "mp4v")

    make_browser_playable(src)

    assert codec_of(src) == "h264"


def test_is_browser_ready_distinguishes_h264_from_mpeg4(tmp_path):
    mpeg4 = write_clip(tmp_path / "old.mp4", "mp4v")
    assert not is_browser_ready(mpeg4)

    make_browser_playable(mpeg4)
    assert is_browser_ready(mpeg4)


def test_is_browser_ready_is_false_for_a_file_that_is_not_a_video(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert not is_browser_ready(junk)


def test_ensure_browser_copy_converts_an_old_file_beside_it_and_keeps_the_original(tmp_path):
    old = write_clip(tmp_path / "old.mp4", "mp4v")
    original_bytes = old.read_bytes()

    served = ensure_browser_copy(old)

    assert served == tmp_path / "old.web.mp4"
    assert codec_of(served) == "h264"
    assert old.read_bytes() == original_bytes


def test_ensure_browser_copy_reuses_the_converted_copy_instead_of_converting_again(tmp_path):
    old = write_clip(tmp_path / "old.mp4", "mp4v")

    first = ensure_browser_copy(old)
    first_mtime = first.stat().st_mtime_ns
    second = ensure_browser_copy(old)

    assert second == first
    assert second.stat().st_mtime_ns == first_mtime


def test_ensure_browser_copy_returns_a_ready_file_unchanged(tmp_path):
    ready = write_clip(tmp_path / "ready.mp4", "mp4v")
    make_browser_playable(ready)

    assert ensure_browser_copy(ready) == ready
    assert not (tmp_path / "ready.web.mp4").exists()
