from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import cv2

from trafficwatch.types import Frame


class IngestError(ValueError):
    """The video cannot be used: missing, unreadable, no frames, or no valid frame rate."""


class VideoIngestor:
    """Reads a video file and yields frames with timestamps derived from the file's fps."""

    def __init__(self, video_path: str | Path, frame_stride: int = 1) -> None:
        if frame_stride < 1:
            raise IngestError(f"frame_stride must be >= 1, got {frame_stride}")
        self.video_path = Path(video_path)
        self.frame_stride = frame_stride
        capture = self._open()
        try:
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            readable, _ = capture.read()
        finally:
            capture.release()
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise IngestError(f"{self.video_path}: no valid frame rate ({self.fps})")
        if not readable:
            raise IngestError(f"{self.video_path}: no readable frames")

    def _open(self) -> cv2.VideoCapture:
        if not self.video_path.is_file():
            raise IngestError(f"video not found: {self.video_path}")
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise IngestError(f"cannot open video: {self.video_path}")
        return capture

    def frames(self) -> Iterator[Frame]:
        capture = self._open()
        try:
            index = 0
            while True:
                readable, image = capture.read()
                if not readable:
                    return
                if index % self.frame_stride == 0:
                    yield Frame(index, index / self.fps, image)
                index += 1
        finally:
            capture.release()
