from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from trafficwatch.types import Frame


class VideoIngestor:
    """Reads a video file and yields frames with timestamps derived from the file's fps."""

    def __init__(self, video_path: str | Path, frame_stride: int = 1) -> None:
        self.video_path = Path(video_path)
        self.frame_stride = frame_stride

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError("Phase 2")
