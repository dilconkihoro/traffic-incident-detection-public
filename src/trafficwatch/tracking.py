from __future__ import annotations

from collections.abc import Sequence

from trafficwatch.config import TrackerConfig
from trafficwatch.types import Detection, Frame, Track, TrackPoint


class TrajectoryStore:
    """Per-track history of confirmed-track samples."""

    def add(self, track: Track, frame_idx: int, t_s: float) -> None:
        raise NotImplementedError("Phase 2")

    def history(self, track_id: int) -> Sequence[TrackPoint]:
        raise NotImplementedError("Phase 2")


class VehicleTracker:
    """DeepSORT wrapper (deep-sort-realtime); emits confirmed tracks only."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg

    def update(self, detections: Sequence[Detection], frame: Frame) -> list[Track]:
        raise NotImplementedError("Phase 2")
