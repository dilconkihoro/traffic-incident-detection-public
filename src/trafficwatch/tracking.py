from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from trafficwatch.config import TrackerConfig
from trafficwatch.types import Detection, Frame, Track, TrackPoint


class TrajectoryStore:
    """Bounded per-track history of confirmed-track samples."""

    def __init__(self, max_samples: int) -> None:
        if max_samples < 2:
            raise ValueError(f"max_samples must be at least 2, got {max_samples}")
        self._max_samples = max_samples
        self._history: dict[int, deque[TrackPoint]] = {}

    def add(self, track: Track, frame_idx: int, t_s: float) -> None:
        points = self._history.setdefault(track.track_id, deque(maxlen=self._max_samples))
        points.append(TrackPoint(frame_idx, t_s, track.xyxy))

    def history(self, track_id: int) -> Sequence[TrackPoint]:
        return tuple(self._history.get(track_id, ()))

    def snapshot(self, track_ids: Iterable[int]) -> dict[int, Sequence[TrackPoint]]:
        return {tid: tuple(self._history[tid]) for tid in track_ids if tid in self._history}

    def drop_stale(self, min_frame_idx: int) -> None:
        """Forget tracks whose latest sample is older than ``min_frame_idx``."""
        for tid in [t for t, pts in self._history.items() if pts[-1].frame_idx < min_frame_idx]:
            del self._history[tid]


class VehicleTracker:
    """DeepSORT wrapper (deep-sort-realtime); emits confirmed tracks matched in this frame."""

    def __init__(self, cfg: TrackerConfig) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.cfg = cfg
        self._tracker = DeepSort(
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            max_iou_distance=cfg.max_iou_distance,
            embedder=cfg.embedder,
        )

    def update(self, detections: Sequence[Detection], frame: Frame) -> list[Track]:
        raw = [
            (
                [d.xyxy[0], d.xyxy[1], d.xyxy[2] - d.xyxy[0], d.xyxy[3] - d.xyxy[1]],
                d.conf,
                d.cls_name,
            )
            for d in detections
        ]
        tracks: list[Track] = []
        for tr in self._tracker.update_tracks(raw, frame=frame.image):
            if not tr.is_confirmed() or tr.time_since_update > 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in tr.to_ltrb())
            tracks.append(
                Track(
                    track_id=int(tr.track_id),
                    xyxy=(x1, y1, x2, y2),
                    cls_name=tr.get_det_class() or "unknown",
                    age=int(tr.age),
                    hits=int(tr.hits),
                    confirmed=True,
                )
            )
        return tracks
