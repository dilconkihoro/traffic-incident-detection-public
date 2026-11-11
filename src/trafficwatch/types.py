"""Typed records that cross module boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

XYXY = tuple[float, float, float, float]


class IncidentType(str, Enum):
    ABRUPT_STOP = "ABRUPT_STOP"
    LANE_OBSTRUCTION = "LANE_OBSTRUCTION"
    COLLISION_LIKE = "COLLISION_LIKE"


@dataclass(frozen=True, eq=False)
class Frame:
    """One decoded video frame; ``image`` is a uint8 BGR array."""

    idx: int
    t_s: float
    image: np.ndarray


@dataclass(frozen=True, eq=False)
class Prepared:
    """Letterboxed 640x640 frame plus the parameters needed to invert box coordinates."""

    image640: np.ndarray
    scale: float
    pad: tuple[float, float]
    original_wh: tuple[int, int]

    def to_original(self, xyxy: XYXY) -> XYXY:
        """Map a box from letterboxed coordinates back to the original frame, clipped to it."""
        (left, top), (w, h) = self.pad, self.original_wh
        x1, y1, x2, y2 = xyxy
        return (
            min(max((x1 - left) / self.scale, 0.0), float(w)),
            min(max((y1 - top) / self.scale, 0.0), float(h)),
            min(max((x2 - left) / self.scale, 0.0), float(w)),
            min(max((y2 - top) / self.scale, 0.0), float(h)),
        )


@dataclass(frozen=True)
class Detection:
    xyxy: XYXY
    conf: float
    cls_id: int
    cls_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "xyxy": list(self.xyxy),
            "conf": self.conf,
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
        }


@dataclass(frozen=True)
class Track:
    track_id: int
    xyxy: XYXY
    cls_name: str
    age: int
    hits: int
    confirmed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "xyxy": list(self.xyxy),
            "cls_name": self.cls_name,
            "age": self.age,
            "hits": self.hits,
            "confirmed": self.confirmed,
        }


@dataclass(frozen=True)
class TrackPoint:
    """One trajectory sample for a confirmed track."""

    frame_idx: int
    t_s: float
    xyxy: XYXY

    def to_dict(self) -> dict[str, Any]:
        return {"frame_idx": self.frame_idx, "t_s": self.t_s, "xyxy": list(self.xyxy)}


@dataclass(frozen=True)
class SceneContext:
    """Trajectory history of every active track up to and including ``t_s``."""

    frame_idx: int
    t_s: float
    trajectories: Mapping[int, Sequence[TrackPoint]]


@dataclass(frozen=True)
class Incident:
    """``vehicle_ids`` pairs with ``track_ids`` position by position; the behaviour rules leave it
    empty and the pipeline fills it from the vehicle registry before the incident is logged."""

    type: IncidentType
    track_ids: tuple[int, ...]
    frame_onset: int
    frame_flag: int
    t_onset_s: float
    t_flag_s: float
    bbox_union: XYXY
    score: float
    vehicle_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "track_ids": list(self.track_ids),
            "vehicle_ids": list(self.vehicle_ids),
            "frame_onset": self.frame_onset,
            "frame_flag": self.frame_flag,
            "t_onset_s": self.t_onset_s,
            "t_flag_s": self.t_flag_s,
            "bbox_union": list(self.bbox_union),
            "score": self.score,
        }


@dataclass(frozen=True)
class VehicleProfile:
    """Anonymous identity and visible description of one tracked vehicle (decision D-28)."""

    vehicle_id: str
    track_id: int
    cls_name: str
    colour: str
    direction: str
    first_seen_s: float
    last_seen_s: float
    frames_seen: int

    @property
    def description(self) -> str:
        return (
            f"{self.colour} {self.cls_name}, {self.direction}, "
            f"seen {self.first_seen_s:.1f} s to {self.last_seen_s:.1f} s"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "track_id": self.track_id,
            "cls_name": self.cls_name,
            "colour": self.colour,
            "direction": self.direction,
            "first_seen_s": self.first_seen_s,
            "last_seen_s": self.last_seen_s,
            "frames_seen": self.frames_seen,
            "description": self.description,
        }
