"""Typed records that cross module boundaries (specification 3.2.2)."""

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
    type: IncidentType
    track_ids: tuple[int, ...]
    frame_onset: int
    frame_flag: int
    t_onset_s: float
    t_flag_s: float
    bbox_union: XYXY
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "track_ids": list(self.track_ids),
            "frame_onset": self.frame_onset,
            "frame_flag": self.frame_flag,
            "t_onset_s": self.t_onset_s,
            "t_flag_s": self.t_flag_s,
            "bbox_union": list(self.bbox_union),
            "score": self.score,
        }
