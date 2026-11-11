"""Anonymous vehicle identity: a stable identifier and a visible description per tracked vehicle.

Number plates are deliberately not read (decision D-28): the benchmark datasets this project uses
are too low in resolution for legible plates and carry no plate labels, plates are personal data,
and plate reading is outside the approved scope. The identifier is a pseudonym derived from the
video's content and the tracker's id, so the same video gives the same identifiers on every run,
and it carries no information about the vehicle's owner.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from trafficwatch.config import IdentityConfig
from trafficwatch.types import Frame, Track, VehicleProfile

# Unambiguous when read aloud or off a screen: no 0/O, 1/I/L, or U.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
UNKNOWN = "unknown"


def _code(digest: bytes, length: int) -> str:
    value = int.from_bytes(digest, "big")
    chars = []
    for _ in range(length):
        value, index = divmod(value, len(CODE_ALPHABET))
        chars.append(CODE_ALPHABET[index])
    return "".join(chars)


def _most_common(counts: Counter[str]) -> str:
    """Highest count wins; ties go to the alphabetically first name, so output is deterministic."""
    if not counts:
        return UNKNOWN
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def colour_name(
    image: np.ndarray, xyxy: tuple[float, float, float, float], cfg: IdentityConfig
) -> str | None:
    """Majority colour name of the central part of a box, or None when the crop is empty."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    margin_x = (x2 - x1) * (1.0 - cfg.colour_crop_fraction) / 2.0
    margin_y = (y2 - y1) * (1.0 - cfg.colour_crop_fraction) / 2.0
    left, right = int(max(0.0, x1 + margin_x)), int(min(float(width), x2 - margin_x))
    top, bottom = int(max(0.0, y1 + margin_y)), int(min(float(height), y2 - margin_y))
    if right <= left or bottom <= top:
        return None
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)

    achromatic = sat <= cfg.achromatic_max_saturation
    votes: Counter[str] = Counter()
    votes["black"] = int(np.count_nonzero(achromatic & (val <= cfg.black_max_value)))
    votes["white"] = int(np.count_nonzero(achromatic & (val >= cfg.white_min_value)))
    votes["grey"] = int(
        np.count_nonzero(achromatic & (val > cfg.black_max_value) & (val < cfg.white_min_value))
    )
    chromatic = ~achromatic
    for name, lo, hi in cfg.hue_bins:
        votes[name] += int(np.count_nonzero(chromatic & (hue >= lo) & (hue < hi)))
    votes = Counter({name: n for name, n in votes.items() if n > 0})
    return _most_common(votes) if votes else None


@dataclass
class _Observations:
    vehicle_id: str
    track_id: int
    first_seen_s: float
    last_seen_s: float
    first_centre: tuple[float, float]
    last_centre: tuple[float, float]
    frames_seen: int = 0
    classes: Counter[str] = field(default_factory=Counter)
    colours: Counter[str] = field(default_factory=Counter)
    heights: list[float] = field(default_factory=list)


class VehicleRegistry:
    """Assigns each confirmed track an anonymous vehicle id and accumulates its description."""

    def __init__(self, cfg: IdentityConfig, video_sha256: str) -> None:
        self.cfg = cfg
        self._video_sha256 = video_sha256
        self._vehicles: dict[int, _Observations] = {}
        self._codes: set[str] = set()

    def _new_id(self, track_id: int) -> str:
        digest = hashlib.sha256(f"{self._video_sha256}:{track_id}".encode()).digest()
        length = self.cfg.code_length
        code = _code(digest, length)
        while code in self._codes:
            length += 1
            code = _code(digest, length)
        self._codes.add(code)
        return f"{self.cfg.id_prefix}-{code}"

    def observe(self, track: Track, frame: Frame) -> str:
        """Records one sighting and returns the vehicle id, which never changes for a track."""
        x1, y1, x2, y2 = track.xyxy
        centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        seen = self._vehicles.get(track.track_id)
        if seen is None:
            seen = _Observations(
                vehicle_id=self._new_id(track.track_id),
                track_id=track.track_id,
                first_seen_s=frame.t_s,
                last_seen_s=frame.t_s,
                first_centre=centre,
                last_centre=centre,
            )
            self._vehicles[track.track_id] = seen
        if seen.frames_seen % self.cfg.colour_sample_every == 0:
            colour = colour_name(frame.image, track.xyxy, self.cfg)
            if colour is not None:
                seen.colours[colour] += 1
        seen.frames_seen += 1
        seen.last_seen_s = frame.t_s
        seen.last_centre = centre
        seen.classes[track.cls_name] += 1
        seen.heights.append(y2 - y1)
        return seen.vehicle_id

    def vehicle_id(self, track_id: int) -> str:
        return self._vehicles[track_id].vehicle_id

    def _direction(self, seen: _Observations) -> str:
        dx = seen.last_centre[0] - seen.first_centre[0]
        dy = seen.last_centre[1] - seen.first_centre[1]
        body_height = float(np.median(seen.heights)) if seen.heights else 0.0
        if body_height <= 0.0 or np.hypot(dx, dy) < self.cfg.min_travel_bh * body_height:
            return "stationary"
        if abs(dx) >= abs(dy):
            return "moving right" if dx > 0 else "moving left"
        return "moving down the frame" if dy > 0 else "moving up the frame"

    def profiles(self) -> list[VehicleProfile]:
        """One profile per vehicle seen so far, in order of first appearance."""
        return [
            VehicleProfile(
                vehicle_id=seen.vehicle_id,
                track_id=seen.track_id,
                cls_name=_most_common(seen.classes),
                colour=_most_common(seen.colours),
                direction=self._direction(seen),
                first_seen_s=seen.first_seen_s,
                last_seen_s=seen.last_seen_s,
                frames_seen=seen.frames_seen,
            )
            for seen in self._vehicles.values()
        ]
