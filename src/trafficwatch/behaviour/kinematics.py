"""Pure geometry and kinematics helpers for the behaviour rules (docs/02 section 5.5)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median

from trafficwatch.config import Polygon
from trafficwatch.types import XYXY, TrackPoint

TOL = 1e-9
MIN_DISPLACEMENT_PX = 1e-6

Point = tuple[float, float]


def ground_point(xyxy: XYXY) -> Point:
    """Bottom-centre of the box."""
    return ((xyxy[0] + xyxy[2]) / 2.0, xyxy[3])


def median_height(points: Sequence[TrackPoint]) -> float:
    return max(median(p.xyxy[3] - p.xyxy[1] for p in points), TOL)


def sample_period(points: Sequence[TrackPoint]) -> float:
    """Seconds per sample, from the first and last timestamps of a track history."""
    return (points[-1].t_s - points[0].t_s) / (len(points) - 1)


def samples_in(seconds: float, period_s: float) -> int:
    return max(1, round(seconds / period_s))


def speed_series(points: Sequence[TrackPoint], window_s: float) -> list[float | None]:
    """Body heights per second: ground-point travel over ``window_s``, over median box height.

    Entry k is None until a full window of history exists.
    """
    window = samples_in(window_s, sample_period(points))
    height = median_height(points)
    ground = [ground_point(p.xyxy) for p in points]
    series: list[float | None] = [None] * len(points)
    for k in range(window, len(points)):
        series[k] = math.dist(ground[k], ground[k - window]) / (window_s * height)
    return series


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def in_any(point: Point, polygons: Sequence[Polygon]) -> bool:
    return any(point_in_polygon(point, polygon) for polygon in polygons)


def box_iou(a: XYXY, b: XYXY) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: Sequence[XYXY]) -> XYXY:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def heading_deg(start: Point, end: Point) -> float | None:
    """Direction of travel in degrees, or None when the displacement is negligible."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < MIN_DISPLACEMENT_PX:
        return None
    return math.degrees(math.atan2(dy, dx))


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def heading_change_exceeds(pre: float, post: float, threshold_deg: float) -> bool:
    return angle_diff_deg(pre, post) >= threshold_deg - TOL


def post_contact_window(contact: int, n_post: int, n_samples: int) -> range:
    """Sample indices after contact that count as evidence: contact + 1 through contact + n_post."""
    return range(contact + 1, min(n_samples, contact + n_post + 1))


def motion_rearms(motion_s: float, rearm_s: float) -> bool:
    return motion_s >= rearm_s - TOL


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
