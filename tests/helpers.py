"""Synthetic-trajectory builders for the white-box rule tests.

Every scenario is sampled at 10 Hz with a 100 px body height, so 1 body-height per second is
10 px per frame and the speed estimate v_k = |g_k - g_{k-5}| / (0.5 s * 100 px) is the sum of the
last five per-frame steps divided by 50. All step sizes used in tests are exact in binary
floating point unless a comment says otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from trafficwatch.behaviour.rules import Rule
from trafficwatch.config import (
    AbruptStopConfig,
    BehaviourConfig,
    CollisionConfig,
    LaneObstructionConfig,
    Polygon,
    SuppressionConfig,
    ZonesConfig,
)
from trafficwatch.types import Incident, SceneContext, TrackPoint

DT = 0.1
BODY_HEIGHT = 100.0
BOX_WIDTH = 110.0
N_INIT = 3

Step = tuple[float, float]


def behaviour_cfg(
    *,
    driving_lanes: tuple[Polygon, ...] = (),
    exclusion: tuple[Polygon, ...] = (),
) -> BehaviourConfig:
    """Test-owned copy of the docs/02 defaults, independent of configs/pipeline.yaml."""
    return BehaviourConfig(
        speed_unit="body_heights_per_s",
        v_stop=0.25,
        v_move=1.0,
        smoothing_window_s=0.5,
        abrupt_stop=AbruptStopConfig(
            window_before_s=1.0, window_after_s=0.5, drop_ratio=0.8, max_decel_s=1.0
        ),
        lane_obstruction=LaneObstructionConfig(
            min_stationary_s=15.0, min_flow_fraction=0.3, rearm_s=5.0
        ),
        collision=CollisionConfig(
            v_close=1.5, iou_min=0.10, min_frames=3, post_window_s=1.5, heading_change_deg=45.0
        ),
        suppression=SuppressionConfig(queue_gap_bh=1.5, global_slowdown_fraction=0.5),
        zones=ZonesConfig(driving_lanes=driving_lanes, exclusion=exclusion),
    )


def hold(n: int) -> list[Step]:
    return [(0.0, 0.0)] * n


def run(n: int, dx: float, dy: float = 0.0) -> list[Step]:
    return [(dx, dy)] * n


def track(steps: Sequence[Step], *, x0: float, y0: float = 500.0) -> list[TrackPoint]:
    """Sample 0 has its ground point (bottom-centre) at (x0, y0); each step moves it by (dx, dy)."""
    x, y = x0, y0
    points = [_point(0, x, y)]
    for i, (dx, dy) in enumerate(steps, start=1):
        x, y = x + dx, y + dy
        points.append(_point(i, x, y))
    return points


def _point(frame: int, gx: float, gy: float) -> TrackPoint:
    half = BOX_WIDTH / 2
    return TrackPoint(frame, round(frame * DT, 9), (gx - half, gy - BODY_HEIGHT, gx + half, gy))


def heading_step(speed_px: float, degrees: float) -> Step:
    rad = math.radians(degrees)
    return (speed_px * math.cos(rad), speed_px * math.sin(rad))


def replay(rule: Rule, trajectories: Mapping[int, Sequence[TrackPoint]]) -> list[Incident]:
    """Feed the rule frame by frame with history truncated to the current frame (no look-ahead).

    A track is active at frame f only while it has a sample at f; all tracks start at frame 0.
    """
    last_frame = max(len(points) for points in trajectories.values()) - 1
    incidents: list[Incident] = []
    for f in range(last_frame + 1):
        visible = {tid: points[: f + 1] for tid, points in trajectories.items() if len(points) > f}
        incidents.extend(rule.evaluate(SceneContext(f, round(f * DT, 9), visible)))
    return incidents


def involving(incidents: Sequence[Incident], track_id: int) -> list[Incident]:
    return [i for i in incidents if track_id in i.track_ids]


def square(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


__all__ = [
    "BODY_HEIGHT",
    "BOX_WIDTH",
    "DT",
    "N_INIT",
    "behaviour_cfg",
    "heading_step",
    "hold",
    "involving",
    "replay",
    "run",
    "square",
    "track",
]
