"""Rule-based incident checks (docs/02 section 5); thresholds come only from BehaviourConfig.

Estimator definitions are pinned in docs/02 section 5.5 (decision D-19) and by the white-box
tests in tests/test_rules_*.py.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from trafficwatch.behaviour.kinematics import (
    TOL,
    Point,
    box_iou,
    clip01,
    ground_point,
    heading_change_exceeds,
    heading_deg,
    in_any,
    median_height,
    motion_rearms,
    post_contact_window,
    sample_period,
    samples_in,
    speed_series,
    union_box,
)
from trafficwatch.config import BehaviourConfig
from trafficwatch.types import Incident, IncidentType, SceneContext, TrackPoint

Trajectories = dict[int, Sequence[TrackPoint]]
Speeds = list[float | None]


class Rule(ABC):
    """A rule sees the trajectory history up to the current frame and returns new incidents."""

    def __init__(self, cfg: BehaviourConfig, n_init: int) -> None:
        self.cfg = cfg
        self.n_init = n_init

    @abstractmethod
    def evaluate(self, ctx: SceneContext) -> list[Incident]: ...

    def _usable(self, ctx: SceneContext) -> Trajectories:
        """Tracks old enough for any rule to use (a speed needs at least two samples)."""
        floor = max(self.n_init, 2)
        return {tid: pts for tid, pts in ctx.trajectories.items() if len(pts) >= floor}

    def _speeds(self, usable: Trajectories) -> dict[int, Speeds]:
        return {tid: speed_series(pts, self.cfg.smoothing_window_s) for tid, pts in usable.items()}


def _defined(values: Sequence[float | None]) -> list[float]:
    return [v for v in values if v is not None]


class AbruptStopRule(Rule):
    """Flags a vehicle that goes from moving to stationary within max_decel_s (docs/02 5.1)."""

    def evaluate(self, ctx: SceneContext) -> list[Incident]:
        usable = self._usable(ctx)
        speeds = self._speeds(usable)
        floor = self.cfg.abrupt_stop.drop_ratio
        incidents: list[Incident] = []
        for tid, pts in usable.items():
            hit = self._detect(pts, speeds[tid])
            if hit is None:
                continue
            last_moving, ratio = hit
            if self._suppressed(tid, last_moving, usable, speeds):
                continue
            now = pts[-1]
            score = clip01((ratio - floor) / (1 - floor)) if floor < 1 else 1.0
            onset = pts[last_moving]
            incidents.append(
                Incident(
                    IncidentType.ABRUPT_STOP,
                    (tid,),
                    onset.frame_idx,
                    now.frame_idx,
                    onset.t_s,
                    now.t_s,
                    now.xyxy,
                    score,
                )
            )
        return incidents

    def _windows(self, pts: Sequence[TrackPoint]) -> tuple[int, int, int, int]:
        period, ab = sample_period(pts), self.cfg.abrupt_stop
        return (
            samples_in(self.cfg.smoothing_window_s, period),
            samples_in(ab.window_before_s, period),
            samples_in(ab.window_after_s, period),
            samples_in(ab.max_decel_s, period),
        )

    def _detect(self, pts: Sequence[TrackPoint], v: Speeds) -> tuple[int, float] | None:
        """Return (last moving sample, drop ratio) when the current frame closes a stop pattern."""
        cfg = self.cfg
        smooth, n_before, n_after, n_decel = self._windows(pts)
        k = len(pts) - 1
        b = k - n_after + 1
        if b < smooth or v[b] is None or v[b] > cfg.v_stop + TOL:
            return None
        last_moving = None
        for j in range(b - 1, max(smooth, b - n_decel) - 1, -1):
            if v[j] >= cfg.v_move - TOL:
                last_moving = j
                break
            if v[j] <= cfg.v_stop + TOL:
                return None
        if last_moving is None:
            return None
        post = median(_defined(v[b : k + 1]))
        pre = median(_defined(v[max(smooth, last_moving - n_before + 1) : last_moving + 1]))
        if post > cfg.v_stop + TOL or pre < cfg.v_move - TOL:
            return None
        ratio = (pre - post) / pre
        if ratio < cfg.abrupt_stop.drop_ratio - TOL:
            return None
        return last_moving, ratio

    def _suppressed(
        self, tid: int, last_moving: int, usable: Trajectories, speeds: dict[int, Speeds]
    ) -> bool:
        cfg = self.cfg
        pts = usable[tid]
        smooth = self._windows(pts)[0]
        here = ground_point(pts[-1].xyxy)
        if in_any(here, cfg.zones.exclusion):
            return True

        before = ground_point(pts[last_moving - smooth].xyxy)
        after = ground_point(pts[last_moving].xyxy)
        heading = (after[0] - before[0], after[1] - before[1])
        height = median_height(pts)
        if math.hypot(*heading) > TOL:
            for other_id, other in usable.items():
                if other_id == tid:
                    continue
                there = ground_point(other[-1].xyxy)
                offset = (there[0] - here[0], there[1] - here[1])
                close = math.hypot(*offset) / height <= cfg.suppression.queue_gap_bh + TOL
                ahead = offset[0] * heading[0] + offset[1] * heading[1] > 0
                if close and ahead and self._stopped_or_stopping(other, speeds[other_id]):
                    return True

        recent = sum(self._stopped_recently(usable[t], speeds[t]) for t in usable)
        return recent / len(usable) >= cfg.suppression.global_slowdown_fraction - TOL

    def _stopped_or_stopping(self, pts: Sequence[TrackPoint], v: Speeds) -> bool:
        cfg = self.cfg
        _, n_before, n_after, _ = self._windows(pts)
        if v[-1] is None:
            return False
        if v[-1] <= cfg.v_stop + TOL:
            return True
        recent_peak = max(_defined(v[-(n_before + n_after) :]), default=0.0)
        return v[-1] < cfg.v_move and recent_peak >= cfg.v_move - TOL

    def _stopped_recently(self, pts: Sequence[TrackPoint], v: Speeds) -> bool:
        cfg = self.cfg
        _, n_before, n_after, n_decel = self._windows(pts)
        tail = _defined(v[-n_after:])
        if len(tail) < n_after or median(tail) > cfg.v_stop + TOL:
            return False
        return max(_defined(v[-(n_before + n_after + n_decel) :]), default=0.0) >= cfg.v_move - TOL


@dataclass
class _Episode:
    start_frame: int | None = None
    start_t: float = 0.0
    motion_since: float | None = None
    armed: bool = True


class LaneObstructionRule(Rule):
    """Flags a vehicle stationary in a driving lane while traffic keeps flowing (docs/02 5.2)."""

    def __init__(self, cfg: BehaviourConfig, n_init: int) -> None:
        super().__init__(cfg, n_init)
        self._episodes: dict[int, _Episode] = {}

    def evaluate(self, ctx: SceneContext) -> list[Incident]:
        cfg, lo = self.cfg, self.cfg.lane_obstruction
        usable = self._usable(ctx)
        now_speed = {tid: v[-1] for tid, v in self._speeds(usable).items()}
        self._episodes = {t: e for t, e in self._episodes.items() if t in ctx.trajectories}
        incidents: list[Incident] = []
        for tid, pts in usable.items():
            v = now_speed[tid]
            if v is None:
                continue
            here = pts[-1]
            episode = self._episodes.setdefault(tid, _Episode())
            if v > cfg.v_stop + TOL:
                episode.start_frame = None
                if episode.motion_since is None:
                    episode.motion_since = here.t_s
                continue
            if episode.start_frame is None:
                episode.start_frame, episode.start_t = here.frame_idx, here.t_s
                moved = None if episode.motion_since is None else here.t_s - episode.motion_since
                if not episode.armed and moved is not None and motion_rearms(moved, lo.rearm_s):
                    episode.armed = True
                episode.motion_since = None
            if not episode.armed or here.t_s - episode.start_t < lo.min_stationary_s - TOL:
                continue
            ground = ground_point(here.xyxy)
            if in_any(ground, cfg.zones.exclusion):
                continue
            if cfg.zones.driving_lanes and not in_any(ground, cfg.zones.driving_lanes):
                continue
            others = [t for t in usable if t != tid]
            moving = sum(
                1 for t in others if now_speed[t] is not None and now_speed[t] >= cfg.v_move - TOL
            )
            flow = moving / len(others) if others else 0.0
            if flow < lo.min_flow_fraction - TOL:
                continue
            episode.armed = False
            margin = (flow - lo.min_flow_fraction) / (1 - lo.min_flow_fraction)
            score = clip01(margin) if lo.min_flow_fraction < 1 else 1.0
            incidents.append(
                Incident(
                    IncidentType.LANE_OBSTRUCTION,
                    (tid,),
                    episode.start_frame,
                    here.frame_idx,
                    episode.start_t,
                    here.t_s,
                    here.xyxy,
                    score,
                )
            )
        return incidents


@dataclass
class _Vehicle:
    points: Sequence[TrackPoint]
    speed: dict[int, float | None]
    index: dict[int, int]
    ground: list[Point]

    @classmethod
    def build(cls, pts: Sequence[TrackPoint], speeds: Speeds) -> _Vehicle:
        return cls(
            pts,
            {p.frame_idx: v for p, v in zip(pts, speeds, strict=True)},
            {p.frame_idx: n for n, p in enumerate(pts)},
            [ground_point(p.xyxy) for p in pts],
        )

    def heading_at(self, frame: int, window: int) -> float | None:
        n = self.index[frame]
        return heading_deg(self.ground[n - window], self.ground[n]) if n >= window else None


class CollisionRule(Rule):
    """Flags converging vehicles whose contact is followed by abrupt change (docs/02 5.3)."""

    def __init__(self, cfg: BehaviourConfig, n_init: int) -> None:
        super().__init__(cfg, n_init)
        self._emitted: set[tuple[int, int, int]] = set()

    def evaluate(self, ctx: SceneContext) -> list[Incident]:
        usable = self._usable(ctx)
        speeds = self._speeds(usable)
        ids = sorted(usable)
        incidents: list[Incident] = []
        for x, i in enumerate(ids):
            for j in ids[x + 1 :]:
                found = self._pair(i, j, usable, speeds)
                if found is not None:
                    incidents.append(found)
        return incidents

    @staticmethod
    def _closing_run(closing: Speeds, v_close: float, min_frames: int) -> tuple[int, int] | None:
        """Latest maximal run of samples at or above v_close that is at least min_frames long."""
        best, start = None, None
        for m, value in enumerate([*closing, None]):
            if value is not None and value >= v_close - TOL:
                start = m if start is None else start
            elif start is not None:
                best = (start, m - 1) if m - start >= min_frames else best
                start = None
        return best

    def _pair(
        self, i: int, j: int, usable: Trajectories, speeds: dict[int, Speeds]
    ) -> Incident | None:
        cfg, co = self.cfg, self.cfg.collision
        pi, pj = usable[i], usable[j]
        period = sample_period(pi)
        smooth = samples_in(cfg.smoothing_window_s, period)
        by_frame = {p.frame_idx: p for p in pj}
        pairs = [(p, by_frame[p.frame_idx]) for p in pi if p.frame_idx in by_frame]
        if len(pairs) <= smooth + co.min_frames:
            return None

        mean_height = (median_height(pi) + median_height(pj)) / 2
        dist = [
            math.dist(ground_point(a.xyxy), ground_point(b.xyxy)) / mean_height for a, b in pairs
        ]
        closing: Speeds = [None] * smooth + [
            (dist[m - smooth] - dist[m]) / cfg.smoothing_window_s for m in range(smooth, len(dist))
        ]
        run = self._closing_run(closing, co.v_close, co.min_frames)
        if run is None:
            return None
        contact = min(range(run[0], len(dist)), key=lambda m: (round(dist[m], 9), m))
        if run[0] > contact + smooth or run[1] < contact - smooth:
            return None

        contact_a, contact_b = pairs[contact]
        key = (i, j, contact_a.frame_idx)
        if key in self._emitted:
            return None
        if box_iou(contact_a.xyxy, contact_b.xyxy) < co.iou_min - TOL:
            return None
        window = post_contact_window(contact, samples_in(co.post_window_s, period), len(pairs))
        frames = [pairs[m][0].frame_idx for m in window]
        vehicles = (_Vehicle.build(pi, speeds[i]), _Vehicle.build(pj, speeds[j]))
        if not self._evidence(vehicles, contact_a.frame_idx, frames, smooth):
            return None

        self._emitted.add(key)
        last_a, last_b = pairs[-1]
        peak = max(_defined(closing[run[0] : run[1] + 1]))
        return Incident(
            IncidentType.COLLISION_LIKE,
            (i, j),
            contact_a.frame_idx,
            last_a.frame_idx,
            contact_a.t_s,
            last_a.t_s,
            union_box([last_a.xyxy, last_b.xyxy]),
            clip01((peak - co.v_close) / co.v_close),
        )

    def _evidence(
        self, vehicles: tuple[_Vehicle, _Vehicle], contact: int, frames: list[int], smooth: int
    ) -> bool:
        cfg, co = self.cfg, self.cfg.collision

        def at_most_stop(veh: _Vehicle, f: int) -> bool:
            s = veh.speed[f]
            return s is not None and s <= cfg.v_stop + TOL

        for veh in vehicles:
            pre_speed = veh.speed[contact]
            was_moving = pre_speed is not None and pre_speed >= cfg.v_move - TOL
            if was_moving and any(at_most_stop(veh, f) for f in frames):
                return True
            pre_heading = veh.heading_at(contact, smooth)
            if pre_heading is None:
                continue
            for f in frames:
                if at_most_stop(veh, f) or veh.speed[f] is None:
                    continue
                post_heading = veh.heading_at(f, smooth)
                if post_heading is not None and heading_change_exceeds(
                    pre_heading, post_heading, co.heading_change_deg
                ):
                    return True
        return any(all(at_most_stop(veh, f) for veh in vehicles) for f in frames)
