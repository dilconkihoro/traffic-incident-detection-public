from __future__ import annotations

import math
from collections.abc import Sequence

from trafficwatch.behaviour.rules import AbruptStopRule, CollisionRule, LaneObstructionRule
from trafficwatch.config import BehaviourConfig, PipelineConfig
from trafficwatch.tracking import TrajectoryStore
from trafficwatch.types import Incident, SceneContext, Track


def history_samples(behaviour: BehaviourConfig, period_s: float) -> int:
    """Samples of history the rules need, derived from the longest configured window."""
    stop, collision = behaviour.abrupt_stop, behaviour.collision
    smooth = behaviour.smoothing_window_s
    window_s = max(
        stop.window_before_s + stop.window_after_s + stop.max_decel_s,
        collision.post_window_s + 2 * smooth,
    )
    return math.ceil(round((window_s + 2 * smooth) / period_s, 9)) + collision.min_frames + 1


class BehaviourAnalyzer:
    """Runs the three rules once per frame over the confirmed tracks."""

    def __init__(self, cfg: PipelineConfig, period_s: float) -> None:
        self.cfg = cfg
        self._store = TrajectoryStore(history_samples(cfg.behaviour, period_s))
        n_init = cfg.tracker.n_init
        self._rules = (
            AbruptStopRule(cfg.behaviour, n_init),
            LaneObstructionRule(cfg.behaviour, n_init),
            CollisionRule(cfg.behaviour, n_init),
        )

    def step(self, tracks: Sequence[Track], frame_idx: int, t_s: float) -> list[Incident]:
        confirmed = [t for t in tracks if t.confirmed]
        for track in confirmed:
            self._store.add(track, frame_idx, t_s)
        ctx = SceneContext(frame_idx, t_s, self._store.snapshot(t.track_id for t in confirmed))
        incidents = [incident for rule in self._rules for incident in rule.evaluate(ctx)]
        self._store.drop_stale(frame_idx - self.cfg.tracker.max_age)
        return incidents
