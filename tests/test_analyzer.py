"""TC-IT-02: trajectories feed the analyzer; the bounded store keeps the rules working."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from helpers import behaviour_cfg, hold, run, track
from trafficwatch.behaviour.analyzer import BehaviourAnalyzer, history_samples
from trafficwatch.config import load_config
from trafficwatch.tracking import TrajectoryStore
from trafficwatch.types import IncidentType, Track

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def pipeline_cfg():
    return dataclasses.replace(load_config(SHIPPED, env={}), behaviour=behaviour_cfg())


def as_track(track_id: int, point, n: int) -> Track:
    return Track(track_id, point.xyxy, "car", age=n, hits=n, confirmed=True)


def feed(analyzer: BehaviourAnalyzer, trajectories):
    incidents = []
    last = max(len(points) for points in trajectories.values())
    for f in range(last):
        tracks = [as_track(tid, pts[f], f + 1) for tid, pts in trajectories.items() if len(pts) > f]
        incidents.extend(analyzer.step(tracks, f, round(f * 0.1, 9)))
    return incidents


def test_history_samples_is_derived_from_the_longest_window():
    # Longest window is max(1.0 + 0.5 + 1.0, 1.5 + 2 * 0.5) = 2.5 s; plus 2 * 0.5 s margin = 3.5 s.
    # At 10 Hz that is 35 samples, plus min_frames (3) and 1: 39. At 5 Hz: ceil(17.5) + 4 = 22.
    assert history_samples(behaviour_cfg(), 0.1) == 39
    assert history_samples(behaviour_cfg(), 0.2) == 22


def test_store_is_bounded_and_returns_only_requested_tracks():
    store = TrajectoryStore(max_samples=3)
    for f in range(5):
        store.add(Track(1, (0, 0, 10, 10), "car", f, f, True), f, f * 0.1)
    store.add(Track(2, (0, 0, 10, 10), "car", 1, 1, True), 4, 0.4)
    assert [p.frame_idx for p in store.history(1)] == [2, 3, 4]
    assert list(store.snapshot([2, 9])) == [2]
    assert store.history(9) == ()


def test_store_drops_tracks_not_seen_recently():
    store = TrajectoryStore(max_samples=3)
    store.add(Track(1, (0, 0, 1, 1), "car", 1, 1, True), 5, 0.5)
    store.add(Track(2, (0, 0, 1, 1), "car", 1, 1, True), 40, 4.0)
    store.drop_stale(10)
    assert store.history(1) == ()
    assert len(store.history(2)) == 1


def test_store_needs_room_for_two_samples():
    with pytest.raises(ValueError, match="max_samples"):
        TrajectoryStore(max_samples=1)


def test_analyzer_flags_an_abrupt_stop_through_the_bounded_store():
    steps = run(30, 20.0) + hold(30)
    trajectories = {1: track(steps, x0=100.0)}
    for i, lane in enumerate((200.0, 800.0, 1100.0, 1400.0)):
        trajectories[100 + i] = track(run(len(steps), 20.0), x0=100.0, y0=lane)
    found = feed(BehaviourAnalyzer(pipeline_cfg(), period_s=0.1), trajectories)
    assert [(i.type, i.track_ids, i.frame_flag) for i in found] == [
        (IncidentType.ABRUPT_STOP, (1,), 39)
    ]


def test_analyzer_flags_a_head_on_collision_and_ignores_unconfirmed_tracks():
    trajectories = {
        1: track(run(40, 10.0) + hold(30), x0=0.0),
        2: track(run(40, -10.0) + hold(30), x0=890.0),
    }
    found = feed(BehaviourAnalyzer(pipeline_cfg(), period_s=0.1), trajectories)
    assert [i.type for i in found] == [IncidentType.COLLISION_LIKE]
    tentative = Track(1, (0, 0, 10, 10), "car", 1, 1, confirmed=False)
    assert BehaviourAnalyzer(pipeline_cfg(), 0.1).step([tentative], 0, 0.0) == []
