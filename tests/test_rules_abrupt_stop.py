"""White-box target tests for AbruptStopRule (docs/02 section 5.1, TC-UT-04).

They define the behaviour Phase 2 must implement and fail with NotImplementedError until then.
Speed samples: v_k = (sum of the last five per-frame steps) / 50 in body-heights per second.
Definitions pinned here (decision D-19): pre-speed is the median of v over the window_before_s
samples ending at the last moving sample a (v >= v_move); b is the first later sample with
v <= v_stop; deceleration time is (b - a) * dt; post-speed is the median of v over the
window_after_s samples starting at b; the flag is raised at the last of those samples.
Comparisons use a 1e-9 tolerance.
"""

from __future__ import annotations

import pytest

from helpers import N_INIT, behaviour_cfg, hold, involving, replay, run, square, track
from trafficwatch.behaviour.rules import AbruptStopRule
from trafficwatch.types import IncidentType, TrackPoint

pytestmark = pytest.mark.target

MOVE_FRAMES = 30
SUBJECT = 1


def fillers(n: int, count: int) -> dict[int, list[TrackPoint]]:
    """Vehicles cruising at 2.0 bh/s in lanes at least 3 body heights from the subject's lane."""
    lanes = [200.0, 800.0, 1100.0, 1400.0, 1700.0]
    return {100 + i: track(run(n, 20.0), x0=100.0, y0=lanes[i]) for i in range(count)}


def scene(subject_steps, *, others=None, filler_count=4, x0=100.0):
    subject = track(subject_steps, x0=x0)
    n = len(subject_steps)
    tracks = {SUBJECT: subject, **fillers(n, filler_count)}
    tracks.update(others or {})
    return tracks


def flags(tracks, *, cfg=None, n_init=N_INIT):
    rule = AbruptStopRule(cfg or behaviour_cfg(), n_init)
    return involving(replay(rule, tracks), SUBJECT)


def test_clean_positive_flags_at_hand_computed_frame():
    # v_5..v_30 = 2.0; v_31..v_35 = 1.6, 1.2, 0.8, 0.4, 0.0. a = 32, b = 35, decel 0.3 s.
    # Post window [35, 39] has median 0, pre median is 2.0, drop ratio 1.0: flag at frame 39.
    found = flags(scene(run(MOVE_FRAMES, 20.0) + hold(20)))
    assert len(found) == 1
    assert found[0].type is IncidentType.ABRUPT_STOP
    assert found[0].track_ids == (SUBJECT,)
    assert found[0].frame_flag == 39
    assert found[0].t_flag_s == pytest.approx(3.9)
    assert 30 <= found[0].frame_onset <= 32


def test_constant_speed_is_not_flagged():
    assert flags(scene(run(50, 20.0))) == []


def test_gradual_deceleration_over_five_seconds_is_not_flagged():
    # Step ramps 20 -> 0 over 50 frames, so v falls from 1.0 to 0.25 in about 1.9 s > max_decel_s.
    ramp = [20.0 * (1 - i / 50) for i in range(1, 51)]
    steps = run(MOVE_FRAMES, 20.0) + [(dx, 0.0) for dx in ramp] + hold(10)
    assert flags(scene(steps)) == []


@pytest.mark.parametrize(
    ("pre_step", "expected"),
    [(10.0, 1), (9.9, 0)],
    ids=["v_move_1.00_flags", "v_move_0.99_none"],
)
def test_v_move_boundary(pre_step, expected):
    # pre_step 10: v_30 = 1.0 (a = 30), v_31..v_34 = 0.8, 0.6, 0.4, 0.2 so b = 34, flag at 38.
    # pre_step 9.9: v never reaches 1.0, so there is no moving sample.
    found = flags(scene(run(MOVE_FRAMES, pre_step) + hold(20)))
    assert len(found) == expected
    if expected:
        assert found[0].frame_flag == 38


@pytest.mark.parametrize(
    ("post_step", "expected"),
    [(2.5, 1), (2.6, 0)],
    ids=["v_stop_0.25_flags", "v_stop_0.26_none"],
)
def test_v_stop_boundary(post_step, expected):
    # Steady post speed is post_step / 10: 0.25 is stationary, 0.26 never is. For 2.5:
    # v_31..v_35 = 1.65, 1.3, 0.95, 0.6, 0.25 so a = 32, b = 35, post median 0.25, ratio 0.875.
    found = flags(scene(run(MOVE_FRAMES, 20.0) + run(30, post_step)))
    assert len(found) == expected
    if expected:
        assert found[0].frame_flag == 39


@pytest.mark.parametrize(
    ("post_step", "expected"),
    [(2.0, 1), (2.1, 0)],
    ids=["drop_ratio_0.80_flags", "drop_ratio_0.79_none"],
)
def test_drop_ratio_boundary(post_step, expected):
    # Pre speed 1.0, post 0.2 or 0.21 (both at or below v_stop, so only the ratio binds):
    # (1.0 - 0.2) / 1.0 = 0.80 flags, (1.0 - 0.21) / 1.0 = 0.79 does not.
    assert len(flags(scene(run(MOVE_FRAMES, 10.0) + run(30, post_step)))) == expected


@pytest.mark.parametrize(
    ("stage_two_frames", "expected"),
    [(10, 1), (11, 0)],
    ids=["max_decel_1.0s_flags", "max_decel_1.1s_none"],
)
def test_max_decel_boundary(stage_two_frames, expected):
    # Step 20, then L frames at step 6 (v = 0.6), then stop. a = 33 (v = 1.16); the first
    # v <= 0.25 is b = 30 + L + 3, so deceleration takes L samples: 1.0 s flags, 1.1 s does not.
    steps = run(MOVE_FRAMES, 20.0) + run(stage_two_frames, 6.0) + hold(20)
    found = flags(scene(steps))
    assert len(found) == expected
    if expected:
        assert found[0].frame_flag == 47


def test_track_younger_than_n_init_is_ignored():
    assert flags(scene(run(MOVE_FRAMES, 20.0) + hold(20)), n_init=100) == []


@pytest.mark.parametrize(
    ("gap_px", "expected"),
    [(150.0, 0), (160.0, 1)],
    ids=["queue_gap_1.5bh_suppressed", "queue_gap_1.6bh_flags"],
)
def test_queue_suppression(gap_px, expected):
    # Subject stops with its ground point at x = 700; a parked vehicle sits gap_px ahead.
    lead = {2: track(hold(60), x0=700.0 + gap_px)}
    steps = run(MOVE_FRAMES, 20.0) + hold(30)
    assert len(flags(scene(steps, others=lead))) == expected


@pytest.mark.parametrize(
    ("filler_count", "expected"),
    [(2, 0), (3, 1)],
    ids=["slowdown_2_of_4_suppressed", "slowdown_2_of_5_flags"],
)
def test_global_slowdown_suppression(filler_count, expected):
    # Two vehicles stop in the same window; 2 of 4 active tracks (0.5) suppresses, 2 of 5 (0.4) not.
    steps = run(MOVE_FRAMES, 20.0) + hold(20)
    tracks = {SUBJECT: track(steps, x0=100.0), 2: track(steps, x0=100.0, y0=200.0)}
    for i, lane in enumerate((800.0, 1100.0, 1400.0)[:filler_count]):
        tracks[100 + i] = track(run(len(steps), 20.0), x0=100.0, y0=lane)
    assert len(flags(tracks)) == expected


@pytest.mark.parametrize(
    ("zone", "expected"),
    [(square(600, 400, 800, 600), 0), (square(1000, 400, 1200, 600), 1)],
    ids=["stop_inside_exclusion_suppressed", "stop_outside_exclusion_flags"],
)
def test_exclusion_zone_suppression(zone, expected):
    # The subject stops with its ground point at (700, 500).
    cfg = behaviour_cfg(exclusion=(zone,))
    assert len(flags(scene(run(MOVE_FRAMES, 20.0) + hold(20)), cfg=cfg)) == expected
