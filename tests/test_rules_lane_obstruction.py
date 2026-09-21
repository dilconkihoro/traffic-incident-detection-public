"""White-box target tests for LaneObstructionRule (docs/02 section 5.2).

Fail with NotImplementedError until Phase 2. Definitions pinned here (decision D-19): a sample
is stationary when v <= v_stop; an episode starts at the first sample of a run of stationary
samples and lasts (k - start) * dt; the flag is raised at the first k where that reaches
min_stationary_s. An empty driving_lanes list means the whole frame. Flow fraction is moving
tracks (v >= v_move) over all other active tracks.
"""

from __future__ import annotations

import pytest

from helpers import N_INIT, behaviour_cfg, hold, involving, replay, run, square, track
from trafficwatch.behaviour.rules import LaneObstructionRule
from trafficwatch.types import IncidentType

pytestmark = pytest.mark.target

SUBJECT = 1
# Step 10 for frames 1..20 gives v_20 = 1.0; v_21..v_24 = 0.8, 0.6, 0.4, 0.2, so the first
# stationary sample is 24 (the episode start) and its box sits at ground point (500, 500).
APPROACH = run(20, 10.0)
ONSET_FRAME = 24
FLOW_STEP = 20.0


def others(moving: int, parked: int, n: int):
    """Tracks 10..: cruise at 2.0 bh/s in lane y = 1000 + 300 i. Tracks 50..: parked."""
    tracks = {}
    for i in range(moving):
        tracks[10 + i] = track(run(n, FLOW_STEP), x0=100.0, y0=1000.0 + 300.0 * i)
    for i in range(parked):
        tracks[50 + i] = track(hold(n), x0=100.0, y0=3000.0 + 300.0 * i)
    return tracks


def flags(steps, *, moving=5, parked=0, cfg=None):
    tracks = {SUBJECT: track(steps, x0=300.0)} | others(moving, parked, len(steps))
    rule = LaneObstructionRule(cfg or behaviour_cfg(), N_INIT)
    return involving(replay(rule, tracks), SUBJECT)


def test_flags_exactly_min_stationary_s_after_onset():
    # Stationary run starts at frame 24; 15.0 s later is frame 174 (t = 17.4 s).
    found = flags(APPROACH + hold(154))
    assert len(found) == 1
    assert found[0].type is IncidentType.LANE_OBSTRUCTION
    assert found[0].frame_onset == ONSET_FRAME
    assert found[0].frame_flag == ONSET_FRAME + 150
    assert found[0].t_flag_s - found[0].t_onset_s == pytest.approx(15.0)


def test_one_frame_short_of_15_seconds_is_not_flagged():
    # Last sample is frame 173, so the stationary span is 14.9 s.
    assert flags(APPROACH + hold(153)) == []


def test_everyone_stopped_is_congestion_not_obstruction():
    assert flags(APPROACH + hold(154), moving=0, parked=5) == []


@pytest.mark.parametrize(
    ("moving", "parked", "expected"),
    [(3, 7, 1), (2, 8, 0)],
    ids=["flow_0.30_flags", "flow_0.20_none"],
)
def test_min_flow_fraction_boundary(moving, parked, expected):
    assert len(flags(APPROACH + hold(154), moving=moving, parked=parked)) == expected


@pytest.mark.parametrize(
    ("driving", "exclusion", "expected"),
    [
        ((square(400, 400, 600, 600),), (), 1),
        ((square(1000, 400, 1200, 600),), (), 0),
        ((), (square(400, 400, 600, 600),), 0),
    ],
    ids=["inside_driving_lane_flags", "outside_driving_lane_none", "inside_exclusion_none"],
)
def test_zones(driving, exclusion, expected):
    cfg = behaviour_cfg(driving_lanes=driving, exclusion=exclusion)
    assert len(flags(APPROACH + hold(154), cfg=cfg)) == expected


def test_flags_once_per_stationary_episode():
    assert len(flags(APPROACH + hold(254))) == 1


@pytest.mark.parametrize(
    ("motion_frames", "expected"),
    [(60, 2), (30, 1)],
    ids=["6s_motion_rearms", "3s_motion_does_not_rearm"],
)
def test_rearm_after_motion(motion_frames, expected):
    # First episode flags at frame 174. The vehicle then moves for 6 s or 3 s (rearm_s is 5 s)
    # and stops for another 20 s, which is long enough for a second flag only if re-armed.
    steps = APPROACH + hold(154) + run(motion_frames, FLOW_STEP) + hold(200)
    assert len(flags(steps)) == expected
