"""White-box target tests for CollisionRule (docs/02 section 5.3).

Fail with NotImplementedError until Phase 2. Definitions pinned here (decision D-19): pair
distance is ground-point distance over the mean body height; closing speed is the fall in that
distance over the 0.5 s smoothing window; contact is the closest-approach sample, where full-box
IoU must reach iou_min; evidence is checked on the post_window_s after contact.

Boxes are 110 x 100 px, so a ground-point separation of 90 px overlaps 20 x 100 px: IoU is
2000 / (22000 - 2000) = 0.10 exactly. A separation of 92 px gives 1800 / 20200 = 0.089.
"""

from __future__ import annotations

import pytest

from helpers import N_INIT, behaviour_cfg, heading_step, hold, replay, run, track
from trafficwatch.behaviour.rules import CollisionRule
from trafficwatch.types import IncidentType

pytestmark = pytest.mark.target

A, B = 1, 2


def flags(tracks):
    return replay(CollisionRule(behaviour_cfg(), N_INIT), tracks)


def approach_parked(step: float, frames: int, stop_gap: float, after=None):
    """Vehicle A closes on a parked B at `step` px/frame and stops with `stop_gap` px between
    ground points (or runs `after` steps instead)."""
    x0 = 100.0
    b_x = x0 + stop_gap + step * frames
    a_steps = run(frames, step) + (after if after is not None else hold(30))
    return {A: track(a_steps, x0=x0), B: track(hold(len(a_steps)), x0=b_x)}


def test_head_on_both_stop_flags_after_contact():
    # Each vehicle moves 10 px/frame toward the other from 890 px apart: closing 2.0 bh/s from
    # frame 5. After 40 frames the gap is 890 - 800 = 90 px (IoU 0.10) and both stop.
    tracks = {
        A: track(run(40, 10.0) + hold(30), x0=0.0),
        B: track(run(40, -10.0) + hold(30), x0=890.0),
    }
    found = flags(tracks)
    assert len(found) == 1
    assert found[0].type is IncidentType.COLLISION_LIKE
    assert set(found[0].track_ids) == {A, B}
    assert found[0].frame_flag >= 40


def test_both_stationary_branch_without_abrupt_drop():
    # Both approach at 0.8 bh/s (below v_move, so no abrupt drop) with closing 1.6 bh/s, then stop
    # at a 90 px gap: 890 - 16 * 50 = 90 after 50 frames.
    tracks = {
        A: track(run(50, 8.0) + hold(30), x0=0.0),
        B: track(run(50, -8.0) + hold(30), x0=890.0),
    }
    assert len(flags(tracks)) == 1


def test_abrupt_drop_branch_without_both_stationary():
    # A closes at 2.0 bh/s on B, which creeps away at 0.4 bh/s (closing 1.6), then A stops at a
    # 90 px gap after 40 frames: 90 + 16 * 40 = 730 px start gap. B keeps moving, so only A's
    # abrupt speed drop is evidence.
    tracks = {
        A: track(run(40, 20.0) + hold(30), x0=0.0),
        B: track(run(70, 4.0), x0=730.0),
    }
    assert len(flags(tracks)) == 1


def test_heading_change_branch_flags_on_a_90_degree_turn():
    # A reaches a 90 px gap to parked B at frame 40 (closest approach), then turns to heading 90
    # degrees at 1.4 bh/s: the gap grows, its speed stays above v_stop, so only the heading
    # change (about 80 degrees measured over the smoothing window) is evidence.
    steps = run(40, 20.0) + [heading_step(14.0, 90.0)] * 30
    tracks = {A: track(steps, x0=0.0), B: track(hold(70), x0=890.0)}
    assert len(flags(tracks)) == 1


def test_gentle_20_degree_change_is_not_flagged():
    # Same approach, then a 20 degree deflection at 1.4 bh/s: well under heading_change_deg even
    # allowing for the smoothing window, and no speed drop or stop. The exact 45 versus 44 degree
    # boundary depends on the heading estimator and is covered by its own Phase 2 unit test.
    steps = run(40, 20.0) + [heading_step(14.0, 20.0)] * 30
    tracks = {A: track(steps, x0=0.0), B: track(hold(70), x0=890.0)}
    assert flags(tracks) == []


def test_overlap_without_evidence_is_occlusion():
    # Head-on pair at 1.0 bh/s each pass through each other and keep speed and heading.
    tracks = {A: track(run(120, 10.0), x0=0.0), B: track(run(120, -10.0), x0=890.0)}
    assert flags(tracks) == []


@pytest.mark.parametrize(
    ("step", "expected"),
    [(15.0, 1), (14.0, 0)],
    ids=["v_close_1.5_flags", "v_close_1.4_none"],
)
def test_v_close_boundary(step, expected):
    # Closing speed against a parked vehicle is step / 10 bh/s; A stops at a 90 px gap.
    assert len(flags(approach_parked(step, 40, 90.0))) == expected


@pytest.mark.parametrize(
    ("stop_gap", "expected"),
    [(90.0, 1), (92.0, 0)],
    ids=["iou_0.10_flags", "iou_0.089_none"],
)
def test_iou_min_boundary(stop_gap, expected):
    assert len(flags(approach_parked(20.0, 40, stop_gap))) == expected


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(6, 1), (5, 0)],
    ids=["min_frames_3_flags", "min_frames_2_none"],
)
def test_min_frames_boundary(frames, expected):
    # Closing speed exists from sample 5 (needs a full 0.5 s window). Step 20 from rest gives
    # c = 2.0 for samples 5..T and 1.6 for sample T + 1, then 1.2: T - 3 qualifying samples,
    # so 6 frames of approach give 3 (flags) and 5 frames give 2 (does not).
    assert len(flags(approach_parked(20.0, frames, 90.0))) == expected
