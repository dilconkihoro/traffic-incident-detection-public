"""Hand-computed checks of the pure helpers behind the behaviour rules."""

from __future__ import annotations

import pytest

from helpers import hold, run, square, track
from trafficwatch.behaviour import kinematics as kin


def test_speed_is_none_until_a_full_window_then_step_over_body_height():
    # 10 px per frame at 10 Hz is 100 px/s; height 100 px, so 1.0 body height per second.
    series = kin.speed_series(track(run(8, 10.0), x0=0.0), window_s=0.5)
    assert series[:5] == [None] * 5
    assert series[5:] == [pytest.approx(1.0)] * 4


def test_speed_uses_the_window_in_samples_when_frames_are_strided():
    # Samples 0.2 s apart: a 0.5 s window is round(2.5) = 2 samples, 20 px each step.
    points = track(run(6, 20.0), x0=0.0)
    strided = [p.__class__(p.frame_idx * 2, round(p.t_s * 2, 9), p.xyxy) for p in points]
    series = kin.speed_series(strided, window_s=0.5)
    assert series[:2] == [None, None]
    assert series[2] == pytest.approx(40.0 / (0.5 * 100.0))


def test_stationary_track_has_zero_speed():
    assert kin.speed_series(track(hold(8), x0=0.0), 0.5)[5:] == [0.0] * 4


def test_ground_point_and_height():
    assert kin.ground_point((10.0, 20.0, 30.0, 60.0)) == (20.0, 60.0)
    assert kin.median_height(track(hold(4), x0=0.0)) == 100.0


def test_point_in_polygon_convex_and_concave():
    box = square(0, 0, 10, 10)
    assert kin.point_in_polygon((5, 5), box)
    assert not kin.point_in_polygon((15, 5), box)
    notch = ((0, 0), (10, 0), (10, 10), (5, 4), (0, 10))
    assert kin.point_in_polygon((2, 2), notch)
    assert not kin.point_in_polygon((5, 8), notch)
    assert kin.in_any((5, 5), [square(20, 20, 30, 30), box])
    assert not kin.in_any((5, 5), [])


def test_box_iou_hand_values():
    assert kin.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert kin.box_iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0.0
    # 110 x 100 boxes overlapping by 20 px: 2000 / (22000 - 2000).
    assert kin.box_iou((0, 0, 110, 100), (90, 0, 200, 100)) == pytest.approx(0.10)


def test_union_box():
    assert kin.union_box([(0, 5, 10, 20), (4, 0, 12, 8)]) == (0, 0, 12, 20)


def test_heading_and_angle_difference():
    assert kin.heading_deg((0, 0), (10, 10)) == pytest.approx(45.0)
    assert kin.heading_deg((0, 0), (-10, 0)) == pytest.approx(180.0)
    assert kin.heading_deg((5, 5), (5, 5 + 1e-9)) is None
    assert kin.angle_diff_deg(350, 10) == pytest.approx(20.0)
    assert kin.angle_diff_deg(-170, 170) == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("post", "expected"),
    [(45.0, True), (44.0, False), (46.0, True), (-45.0, True), (44.9999999995, True)],
    ids=["45_flags", "44_none", "46_flags", "minus_45_flags", "within_tolerance_flags"],
)
def test_heading_change_boundary_at_45_degrees(post, expected):
    assert kin.heading_change_exceeds(0.0, post, 45.0) is expected


@pytest.mark.parametrize(
    ("index", "included"),
    [(11, True), (25, True), (26, False), (10, False)],
    ids=["first_after_contact", "at_1.5s", "at_1.6s", "contact_itself"],
)
def test_post_contact_window_boundary(index, included):
    # Contact at sample 10, 1.5 s at 10 Hz is 15 samples: 11 through 25 count as evidence.
    assert (index in kin.post_contact_window(10, 15, 100)) is included


def test_post_contact_window_is_clipped_to_available_samples():
    assert list(kin.post_contact_window(10, 15, 14)) == [11, 12, 13]
    assert list(kin.post_contact_window(10, 15, 11)) == []


@pytest.mark.parametrize(
    ("motion_s", "expected"),
    [(5.0, True), (4.99, False), (4.9999999995, True), (6.4, True), (3.4, False)],
    ids=["exactly_5s_rearms", "4.99s_does_not", "within_tolerance_rearms", "6.4s", "3.4s"],
)
def test_rearm_boundary_at_exactly_rearm_s(motion_s, expected):
    assert kin.motion_rearms(motion_s, 5.0) is expected


def test_samples_in_and_clip():
    assert kin.samples_in(0.5, 0.1) == 5
    assert kin.samples_in(1.0, 0.1) == 10
    assert kin.samples_in(0.04, 0.1) == 1
    assert kin.clip01(-0.2) == 0.0
    assert kin.clip01(1.7) == 1.0
    assert kin.clip01(0.4) == 0.4
