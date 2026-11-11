"""White-box tests for anonymous vehicle identity (decision D-28): identifiers, colour naming,
direction of travel and the majority votes, each against a hand-built input."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from trafficwatch.config import load_config
from trafficwatch.identity import CODE_ALPHABET, VehicleRegistry, _code, colour_name
from trafficwatch.types import Frame, Track

SHIPPED = "configs/pipeline.yaml"
SHA = "a" * 64


@pytest.fixture(scope="module")
def cfg():
    return load_config(SHIPPED, env={}).identity


def solid(bgr, size=(40, 40)) -> np.ndarray:
    image = np.zeros((*size, 3), dtype=np.uint8)
    image[:] = bgr
    return image


def frame(idx: int, image: np.ndarray | None = None, fps: float = 10.0) -> Frame:
    return Frame(idx, idx / fps, solid((128, 128, 128)) if image is None else image)


def track(track_id: int, xyxy, cls_name: str = "car") -> Track:
    return Track(track_id, xyxy, cls_name, 5, 5, True)


FULL = (0.0, 0.0, 40.0, 40.0)


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((255, 255, 255), "white"),
        ((0, 0, 0), "black"),
        ((128, 128, 128), "grey"),
        ((0, 0, 255), "red"),  # hue 0
        ((0, 128, 255), "orange"),  # hue 15
        ((0, 255, 255), "yellow"),  # hue 30
        ((0, 255, 0), "green"),  # hue 60
        ((255, 0, 0), "blue"),  # hue 120
        ((255, 0, 170), "purple"),  # hue 140
        ((60, 0, 255), "red"),  # hue 173, the wrap-around end of red
    ],
)
def test_solid_colours_are_named_correctly(cfg, bgr, expected):
    assert colour_name(solid(bgr), FULL, cfg) == expected


def test_only_the_central_part_of_the_box_decides_the_colour(cfg):
    image = solid((255, 0, 0))  # blue surround, like road or a neighbouring vehicle
    image[12:28, 12:28] = (0, 0, 255)  # red body in the middle half
    assert colour_name(image, FULL, cfg) == "red"


def test_a_box_outside_the_image_has_no_colour(cfg):
    assert colour_name(solid((0, 0, 255)), (100.0, 100.0, 140.0, 140.0), cfg) is None


def test_identifier_is_prefixed_short_and_from_the_safe_alphabet(cfg):
    registry = VehicleRegistry(cfg, SHA)
    vehicle_id = registry.observe(track(1, FULL), frame(0))
    assert re.fullmatch(rf"VH-[{CODE_ALPHABET}]{{4}}", vehicle_id)


def test_identifier_is_stable_for_a_track_and_distinct_between_tracks(cfg):
    registry = VehicleRegistry(cfg, SHA)
    first = registry.observe(track(1, FULL), frame(0))
    other = registry.observe(track(2, FULL), frame(0))
    assert registry.observe(track(1, FULL), frame(1)) == first
    assert registry.vehicle_id(1) == first and first != other


def test_same_video_gives_the_same_identifiers_and_another_video_does_not(cfg):
    def ids(sha):
        registry = VehicleRegistry(cfg, sha)
        return [registry.observe(track(t, FULL), frame(0)) for t in (1, 2, 3)]

    assert ids(SHA) == ids(SHA)
    assert ids(SHA) != ids("b" * 64)


def test_a_clash_lengthens_the_code_instead_of_reusing_it(cfg):
    registry = VehicleRegistry(cfg, SHA)
    import hashlib

    digest = hashlib.sha256(f"{SHA}:7".encode()).digest()
    registry._codes.add(_code(digest, cfg.code_length))

    vehicle_id = registry.observe(track(7, FULL), frame(0))

    assert vehicle_id == f"VH-{_code(digest, cfg.code_length + 1)}"


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((0, 100), (100, 100), "moving right"),
        ((100, 100), (0, 100), "moving left"),
        ((100, 0), (100, 100), "moving down the frame"),
        ((100, 100), (100, 0), "moving up the frame"),
        ((100, 100), (110, 105), "stationary"),  # about 0.6 body heights, under min_travel_bh 1.0
    ],
)
def test_direction_comes_from_net_movement_in_body_heights(cfg, start, end, expected):
    registry = VehicleRegistry(cfg, SHA)
    for idx, (cx, cy) in enumerate((start, end)):
        box = (cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0)  # body height 20 px
        registry.observe(track(1, box), frame(idx, solid((0, 0, 0), (300, 300))))
    (profile,) = registry.profiles()
    assert profile.direction == expected


def test_class_and_colour_are_majority_votes_with_deterministic_ties(cfg):
    every_frame = dataclasses.replace(cfg, colour_sample_every=1)
    registry = VehicleRegistry(every_frame, SHA)
    sightings = [("van", (255, 255, 255)), ("car", (255, 255, 255)), ("van", (0, 0, 255))]
    for idx, (cls_name, bgr) in enumerate(sightings):
        registry.observe(track(1, FULL, cls_name), frame(idx, solid(bgr)))
    (profile,) = registry.profiles()
    assert (profile.cls_name, profile.colour) == ("van", "white")

    tied = VehicleRegistry(every_frame, SHA)
    tied.observe(track(1, FULL, "van"), frame(0))
    tied.observe(track(1, FULL, "car"), frame(1))
    assert tied.profiles()[0].cls_name == "car"  # 1 to 1: alphabetical, not arrival order


def test_colour_is_sampled_only_every_nth_sighting(cfg):
    sparse = dataclasses.replace(cfg, colour_sample_every=3)
    registry = VehicleRegistry(sparse, SHA)
    colours = [(0, 0, 255), (255, 255, 255), (255, 255, 255), (0, 0, 255)]
    for idx, bgr in enumerate(colours):
        registry.observe(track(1, FULL), frame(idx, solid(bgr)))
    assert registry.profiles()[0].colour == "red"  # sightings 0 and 3 sampled, both red


def test_profile_records_sightings_and_reads_as_a_description(cfg):
    registry = VehicleRegistry(cfg, SHA)
    for idx in range(5):
        cx = 20.0 + 30.0 * idx
        registry.observe(
            track(4, (cx - 10, 40.0, cx + 10, 60.0), "bus"),
            frame(idx, solid((255, 255, 255), (100, 200))),
        )
    (profile,) = registry.profiles()
    assert (profile.track_id, profile.frames_seen) == (4, 5)
    assert (profile.first_seen_s, profile.last_seen_s) == (0.0, 0.4)
    assert profile.description == "white bus, moving right, seen 0.0 s to 0.4 s"
    assert profile.to_dict()["description"] == profile.description


def test_profiles_keep_order_of_first_appearance(cfg):
    registry = VehicleRegistry(cfg, SHA)
    for track_id in (5, 2, 9):
        registry.observe(track(track_id, FULL), frame(0))
    assert [p.track_id for p in registry.profiles()] == [5, 2, 9]
