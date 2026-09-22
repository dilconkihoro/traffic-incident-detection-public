"""TC-IT-01: detections initialise tracks; only confirmed, freshly matched tracks are emitted."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from trafficwatch.config import load_config
from trafficwatch.tracking import VehicleTracker
from trafficwatch.types import Detection, Frame

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"
WIDTH, HEIGHT = 320, 240
BOX_W, BOX_H = 60, 40


def texture(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (BOX_H, BOX_W, 3), dtype=np.uint8)


def scene(objects: dict[int, int], y_of: dict[int, int]) -> tuple[Frame, list[Detection]]:
    """Objects are textured patches on noise; ``objects`` maps texture seed to left edge x."""
    image = np.random.default_rng(999).integers(60, 90, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    detections = []
    for seed, x in objects.items():
        y = y_of[seed]
        image[y : y + BOX_H, x : x + BOX_W] = texture(seed)
        detections.append(Detection((x, y, x + BOX_W, y + BOX_H), 0.9, 0, "car"))
    return Frame(0, 0.0, image), detections


@pytest.fixture(scope="module")
def tracker_cfg():
    return load_config(SHIPPED, env={}).tracker


def test_track_is_confirmed_only_after_n_init_matches_and_keeps_its_id(tracker_cfg):
    tracker = VehicleTracker(tracker_cfg)
    emitted = []
    for f in range(8):
        frame, dets = scene({1: 20 + 4 * f}, {1: 100})
        emitted.append(tracker.update(dets, dataclasses.replace(frame, idx=f)))
    assert [len(e) for e in emitted[: tracker_cfg.n_init - 1]] == [0] * (tracker_cfg.n_init - 1)
    later = [tracks[0] for tracks in emitted[tracker_cfg.n_init :]]
    assert len({t.track_id for t in later}) == 1
    assert all(t.confirmed and t.cls_name == "car" for t in later)
    x1, y1, x2, y2 = later[-1].xyxy
    assert (x2 - x1) == pytest.approx(BOX_W, abs=6) and (y2 - y1) == pytest.approx(BOX_H, abs=6)


def test_two_objects_get_two_distinct_ids(tracker_cfg):
    tracker = VehicleTracker(tracker_cfg)
    for f in range(8):
        frame, dets = scene({1: 20 + 4 * f, 2: 200 - 4 * f}, {1: 40, 2: 170})
        tracks = tracker.update(dets, frame)
    assert len({t.track_id for t in tracks}) == 2


def test_a_missed_frame_is_not_emitted_but_the_id_survives(tracker_cfg):
    tracker = VehicleTracker(tracker_cfg)
    ids = []
    for f in range(10):
        frame, dets = scene({1: 20 + 4 * f}, {1: 100})
        tracks = tracker.update([] if f == 6 else dets, frame)
        if f == 6:
            assert tracks == []
        elif tracks:
            ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_no_detections_and_no_tracks_gives_an_empty_list(tracker_cfg):
    frame, _ = scene({}, {})
    assert VehicleTracker(tracker_cfg).update([], frame) == []
