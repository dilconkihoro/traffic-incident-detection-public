"""Tier C scenario generators (docs/05 Tier C, decision D-03): geometry sanity, and that each
scenario actually produces the incident type it claims when run through the real pipeline."""

from __future__ import annotations

import dataclasses

import pytest

from synth import ScriptedDetector, render_scene, write_video
from trafficwatch.behaviour.kinematics import box_iou
from trafficwatch.config import load_config
from trafficwatch.demo.scenarios import BOX_H, BOX_W, SCENARIOS
from trafficwatch.pipeline import Pipeline
from trafficwatch.types import IncidentType


def cfg_for(tmp_path):
    cfg = load_config("configs/pipeline.yaml", standin=True)
    return dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, dir=str(tmp_path)))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_path_stays_inside_its_own_canvas(name):
    scenario = SCENARIOS[name]
    width, height = scenario.size
    for path in scenario.paths.values():
        for x, y in path:
            assert 0 <= x <= width - BOX_W, (name, x)
            assert 0 <= y <= height - BOX_H, (name, y)


def test_collision_paths_actually_overlap_at_the_contact_frame():
    from trafficwatch.demo.scenarios import collision_paths

    paths = collision_paths(n_approach=40, n_hold=10, gap_at_contact=20)
    x1, y1 = paths[1][40]
    x2, y2 = paths[2][40]
    box1, box2 = (x1, y1, x1 + BOX_W, y1 + BOX_H), (x2, y2, x2 + BOX_W, y2 + BOX_H)
    assert box_iou(box1, box2) > 0.10  # comfortably clears the real iou_min


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("abrupt_stop", {IncidentType.ABRUPT_STOP}),
        ("free_flow", set()),
        ("lane_obstruction", {IncidentType.ABRUPT_STOP, IncidentType.LANE_OBSTRUCTION}),
        ("collision", {IncidentType.COLLISION_LIKE}),
    ],
)
def test_each_scenario_produces_its_claimed_incidents_end_to_end(tmp_path, name, expected):
    scenario = SCENARIOS[name]
    frames, detections = render_scene(scenario.paths, scenario.size)
    clip = write_video(tmp_path / f"{name}.mp4", frames, scenario.fps)
    result = Pipeline(cfg_for(tmp_path), ScriptedDetector(detections)).run(clip)
    assert {i.type for i in result.incidents} == expected
