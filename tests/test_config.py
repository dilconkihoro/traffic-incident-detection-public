"""TC-UT-03: configuration validation fails closed."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from trafficwatch.config import (
    ConfigError,
    behaviour_hash,
    config_hash,
    load_config,
    parse_config,
)

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"
SECTIONS = ["run", "detector", "preprocess", "tracker", "behaviour"]


def shipped_data() -> dict:
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))


def leaf_paths(node: dict, prefix: str = "") -> list[str]:
    paths = []
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            paths.extend(leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


def set_path(data: dict, path: str, value) -> dict:
    data = copy.deepcopy(data)
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    node[leaf] = value
    return data


def delete_path(data: dict, path: str) -> dict:
    data = copy.deepcopy(data)
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    del node[leaf]
    return data


def test_shipped_config_is_valid():
    cfg = load_config(SHIPPED, env={})
    assert cfg.detector.imgsz == 640
    assert cfg.behaviour.v_stop < cfg.behaviour.v_move


BAD_VALUES = [
    ("run.seed", -1),
    ("run.seed", True),
    ("run.seed", "0"),
    ("run.frame_stride", 0),
    ("detector.weights", ""),
    ("detector.weights_kind", "pretrained"),
    ("detector.imgsz", 600),
    ("detector.imgsz", 0),
    ("detector.conf", 0),
    ("detector.conf", 1.0),
    ("detector.conf", "0.3"),
    ("detector.iou_nms", 0),
    ("detector.iou_nms", 1.5),
    ("detector.classes", []),
    ("detector.classes", ["car", "car"]),
    ("detector.classes", "car"),
    ("preprocess.low_light_enhance", "no"),
    ("preprocess.low_light_enhance", 1),
    ("tracker.max_age", 0),
    ("tracker.n_init", 0),
    ("tracker.max_iou_distance", 1.5),
    ("tracker.embedder", "resnet"),
    ("behaviour.speed_unit", "px_per_s"),
    ("behaviour.v_stop", -0.1),
    ("behaviour.v_stop", 1.0),
    ("behaviour.v_move", 0),
    ("behaviour.smoothing_window_s", 0),
    ("behaviour.abrupt_stop.window_before_s", 0),
    ("behaviour.abrupt_stop.drop_ratio", 0),
    ("behaviour.abrupt_stop.drop_ratio", 1.1),
    ("behaviour.lane_obstruction.min_flow_fraction", 1.1),
    ("behaviour.lane_obstruction.min_flow_fraction", -0.1),
    ("behaviour.collision.iou_min", 0),
    ("behaviour.collision.min_frames", 0),
    ("behaviour.collision.min_frames", 2.5),
    ("behaviour.collision.heading_change_deg", 0),
    ("behaviour.collision.heading_change_deg", 181),
    ("behaviour.suppression.global_slowdown_fraction", 0),
    ("behaviour.zones.driving_lanes", "lane"),
    ("behaviour.zones.driving_lanes", [[[0, 0], [1, 1]]]),
    ("behaviour.zones.driving_lanes", [[[0, 0], [1, 1], [2, "x"]]]),
    ("behaviour.zones.exclusion", [[[0, 0], [1, 1], [-2, 3]]]),
]


@pytest.mark.parametrize(("path", "value"), BAD_VALUES, ids=[f"{p}={v!r}" for p, v in BAD_VALUES])
def test_bad_value_is_rejected_and_named(path, value):
    with pytest.raises(ConfigError) as excinfo:
        parse_config(set_path(shipped_data(), path, value), env={})
    assert path in str(excinfo.value)


@pytest.mark.parametrize("path", leaf_paths(shipped_data()))
def test_missing_leaf_is_rejected(path):
    with pytest.raises(ConfigError, match=f"{path}: missing"):
        parse_config(delete_path(shipped_data(), path), env={})


@pytest.mark.parametrize("section", SECTIONS)
def test_missing_section_is_rejected(section):
    with pytest.raises(ConfigError, match=f"{section}: missing"):
        parse_config(delete_path(shipped_data(), section), env={})


@pytest.mark.parametrize("where", ["", "detector.", "behaviour.", "behaviour.collision."])
def test_unknown_key_is_rejected(where):
    data = shipped_data()
    target = data
    for key in [k for k in where.split(".") if k]:
        target = target[key]
    target["surprise"] = 1
    with pytest.raises(ConfigError, match="surprise: unknown key"):
        parse_config(data, env={})


def test_all_violations_are_reported_together():
    data = set_path(shipped_data(), "detector.conf", 2.0)
    data = set_path(data, "tracker.n_init", 0)
    with pytest.raises(ConfigError) as excinfo:
        parse_config(data, env={})
    assert "detector.conf" in str(excinfo.value)
    assert "tracker.n_init" in str(excinfo.value)


def test_standin_weights_are_refused_for_a_final_run():
    data = set_path(shipped_data(), "detector.weights_kind", "coco_standin")
    assert parse_config(data, env={}).detector.weights_kind == "coco_standin"
    with pytest.raises(ConfigError, match="detector.weights_kind"):
        parse_config(data, final=True, env={})


def test_tw_weights_overrides_and_empty_value_is_ignored():
    configured = shipped_data()["detector"]["weights"]
    assert parse_config(
        shipped_data(), env={"TW_WEIGHTS": "weights/R1/best.pt"}
    ).detector.weights == ("weights/R1/best.pt")
    assert parse_config(shipped_data(), env={"TW_WEIGHTS": ""}).detector.weights == configured


def test_valid_polygons_are_loaded():
    data = set_path(
        shipped_data(), "behaviour.zones.driving_lanes", [[[0, 0], [100, 0], [100, 50]]]
    )
    cfg = parse_config(data, env={})
    assert cfg.behaviour.zones.driving_lanes == (((0.0, 0.0), (100.0, 0.0), (100.0, 50.0)),)


def test_hashes_ignore_key_order_and_track_values():
    data = shipped_data()
    reordered = copy.deepcopy(data)
    reordered["behaviour"] = dict(reversed(list(data["behaviour"].items())))
    base = parse_config(data, env={})
    assert behaviour_hash(parse_config(reordered, env={}).behaviour) == behaviour_hash(
        base.behaviour
    )
    changed = parse_config(set_path(data, "behaviour.v_stop", 0.2), env={})
    assert behaviour_hash(changed.behaviour) != behaviour_hash(base.behaviour)
    detector_changed = parse_config(set_path(data, "detector.conf", 0.5), env={})
    assert config_hash(detector_changed) != config_hash(base)
    assert behaviour_hash(detector_changed.behaviour) == behaviour_hash(base.behaviour)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot read"),
        ("", "empty"),
        ("run: [unclosed", "not valid YAML"),
        ("- a\n- b\n", "expected a mapping"),
    ],
    ids=["missing_file", "empty_file", "bad_yaml", "list_root"],
)
def test_unreadable_or_malformed_file_is_rejected(tmp_path, content, message):
    path = tmp_path / "pipeline.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path, env={})
