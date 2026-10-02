"""configs/data.yaml validation fails closed."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from trafficwatch.config import ConfigError
from trafficwatch.data.config import load_data_config, parse_data_config

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


def shipped() -> dict:
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))


def with_value(path: str, value) -> dict:
    data = copy.deepcopy(shipped())
    *parents, leaf = path.split(".")
    node = data
    for key in parents:
        node = node[key]
    node[leaf] = value
    return data


def test_shipped_data_config_is_valid():
    cfg = load_data_config(SHIPPED)
    assert cfg.stride == 5
    assert cfg.unified_classes == ("car", "van", "bus", "other")
    assert cfg.class_map["ua-detrac"]["others"] == "other"
    assert cfg.class_map["dhakaai"]["car"] == "car"
    assert cfg.class_map["dhakaai"]["ambulance"] == "other"
    # Two- and three-wheelers are dropped (mapped to null), not folded into `other` (D-04).
    assert cfg.class_map["dhakaai"]["rickshaw"] is None
    assert cfg.class_map["dhakaai"]["motorbike"] is None


BAD = [
    ("seed", -1),
    ("stride", 0),
    ("val_sequences", 0),
    ("unified_classes", []),
    ("mask_colour", [114, 114]),
    ("mask_colour", [114, 114, 300]),
    ("mask_colour", [114, 114, 1.5]),
    ("ignored_box_overlap", 0),
    ("ignored_box_overlap", 1.5),
    ("dhaka.phash_max_distance", 65),
    ("dhaka.phash_max_distance", -1),
    ("dhaka.fractions", {"train": 0.8, "dhaka_val": 0.3}),
    ("dhaka.fractions", {"dhaka_val": 0.5, "dhaka_test": 0.5}),
    ("dhaka.fractions", {"train": 1.5, "dhaka_val": -0.5}),
    ("class_map", {}),
    ("class_map.ua-detrac", {"car": "sedan"}),
    ("class_map.ua-detrac", ["car"]),
]


@pytest.mark.parametrize(("path", "value"), BAD, ids=[f"{p}={v!r}" for p, v in BAD])
def test_bad_value_is_rejected(path, value):
    with pytest.raises(ConfigError):
        parse_data_config(with_value(path, value))


def test_class_map_target_must_be_in_unified_classes_or_null():
    parsed = parse_data_config(with_value("class_map.dhakaai", {"bicycle": None, "bus": "bus"}))
    assert parsed.class_map["dhakaai"]["bicycle"] is None


def test_missing_and_unknown_keys_are_rejected():
    data = shipped()
    del data["stride"]
    with pytest.raises(ConfigError, match="stride: missing"):
        parse_data_config(data)
    with pytest.raises(ConfigError, match="surprise: unknown key"):
        parse_data_config({**shipped(), "surprise": 1})


def test_unreadable_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_data_config(tmp_path / "nope.yaml")
