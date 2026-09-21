"""configs/data.yaml: dataset conversion settings, validated fail closed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trafficwatch.config import ConfigError, _Node


@dataclass(frozen=True)
class DataConfig:
    seed: int
    stride: int
    val_sequences: int
    unified_classes: tuple[str, ...]
    mask_colour: tuple[int, ...]
    ignored_box_overlap: float
    phash_max_distance: int
    dhaka_fractions: Mapping[str, float]
    class_map: Mapping[str, Mapping[str, str | None]]


def parse_data_config(data: Any) -> DataConfig:
    errors: list[str] = []
    root = _Node(data, "", errors)
    seed = root.integer("seed", ge=0)
    stride = root.integer("stride", ge=1)
    val_sequences = root.integer("val_sequences", ge=1)
    classes = root.names("unified_classes")
    colour = root.int_list("mask_colour", length=3, ge=0, le=255)
    overlap = root.number("ignored_box_overlap", gt=0.0, le=1.0)
    dhaka = root.child("dhaka")
    distance = dhaka.integer("phash_max_distance", ge=0)
    if distance > 64:
        errors.append(f"dhaka.phash_max_distance: must be <= 64, got {distance}")
    fractions = dhaka.fractions("fractions")
    if fractions and "train" not in fractions:
        errors.append("dhaka.fractions: must include a 'train' split")
    dhaka.finish()
    class_map = root.class_maps("class_map", classes)
    root.finish()
    if errors:
        raise ConfigError("invalid data configuration:\n" + "\n".join(f"  - {e}" for e in errors))
    return DataConfig(
        seed, stride, val_sequences, classes, colour, overlap, distance, fractions, class_map
    )


def load_data_config(path: str | Path) -> DataConfig:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read data config {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"data config {path} is not valid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"data config {path} is empty")
    return parse_data_config(data)
