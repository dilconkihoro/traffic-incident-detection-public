"""Pipeline configuration: typed, immutable, validated at startup (fail closed)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

Polygon = tuple[tuple[float, float], ...]

WEIGHTS_KINDS = ("trained", "coco_standin")
EMBEDDERS = ("mobilenet",)
SPEED_UNITS = ("body_heights_per_s",)


class ConfigError(ValueError):
    """Raised with every violation found, one per line, each naming its dotted key."""


@dataclass(frozen=True)
class RunConfig:
    seed: int
    frame_stride: int


@dataclass(frozen=True)
class DetectorConfig:
    weights: str
    weights_kind: str
    imgsz: int
    conf: float
    iou_nms: float
    classes: tuple[str, ...]
    standin_class_map: Mapping[int, str]


@dataclass(frozen=True)
class PreprocessConfig:
    low_light_enhance: bool
    clahe_clip_limit: float
    clahe_tile_grid: int


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int
    n_init: int
    max_iou_distance: float
    embedder: str


@dataclass(frozen=True)
class AbruptStopConfig:
    window_before_s: float
    window_after_s: float
    drop_ratio: float
    max_decel_s: float


@dataclass(frozen=True)
class LaneObstructionConfig:
    min_stationary_s: float
    min_flow_fraction: float
    rearm_s: float


@dataclass(frozen=True)
class CollisionConfig:
    v_close: float
    iou_min: float
    min_frames: int
    post_window_s: float
    heading_change_deg: float


@dataclass(frozen=True)
class SuppressionConfig:
    queue_gap_bh: float
    global_slowdown_fraction: float


@dataclass(frozen=True)
class ZonesConfig:
    driving_lanes: tuple[Polygon, ...]
    exclusion: tuple[Polygon, ...]


@dataclass(frozen=True)
class BehaviourConfig:
    speed_unit: str
    v_stop: float
    v_move: float
    smoothing_window_s: float
    abrupt_stop: AbruptStopConfig
    lane_obstruction: LaneObstructionConfig
    collision: CollisionConfig
    suppression: SuppressionConfig
    zones: ZonesConfig


@dataclass(frozen=True)
class OutputConfig:
    dir: str
    annotated_video: bool
    tracks_csv: bool


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig
    detector: DetectorConfig
    preprocess: PreprocessConfig
    tracker: TrackerConfig
    behaviour: BehaviourConfig
    output: OutputConfig


_MISSING = object()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Node:
    """Reads one mapping level, recording every violation instead of stopping at the first."""

    def __init__(self, data: Any, path: str, errors: list[str]) -> None:
        self._path = path
        self._errors = errors
        self._seen: set[str] = set()
        if isinstance(data, Mapping):
            self._data: Mapping[str, Any] = data
        else:
            errors.append(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
            self._data = {}

    def _name(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _fail(self, key: str, message: str) -> None:
        self._errors.append(f"{self._name(key)}: {message}")

    def _fetch(self, key: str) -> Any:
        self._seen.add(key)
        if key not in self._data:
            self._fail(key, "missing")
            return _MISSING
        return self._data[key]

    def child(self, key: str) -> _Node:
        value = self._fetch(key)
        return _Node({} if value is _MISSING else value, self._name(key), self._errors)

    def finish(self) -> None:
        for key in self._data:
            if key not in self._seen:
                self._fail(str(key), "unknown key")

    def integer(self, key: str, *, ge: int | None = None) -> int:
        value = self._fetch(key)
        if value is _MISSING:
            return 0
        if isinstance(value, bool) or not isinstance(value, int):
            self._fail(key, f"expected an integer, got {value!r}")
            return 0
        if ge is not None and value < ge:
            self._fail(key, f"must be >= {ge}, got {value}")
        return value

    def number(
        self,
        key: str,
        *,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
    ) -> float:
        value = self._fetch(key)
        if value is _MISSING:
            return 0.0
        if not _is_number(value):
            self._fail(key, f"expected a number, got {value!r}")
            return 0.0
        number = float(value)
        bounds = (
            (gt, number > gt if gt is not None else True, f"> {gt}"),
            (ge, number >= ge if ge is not None else True, f">= {ge}"),
            (lt, number < lt if lt is not None else True, f"< {lt}"),
            (le, number <= le if le is not None else True, f"<= {le}"),
        )
        for _, ok, text in bounds:
            if not ok:
                self._fail(key, f"must be {text}, got {number}")
        return number

    def boolean(self, key: str) -> bool:
        value = self._fetch(key)
        if value is _MISSING:
            return False
        if not isinstance(value, bool):
            self._fail(key, f"expected true or false, got {value!r}")
            return False
        return value

    def string(self, key: str, *, choices: tuple[str, ...] | None = None) -> str:
        value = self._fetch(key)
        if value is _MISSING:
            return ""
        if not isinstance(value, str) or not value.strip():
            self._fail(key, f"expected a non-empty string, got {value!r}")
            return ""
        if choices is not None and value not in choices:
            self._fail(key, f"must be one of {list(choices)}, got {value!r}")
        return value

    def names(self, key: str) -> tuple[str, ...]:
        value = self._fetch(key)
        if value is _MISSING:
            return ()
        if not isinstance(value, list) or not value:
            self._fail(key, "expected a non-empty list of names")
            return ()
        if not all(isinstance(v, str) and v.strip() for v in value):
            self._fail(key, "every name must be a non-empty string")
            return ()
        if len(set(value)) != len(value):
            self._fail(key, "names must be unique")
        return tuple(value)

    def int_list(self, key: str, *, length: int, ge: int, le: int) -> tuple[int, ...]:
        value = self._fetch(key)
        if value is _MISSING:
            return ()
        ok = isinstance(value, list) and len(value) == length
        if not ok or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            self._fail(key, f"expected a list of {length} integers, got {value!r}")
            return ()
        if not all(ge <= v <= le for v in value):
            self._fail(key, f"every value must be between {ge} and {le}, got {value}")
        return tuple(value)

    def fractions(self, key: str) -> dict[str, float]:
        """A non-empty mapping of names to positive numbers that sum to 1."""
        value = self._fetch(key)
        if value is _MISSING:
            return {}
        if not isinstance(value, Mapping) or not value:
            self._fail(key, "expected a non-empty mapping of name to fraction")
            return {}
        if not all(
            isinstance(k, str) and k.strip() and _is_number(v) and v > 0 for k, v in value.items()
        ):
            self._fail(
                key, f"every entry must map a name to a positive number, got {dict(value)!r}"
            )
            return {}
        if abs(sum(value.values()) - 1.0) > 1e-9:
            self._fail(key, f"fractions must sum to 1, got {sum(value.values())}")
        return {k: float(v) for k, v in value.items()}

    def class_maps(self, key: str, allowed: tuple[str, ...]) -> dict[str, dict[str, str | None]]:
        """Per dataset, source class to unified class (or null to drop it deliberately)."""
        value = self._fetch(key)
        if value is _MISSING:
            return {}
        if not isinstance(value, Mapping) or not value:
            self._fail(key, "expected a non-empty mapping of dataset to class map")
            return {}
        result: dict[str, dict[str, str | None]] = {}
        for dataset, mapping in value.items():
            where = f"{key}.{dataset}"
            if not isinstance(mapping, Mapping):
                self._fail(where, "expected a mapping of source class to unified class or null")
                continue
            for source, target in mapping.items():
                if target is not None and target not in allowed:
                    self._fail(
                        f"{where}.{source}",
                        f"must be null or one of {list(allowed)}, got {target!r}",
                    )
            result[str(dataset)] = {str(s): t for s, t in mapping.items()}
        return result

    def class_id_map(self, key: str, allowed: tuple[str, ...]) -> dict[int, str]:
        """Source class id to unified class name; every target must be a configured class."""
        value = self._fetch(key)
        if value is _MISSING:
            return {}
        if not isinstance(value, Mapping) or not value:
            self._fail(key, "expected a non-empty mapping of class id to class name")
            return {}
        result: dict[int, str] = {}
        for source, target in value.items():
            if isinstance(source, bool) or not isinstance(source, int) or source < 0:
                self._fail(f"{key}.{source}", "class id must be a non-negative integer")
            elif target not in allowed:
                self._fail(f"{key}.{source}", f"must be one of {list(allowed)}, got {target!r}")
            else:
                result[source] = target
        return result

    def polygons(self, key: str) -> tuple[Polygon, ...]:
        value = self._fetch(key)
        if value is _MISSING:
            return ()
        if not isinstance(value, list):
            self._fail(key, "expected a list of polygons")
            return ()
        result: list[Polygon] = []
        for i, polygon in enumerate(value):
            where = f"{key}[{i}]"
            if not isinstance(polygon, list) or len(polygon) < 3:
                self._fail(where, "a polygon needs at least 3 points")
                continue
            points: list[tuple[float, float]] = []
            for j, point in enumerate(polygon):
                if (
                    not isinstance(point, (list, tuple))
                    or len(point) != 2
                    or not all(_is_number(c) and c >= 0 for c in point)
                ):
                    self._fail(
                        f"{where}[{j}]", f"expected [x, y] with non-negative numbers, got {point!r}"
                    )
                    continue
                points.append((float(point[0]), float(point[1])))
            result.append(tuple(points))
        return tuple(result)


def _run(node: _Node) -> RunConfig:
    cfg = RunConfig(
        seed=node.integer("seed", ge=0), frame_stride=node.integer("frame_stride", ge=1)
    )
    node.finish()
    return cfg


def _detector(node: _Node, weights_override: str | None, final: bool) -> DetectorConfig:
    weights = node.string("weights")
    kind = node.string("weights_kind", choices=WEIGHTS_KINDS)
    imgsz = node.integer("imgsz", ge=32)
    if imgsz % 32 != 0:
        node._fail("imgsz", f"must be a multiple of 32, got {imgsz}")
    conf = node.number("conf", gt=0.0, lt=1.0)
    iou_nms = node.number("iou_nms", gt=0.0, le=1.0)
    classes = node.names("classes")
    cfg = DetectorConfig(
        weights=weights_override or weights,
        weights_kind=kind,
        imgsz=imgsz,
        conf=conf,
        iou_nms=iou_nms,
        classes=classes,
        standin_class_map=node.class_id_map("standin_class_map", classes),
    )
    if final and kind == "coco_standin":
        node._fail("weights_kind", "coco_standin weights are not allowed in a final run")
    node.finish()
    return cfg


def _output(node: _Node) -> OutputConfig:
    cfg = OutputConfig(
        dir=node.string("dir"),
        annotated_video=node.boolean("annotated_video"),
        tracks_csv=node.boolean("tracks_csv"),
    )
    node.finish()
    return cfg


def _preprocess(node: _Node) -> PreprocessConfig:
    cfg = PreprocessConfig(
        low_light_enhance=node.boolean("low_light_enhance"),
        clahe_clip_limit=node.number("clahe_clip_limit", gt=0.0),
        clahe_tile_grid=node.integer("clahe_tile_grid", ge=1),
    )
    node.finish()
    return cfg


def _tracker(node: _Node) -> TrackerConfig:
    cfg = TrackerConfig(
        max_age=node.integer("max_age", ge=1),
        n_init=node.integer("n_init", ge=1),
        max_iou_distance=node.number("max_iou_distance", gt=0.0, le=1.0),
        embedder=node.string("embedder", choices=EMBEDDERS),
    )
    node.finish()
    return cfg


def _abrupt_stop(node: _Node) -> AbruptStopConfig:
    cfg = AbruptStopConfig(
        window_before_s=node.number("window_before_s", gt=0.0),
        window_after_s=node.number("window_after_s", gt=0.0),
        drop_ratio=node.number("drop_ratio", gt=0.0, le=1.0),
        max_decel_s=node.number("max_decel_s", gt=0.0),
    )
    node.finish()
    return cfg


def _lane_obstruction(node: _Node) -> LaneObstructionConfig:
    cfg = LaneObstructionConfig(
        min_stationary_s=node.number("min_stationary_s", gt=0.0),
        min_flow_fraction=node.number("min_flow_fraction", ge=0.0, le=1.0),
        rearm_s=node.number("rearm_s", ge=0.0),
    )
    node.finish()
    return cfg


def _collision(node: _Node) -> CollisionConfig:
    cfg = CollisionConfig(
        v_close=node.number("v_close", gt=0.0),
        iou_min=node.number("iou_min", gt=0.0, le=1.0),
        min_frames=node.integer("min_frames", ge=1),
        post_window_s=node.number("post_window_s", gt=0.0),
        heading_change_deg=node.number("heading_change_deg", gt=0.0, le=180.0),
    )
    node.finish()
    return cfg


def _suppression(node: _Node) -> SuppressionConfig:
    cfg = SuppressionConfig(
        queue_gap_bh=node.number("queue_gap_bh", gt=0.0),
        global_slowdown_fraction=node.number("global_slowdown_fraction", gt=0.0, le=1.0),
    )
    node.finish()
    return cfg


def _zones(node: _Node) -> ZonesConfig:
    cfg = ZonesConfig(
        driving_lanes=node.polygons("driving_lanes"),
        exclusion=node.polygons("exclusion"),
    )
    node.finish()
    return cfg


def _behaviour(node: _Node) -> BehaviourConfig:
    speed_unit = node.string("speed_unit", choices=SPEED_UNITS)
    v_stop = node.number("v_stop", ge=0.0)
    v_move = node.number("v_move", gt=0.0)
    if v_stop >= v_move:
        node._fail("v_stop", f"must be below v_move ({v_move}), got {v_stop}")
    cfg = BehaviourConfig(
        speed_unit=speed_unit,
        v_stop=v_stop,
        v_move=v_move,
        smoothing_window_s=node.number("smoothing_window_s", gt=0.0),
        abrupt_stop=_abrupt_stop(node.child("abrupt_stop")),
        lane_obstruction=_lane_obstruction(node.child("lane_obstruction")),
        collision=_collision(node.child("collision")),
        suppression=_suppression(node.child("suppression")),
        zones=_zones(node.child("zones")),
    )
    node.finish()
    return cfg


def parse_config(
    data: Any, *, final: bool = False, env: Mapping[str, str] | None = None
) -> PipelineConfig:
    """Validate a raw mapping and return the typed config, or raise ConfigError listing all faults.

    ``final`` marks a run whose results may enter result tables; stand-in weights are refused.
    ``TW_WEIGHTS`` in ``env`` (default: the process environment) overrides ``detector.weights``.
    """
    env = os.environ if env is None else env
    errors: list[str] = []
    root = _Node(data, "", errors)
    cfg = PipelineConfig(
        run=_run(root.child("run")),
        detector=_detector(root.child("detector"), env.get("TW_WEIGHTS") or None, final),
        preprocess=_preprocess(root.child("preprocess")),
        tracker=_tracker(root.child("tracker")),
        behaviour=_behaviour(root.child("behaviour")),
        output=_output(root.child("output")),
    )
    root.finish()
    if errors:
        raise ConfigError("invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))
    return cfg


STANDIN_WEIGHTS = "weights/coco_standin/yolov8n.pt"


def _with_standin(data: Any) -> Any:
    if not isinstance(data, Mapping) or not isinstance(data.get("detector"), Mapping):
        return data
    detector = {**data["detector"], "weights": STANDIN_WEIGHTS, "weights_kind": "coco_standin"}
    return {**data, "detector": detector}


def load_config(
    path: str | Path,
    *,
    final: bool = False,
    env: Mapping[str, str] | None = None,
    standin: bool = False,
) -> PipelineConfig:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config {path} is not valid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"config {path} is empty")
    if standin:
        # Standin mode's whole point is to force the general-purpose model regardless of
        # anything else configured; a stray TW_WEIGHTS in the environment (even a leftover
        # .env.example placeholder nobody edited) must never be able to override that silently.
        env = {}
    return parse_config(_with_standin(data) if standin else data, final=final, env=env)


def _hash(obj: Any) -> str:
    canonical = json.dumps(asdict(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def behaviour_hash(cfg: BehaviourConfig) -> str:
    """Value for the ``rule_params_hash`` column of incidents.csv."""
    return _hash(cfg)


def config_hash(cfg: PipelineConfig) -> str:
    return _hash(cfg)
