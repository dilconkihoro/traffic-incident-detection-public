"""Weights contract (TC-IT-03) and detector output mapping, with a scripted stand-in model."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trafficwatch.config import load_config
from trafficwatch.detection import VehicleDetector, WeightsError, check_weights_contract
from trafficwatch.preprocessing import Preprocessor
from trafficwatch.types import Frame

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "configs" / "pipeline.yaml"
ULTRALYTICS = importlib.metadata.version("ultralytics")
CLASSES = ["car", "van", "bus", "other"]


def base_cfg():
    return load_config(SHIPPED, env={})


def write_run(tmp_path, **card_changes):
    run_dir = tmp_path / "weights" / "R1"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"weights")
    card = {"run_id": "R1", "class_names": CLASSES, "imgsz": 640, "ultralytics": ULTRALYTICS}
    card.update(card_changes)
    (run_dir / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    return run_dir / "best.pt"


def trained_cfg(weights):
    return dataclasses.replace(base_cfg().detector, weights=str(weights), weights_kind="trained")


def standin_cfg(tmp_path):
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"coco")
    return dataclasses.replace(
        base_cfg().detector, weights=str(weights), weights_kind="coco_standin"
    )


def test_matching_card_is_accepted_and_returned(tmp_path):
    card = check_weights_contract(trained_cfg(write_run(tmp_path)), ULTRALYTICS)
    assert card is not None and card["run_id"] == "R1"


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"class_names": ["car", "bus"]}, "class_names"),
        ({"imgsz": 960}, "imgsz"),
        ({"ultralytics": "0.0.1"}, "ultralytics"),
        ({"run_id": ""}, "run_id"),
    ],
    ids=["class_names", "imgsz", "ultralytics_version", "no_run_id"],
)
def test_model_card_mismatch_aborts_startup(tmp_path, change, fragment):
    with pytest.raises(WeightsError, match=fragment):
        VehicleDetector(trained_cfg(write_run(tmp_path, **change)), model=object())


def test_missing_card_and_missing_weights_abort_startup(tmp_path):
    weights = write_run(tmp_path)
    (weights.parent / "model_card.json").unlink()
    with pytest.raises(WeightsError, match="model_card.json"):
        VehicleDetector(trained_cfg(weights), model=object())
    with pytest.raises(WeightsError, match="weights file not found"):
        VehicleDetector(trained_cfg(tmp_path / "nope" / "best.pt"), model=object())


def test_every_problem_is_reported_together(tmp_path):
    weights = write_run(tmp_path, imgsz=320, ultralytics="0.0.1")
    with pytest.raises(WeightsError) as excinfo:
        check_weights_contract(trained_cfg(weights), ULTRALYTICS)
    assert "imgsz" in str(excinfo.value) and "ultralytics" in str(excinfo.value)


def test_model_names_that_differ_from_config_abort_startup(tmp_path):
    model = SimpleNamespace(names={0: "car", 1: "truck"})
    with pytest.raises(WeightsError, match="model class names"):
        VehicleDetector(trained_cfg(write_run(tmp_path)), model=model)


class FakeModel:
    names = dict(enumerate(CLASSES))

    def __init__(self, xyxy, conf, cls):
        self.calls = []
        self._result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=np.array(xyxy), conf=np.array(conf), cls=np.array(cls))
        )

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self._result]


def prepared_1080p():
    frame = Frame(0, 0.0, np.zeros((1080, 1920, 3), dtype=np.uint8))
    cfg = base_cfg()
    return Preprocessor(cfg.preprocess, cfg.detector).apply(frame)


def letterbox(box):
    # 1080p frame: scale 1/3, top pad 140.
    return [box[0] / 3, box[1] / 3 + 140, box[2] / 3, box[3] / 3 + 140]


def test_standin_maps_coco_classes_and_returns_original_coordinates(tmp_path):
    model = FakeModel([letterbox((192, 108, 576, 432))], [0.9], [2])
    detector = VehicleDetector(standin_cfg(tmp_path), model=model)
    (detection,) = detector.detect(prepared_1080p())
    assert detection.cls_name == "car" and detection.cls_id == 0
    assert detection.conf == pytest.approx(0.9)
    assert detection.xyxy == pytest.approx((192, 108, 576, 432), abs=1.0)
    assert (detector.weights_kind, detector.weights_run_id) == ("coco_standin", "coco_standin")
    assert len(detector.weights_sha256) == 64


def test_standin_passes_config_thresholds_and_filters_coco_classes(tmp_path):
    model = FakeModel([letterbox((0, 0, 300, 300))], [0.8], [7])
    VehicleDetector(standin_cfg(tmp_path), model=model).detect(prepared_1080p())
    assert model.calls == [
        {"imgsz": 640, "conf": 0.35, "iou": 0.6, "classes": [2, 5, 7], "verbose": False}
    ]


def test_standin_maps_every_configured_coco_class(tmp_path):
    coords = [letterbox((100 * i, 100, 100 * i + 80, 200)) for i in range(1, 4)]
    model = FakeModel(coords, [0.9] * 3, [2, 5, 7])
    names = [
        d.cls_name
        for d in VehicleDetector(standin_cfg(tmp_path), model=model).detect(prepared_1080p())
    ]
    assert names == ["car", "bus", "other"]


def test_unmapped_classes_and_degenerate_boxes_are_dropped(tmp_path):
    coords = [letterbox((100, 100, 300, 300)), letterbox((5000, 100, 5200, 300))]
    model = FakeModel(coords, [0.9, 0.9], [0, 2])
    assert VehicleDetector(standin_cfg(tmp_path), model=model).detect(prepared_1080p()) == []


def test_no_detections_gives_an_empty_list(tmp_path):
    model = FakeModel(np.zeros((0, 4)), [], [])
    assert VehicleDetector(standin_cfg(tmp_path), model=model).detect(prepared_1080p()) == []


def test_trained_weights_use_class_ids_directly_and_carry_the_run_id(tmp_path):
    model = FakeModel([letterbox((192, 108, 576, 432))], [0.7], [3])
    detector = VehicleDetector(trained_cfg(write_run(tmp_path)), model=model)
    (detection,) = detector.detect(prepared_1080p())
    assert (detection.cls_id, detection.cls_name) == (3, "other")
    assert detector.weights_run_id == "R1"
    assert model.calls[0]["classes"] is None


@pytest.mark.standin
def test_real_standin_model_loads_and_returns_boxes_inside_the_frame():
    weights = ROOT / "weights" / "coco_standin" / "yolov8n.pt"
    if not weights.is_file():
        pytest.skip("run scripts/fetch_standin_weights.py first")
    cfg = dataclasses.replace(
        base_cfg().detector, weights=str(weights), weights_kind="coco_standin"
    )
    frame = Frame(0, 0.0, np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8))
    prepared = Preprocessor(base_cfg().preprocess, cfg).apply(frame)
    for detection in VehicleDetector(cfg).detect(prepared):
        x1, y1, x2, y2 = detection.xyxy
        assert 0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
        assert detection.cls_name in CLASSES
