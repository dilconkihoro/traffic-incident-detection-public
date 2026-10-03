from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from trafficwatch.config import DetectorConfig
from trafficwatch.data.manifest import sha256_file
from trafficwatch.types import Detection, Prepared

STANDIN_RUN_ID = "coco_standin"


class WeightsError(ValueError):
    """The weights do not satisfy the weights contract (project rules section 6)."""


def check_weights_contract(cfg: DetectorConfig, ultralytics_version: str) -> dict[str, Any] | None:
    """Refuse weights whose model card disagrees with the config; stand-in weights have no card."""
    weights = Path(cfg.weights)
    problems: list[str] = []
    if not weights.is_file():
        problems.append(f"weights file not found: {weights}")
    card: dict[str, Any] | None = None
    if cfg.weights_kind == "trained":
        card_path = weights.parent / "model_card.json"
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"model_card.json unreadable next to the weights: {exc}")
        if card is not None:
            if card.get("class_names") != list(cfg.classes):
                problems.append(
                    f"model card class_names {card.get('class_names')} differ from "
                    f"config detector.classes {list(cfg.classes)}"
                )
            if card.get("imgsz") != cfg.imgsz:
                problems.append(
                    f"model card imgsz {card.get('imgsz')} differs from config {cfg.imgsz}"
                )
            if card.get("ultralytics") != ultralytics_version:
                problems.append(
                    f"model card ultralytics {card.get('ultralytics')} differs from installed "
                    f"{ultralytics_version}"
                )
            if not card.get("run_id"):
                problems.append("model card has no run_id")
    if problems:
        raise WeightsError("weights contract violated:\n" + "\n".join(f"  - {p}" for p in problems))
    return card


class VehicleDetector:
    """Loads YOLOv8 weights, checks them against model_card.json, and returns detections.

    ``model`` lets tests supply a stand-in object with a YOLO-like ``predict``; the weights
    contract is still enforced.
    """

    def __init__(self, cfg: DetectorConfig, model: object | None = None) -> None:
        self.cfg = cfg
        card = check_weights_contract(cfg, importlib.metadata.version("ultralytics"))
        self.weights_kind = cfg.weights_kind
        self.weights_run_id = card["run_id"] if card else STANDIN_RUN_ID
        self.weights_sha256 = sha256_file(cfg.weights)
        self._coco_ids = (
            sorted(cfg.standin_class_map) if cfg.weights_kind == "coco_standin" else None
        )
        if model is None:
            from ultralytics import YOLO

            model = YOLO(cfg.weights)
        self._model: Any = model
        if card is not None:
            names = list(getattr(self._model, "names", {}).values())
            if names and names != list(cfg.classes):
                raise WeightsError(f"model class names {names} differ from config {cfg.classes}")

    def detect(self, prepared: Prepared) -> list[Detection]:
        results = self._model.predict(
            prepared.image640,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou_nms,
            classes=self._coco_ids,
            verbose=False,
        )
        boxes = results[0].boxes
        detections: list[Detection] = []
        for xyxy, conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
        ):
            class_id = self._unified_class_id(int(cls))
            box = prepared.to_original(tuple(xyxy))
            if class_id is None or box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(Detection(box, float(conf), class_id, self.cfg.classes[class_id]))
        return detections

    def _unified_class_id(self, model_class: int) -> int | None:
        if self._coco_ids is None:
            return model_class if 0 <= model_class < len(self.cfg.classes) else None
        name = self.cfg.standin_class_map.get(model_class)
        return None if name is None else self.cfg.classes.index(name)
