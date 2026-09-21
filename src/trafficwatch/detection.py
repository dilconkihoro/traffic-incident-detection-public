from __future__ import annotations

from trafficwatch.config import DetectorConfig
from trafficwatch.types import Detection, Prepared


class VehicleDetector:
    """Loads YOLOv8 weights, checks them against model_card.json, and returns detections."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg

    def detect(self, prepared: Prepared) -> list[Detection]:
        raise NotImplementedError("Phase 2")
