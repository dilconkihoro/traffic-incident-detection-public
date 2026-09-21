from __future__ import annotations

from trafficwatch.config import DetectorConfig, PreprocessConfig
from trafficwatch.types import Frame, Prepared


class Preprocessor:
    """Letterboxes a frame to the detector size, aspect preserved, without pixel normalisation."""

    def __init__(self, preprocess: PreprocessConfig, detector: DetectorConfig) -> None:
        self.preprocess = preprocess
        self.imgsz = detector.imgsz

    def apply(self, frame: Frame) -> Prepared:
        raise NotImplementedError("Phase 2")
