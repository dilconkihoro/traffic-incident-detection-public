from __future__ import annotations

import cv2

from trafficwatch.config import DetectorConfig, PreprocessConfig
from trafficwatch.types import Frame, Prepared

# Same grey Ultralytics pads with at training time, so inference sees what training saw (D-07).
LETTERBOX_PAD_VALUE = 114


class Preprocessor:
    """Letterboxes a frame to the detector size, aspect preserved, without pixel normalisation."""

    def __init__(self, preprocess: PreprocessConfig, detector: DetectorConfig) -> None:
        self.preprocess = preprocess
        self.imgsz = detector.imgsz
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=preprocess.clahe_clip_limit,
                tileGridSize=(preprocess.clahe_tile_grid, preprocess.clahe_tile_grid),
            )
            if preprocess.low_light_enhance
            else None
        )

    def apply(self, frame: Frame) -> Prepared:
        image = frame.image
        height, width = image.shape[:2]
        if self._clahe is not None:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        scale = min(self.imgsz / height, self.imgsz / width)
        new_w, new_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Ultralytics rounds the half-pad the same way, so both sides agree on the offsets.
        left = round((self.imgsz - new_w) / 2 - 0.1)
        top = round((self.imgsz - new_h) / 2 - 0.1)
        canvas = cv2.copyMakeBorder(
            resized,
            top,
            self.imgsz - new_h - top,
            left,
            self.imgsz - new_w - left,
            cv2.BORDER_CONSTANT,
            value=(LETTERBOX_PAD_VALUE,) * 3,
        )
        return Prepared(canvas, scale, (float(left), float(top)), (width, height))
