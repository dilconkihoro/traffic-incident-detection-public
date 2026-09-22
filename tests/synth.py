"""Synthetic video helpers: real mp4 files without any real footage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from trafficwatch.types import Detection


def write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a VideoWriter for mp4v")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def grey_ramp(n: int, height: int = 48, width: int = 64, step: int = 10) -> list[np.ndarray]:
    """Frame i is uniform grey level ``step * i`` so order can be read back after compression."""
    return [np.full((height, width, 3), step * i, dtype=np.uint8) for i in range(n)]


BOX_W, BOX_H = 60, 40


def render_scene(
    paths: dict[int, list[tuple[int, int]]], size: tuple[int, int] = (640, 480)
) -> tuple[list[np.ndarray], list[list[Detection]]]:
    """Textured 60x40 patches on a noise background; ``paths[id][f]`` is the patch (left, top).

    Returns the frames and, per frame, the exact ground-truth detections in image coordinates.
    """
    width, height = size
    background = np.random.default_rng(999).integers(60, 90, (height, width, 3), dtype=np.uint8)
    textures = {
        vid: np.random.default_rng(vid).integers(0, 256, (BOX_H, BOX_W, 3), dtype=np.uint8)
        for vid in paths
    }
    frames, detections = [], []
    for f in range(max(len(p) for p in paths.values())):
        image, here = background.copy(), []
        for vid, path in paths.items():
            if f < len(path):
                x, y = path[f]
                image[y : y + BOX_H, x : x + BOX_W] = textures[vid]
                here.append(Detection((x, y, x + BOX_W, y + BOX_H), 0.9, 0, "car"))
        frames.append(image)
        detections.append(here)
    return frames, detections


class ScriptedDetector:
    """Returns the ground-truth detections frame by frame; no model involved."""

    weights_kind = "coco_standin"
    weights_run_id = "scripted"
    weights_sha256 = "0" * 64

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self._next = 0

    def detect(self, prepared) -> list[Detection]:
        found = self._per_frame[self._next]
        self._next += 1
        return found


def abrupt_stop_paths(n_frames: int = 51, stop_frame: int = 30) -> dict[int, list[tuple[int, int]]]:
    """Subject 1 cruises at 2.0 body heights per second (8 px per frame) then stops dead;
    four fillers keep cruising in lanes at least 2.75 body heights away."""
    paths = {1: [(50 + 8 * min(f, stop_frame), 200) for f in range(n_frames)]}
    for vid, top in zip((2, 3, 4, 5), (20, 90, 310, 380), strict=True):
        paths[vid] = [(20 + 8 * f, top) for f in range(n_frames)]
    return paths
