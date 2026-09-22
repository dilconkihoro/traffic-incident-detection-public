"""Tier C synthetic scenarios (decision D-03, docs/05 section 1): hand-built trajectories with
known ground truth, used because neither UA-DETRAC nor DhakaAI has real incident labels.

These clips use ``ScriptedDetector``, which returns the exact ground-truth boxes instead of
running the real YOLOv8 model. That proves the tracker, the rule engine, and the output files
are correct; it says nothing about detection accuracy, which needs a trained model on real
footage. Every artifact these scenarios produce must say so and must never enter a result table
(same rule as ``coco_standin`` weights).

All timings use the real thresholds in ``configs/pipeline.yaml``, not shortened test versions, so
a scenario clip genuinely exercises the system as configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from trafficwatch.types import Detection

BOX_W, BOX_H = 60, 40
Paths = dict[int, list[tuple[int, int]]]


def write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a VideoWriter for mp4v")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def render_scene(
    paths: Paths, size: tuple[int, int] = (640, 480)
) -> tuple[list[np.ndarray], list[list[Detection]]]:
    """Textured 60x40 patches on a noise background; ``paths[id][f]`` is the patch (left, top).

    Each vehicle gets its own random texture so DeepSORT's appearance embedder can tell them
    apart, the same way it would use real vehicle appearance. Returns the frames and, per frame,
    the exact ground-truth detections in image coordinates.
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
    """Returns the ground-truth detections frame by frame; no model involved (see module note)."""

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


def abrupt_stop_paths(n_frames: int = 51, stop_frame: int = 30) -> Paths:
    """Subject 1 cruises at 2.0 body heights per second (8 px per frame) then stops dead;
    four fillers keep cruising in lanes at least 2.75 body heights away. Expect one
    ABRUPT_STOP, at frame 39 with the default config (proven in tests/test_pipeline.py)."""
    paths = {1: [(50 + 8 * min(f, stop_frame), 200) for f in range(n_frames)]}
    for vid, top in zip((2, 3, 4, 5), (20, 90, 310, 380), strict=True):
        paths[vid] = [(20 + 8 * f, top) for f in range(n_frames)]
    return paths


def free_flow_paths(n_frames: int = 60) -> Paths:
    """Five vehicles cruise steadily in separate lanes; none stop, none converge. Expect zero
    flags: this is the clean-negative demo, as important to show as a positive one."""
    lanes = (20, 90, 200, 310, 380)
    return {vid: [(20 + 8 * f, top) for f in range(n_frames)] for vid, top in enumerate(lanes, 1)}


def lane_obstruction_paths(n_hold: int = 160) -> Paths:
    """Subject 1 stops for n_hold frames, comfortably past the real min_stationary_s (15.0 s at
    10 fps = 150 frames); three fillers keep moving throughout so flow continues around it.
    A wide canvas is needed: fillers travel a long way over the full hold. This scenario's
    sudden stop also satisfies ABRUPT_STOP, which is realistic (a stall often is a sudden stop
    that then persists), not a bug: expect both incident types."""
    approach = 20
    paths = {1: [(50 + 8 * min(f, approach), 200) for f in range(approach + n_hold)]}
    for vid, top in zip((2, 3, 4), (20, 90, 380), strict=True):
        paths[vid] = [(20 + 5 * f, top) for f in range(approach + n_hold)]
    return paths


def collision_paths(n_approach: int = 40, n_hold: int = 30, gap_at_contact: int = 20) -> Paths:
    """Two vehicles close head-on at 2.0 body heights per second and both stop at contact.

    ``gap_at_contact`` is the left-edge gap in pixels once approach ends: with BOX_W = 60, a gap
    of 20 px gives an overlap of 40 px and IoU 0.5, comfortably clear of iou_min (0.10). The start
    gap is derived from it so the two never depend on a hand-copied constant from a different box
    size. Expect one COLLISION_LIKE."""
    closing_px_per_frame = 10
    start_gap = 2 * closing_px_per_frame * n_approach + gap_at_contact
    return {
        1: [(closing_px_per_frame * min(f, n_approach), 200) for f in range(n_approach + n_hold)],
        2: [
            (start_gap - closing_px_per_frame * min(f, n_approach), 200)
            for f in range(n_approach + n_hold)
        ],
    }


@dataclass(frozen=True)
class Scenario:
    paths: Paths
    size: tuple[int, int]
    fps: float
    description: str


SCENARIOS: dict[str, Scenario] = {
    "abrupt_stop": Scenario(abrupt_stop_paths(), (640, 480), 10.0, "one vehicle stops suddenly"),
    "free_flow": Scenario(free_flow_paths(), (640, 480), 10.0, "steady traffic, no incidents"),
    "lane_obstruction": Scenario(
        lane_obstruction_paths(), (1024, 480), 10.0, "a stalled vehicle blocks a lane"
    ),
    "collision": Scenario(collision_paths(), (960, 480), 10.0, "two vehicles converge and stop"),
}
