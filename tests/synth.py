"""Synthetic video helpers for tests: real mp4 files without any real footage.

The scenario generators, the renderer, and ScriptedDetector live in
trafficwatch.demo.scenarios (docs/05 Tier C) so tests and scripts/make_demo_clip.py share one
source of truth; this module re-exports them plus test-only helpers.
"""

from __future__ import annotations

import numpy as np

from trafficwatch.demo.scenarios import (  # noqa: F401
    BOX_W,
    ScriptedDetector,
    abrupt_stop_paths,
    render_scene,
    write_video,
)


def grey_ramp(n: int, height: int = 48, width: int = 64, step: int = 10) -> list[np.ndarray]:
    """Frame i is uniform grey level ``step * i`` so order can be read back after compression."""
    return [np.full((height, width, 3), step * i, dtype=np.uint8) for i in range(n)]


__all__ = [
    "BOX_W",
    "ScriptedDetector",
    "abrupt_stop_paths",
    "grey_ramp",
    "render_scene",
    "write_video",
]
