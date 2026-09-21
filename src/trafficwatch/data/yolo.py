"""Dataset-agnostic helpers for the YOLO conversion (labels, class map, stride, masking)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

XYXY = tuple[float, float, float, float]


class UnmappedClassError(KeyError):
    """A source class is neither mapped nor explicitly dropped."""


def map_class(source: str, mapping: Mapping[str, str | None], dataset: str) -> str | None:
    """Return the unified class, or None when the class is explicitly dropped (mapped to null)."""
    if source not in mapping:
        raise UnmappedClassError(f"{dataset}: source class {source!r} is not in the class map")
    return mapping[source]


def strided(frame_indices: Iterable[int], stride: int) -> list[int]:
    """Every ``stride``-th frame counting from the first, whatever the first index is."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return sorted(frame_indices)[::stride]


def to_yolo_line(class_index: int, box: XYXY, width: int, height: int) -> str | None:
    """Clip a pixel box to the image and format it as normalised ``class xc yc w h``.

    Returns None when nothing is left of the box after clipping.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    x1, y1 = max(0.0, box[0]), max(0.0, box[1])
    x2, y2 = min(float(width), box[2]), min(float(height), box[3])
    if x2 <= x1 or y2 <= y1:
        return None
    xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{class_index} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _region_mask(shape: tuple[int, int], regions: Sequence[XYXY]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in regions:
        mask[
            max(0, int(round(y1))) : max(0, int(round(y2))),
            max(0, int(round(x1))) : max(0, int(round(x2))),
        ] = True
    return mask


def mask_regions(
    image: np.ndarray, regions: Sequence[XYXY], colour: tuple[int, int, int]
) -> tuple[np.ndarray, float]:
    """Fill ignored regions with a constant colour; return the copy and the masked pixel share."""
    mask = _region_mask(image.shape[:2], regions)
    out = image.copy()
    out[mask] = colour
    return out, float(mask.mean())


def overlap_fraction(box: XYXY, regions: Sequence[XYXY], shape: tuple[int, int]) -> float:
    """Share of the box area that lies inside the union of ignored regions."""
    mask = _region_mask(shape, regions)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area == 0:
        return 0.0
    inside = mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)].sum()
    return float(inside) / area
