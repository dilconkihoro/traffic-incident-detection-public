"""Pascal VOC single-image annotation parsing.

This is DhakaAI's real annotation format, confirmed 2026-09-22 by extracting and inspecting
train.rar directly (docs/03 section 2): one XML file per image, one <object> per box, absolute
pixel <bndbox>. At least one file in the real archive (231.xml) is not actually XML, it holds
YOLO-format label lines saved under a .xml name; parse_directory reports that as a failure
alongside the file, rather than letting it crash the whole batch or silently dropping it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

XYXY = tuple[float, float, float, float]


class VocParseError(ValueError):
    """The file is not a usable Pascal VOC annotation."""


@dataclass(frozen=True)
class VocObject:
    name: str
    bndbox: XYXY  # xmin, ymin, xmax, ymax, absolute pixels
    truncated: bool
    difficult: bool


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...]


def parse_voc_annotation(path: str | Path) -> VocAnnotation:
    """Parse one Pascal VOC <annotation> XML file; raises VocParseError on anything else."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise VocParseError(f"{path}: not well-formed XML: {exc}") from exc
    if root.tag != "annotation":
        raise VocParseError(f"{path}: root element is {root.tag!r}, expected 'annotation'")

    size = root.find("size")
    if size is None:
        raise VocParseError(f"{path}: missing <size>")
    width, height = _int(size, "width", path), _int(size, "height", path)

    filename = (root.findtext("filename") or "").strip()
    if not filename:
        raise VocParseError(f"{path}: missing or empty <filename>")

    objects = tuple(_object(node, path) for node in root.findall("object"))
    return VocAnnotation(filename, width, height, objects)


def parse_directory(dir_path: str | Path) -> tuple[list[VocAnnotation], list[tuple[Path, str]]]:
    """Parse every ``.xml`` in a directory; returns (successes, [(path, reason), ...] failures)."""
    ok: list[VocAnnotation] = []
    failed: list[tuple[Path, str]] = []
    for path in sorted(Path(dir_path).glob("*.xml")):
        try:
            ok.append(parse_voc_annotation(path))
        except VocParseError as exc:
            failed.append((path, str(exc)))
    return ok, failed


def _int(node: ET.Element, tag: str, path: Path) -> int:
    text = node.findtext(tag)
    if text is None or not text.strip():
        raise VocParseError(f"{path}: missing <{tag}>")
    try:
        return int(text)
    except ValueError as exc:
        raise VocParseError(f"{path}: <{tag}> is not an integer: {text!r}") from exc


def _float(node: ET.Element, tag: str, path: Path) -> float:
    text = node.findtext(tag)
    if text is None or not text.strip():
        raise VocParseError(f"{path}: <bndbox> missing <{tag}>")
    try:
        return float(text)
    except ValueError as exc:
        raise VocParseError(f"{path}: <{tag}> is not a number: {text!r}") from exc


def _object(node: ET.Element, path: Path) -> VocObject:
    name = (node.findtext("name") or "").strip()
    if not name:
        raise VocParseError(f"{path}: an <object> has no <name>")
    box = node.find("bndbox")
    if box is None:
        raise VocParseError(f"{path}: object {name!r} has no <bndbox>")
    xmin, ymin, xmax, ymax = (_float(box, tag, path) for tag in ("xmin", "ymin", "xmax", "ymax"))
    if xmax <= xmin or ymax <= ymin:
        raise VocParseError(
            f"{path}: object {name!r} has a degenerate box ({xmin}, {ymin}, {xmax}, {ymax})"
        )
    truncated = (node.findtext("truncated") or "0").strip() == "1"
    difficult = (node.findtext("difficult") or "0").strip() == "1"
    return VocObject(name, (xmin, ymin, xmax, ymax), truncated, difficult)
