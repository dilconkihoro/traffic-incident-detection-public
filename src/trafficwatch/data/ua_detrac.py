"""UA-DETRAC per-sequence annotation parsing.

Real format, confirmed by extracting and inspecting the train and test annotation archives
directly (docs/03 section 2): one XML file per sequence, `<sequence name="...">`, a single
`<sequence_attribute camera_state="..." sence_weather="..."/>` (the source dataset's own
misspelling, not ours), zero or more `<ignored_region><box .../></ignored_region>`, then one
`<frame num="N"><target_list><target id="K"><box left top width height/>` per frame, `id` the
persistent track ID within the sequence. Box format is left/top/width/height, unlike DhakaAI's
xmin/ymin/xmax/ymax; converted to xyxy at parse time so every downstream consumer (yolo.py's
to_yolo_line, mask_regions, overlap_fraction) works in one convention regardless of source.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

XYXY = tuple[float, float, float, float]


class UaDetracParseError(ValueError):
    """The file is not a usable UA-DETRAC sequence annotation."""


@dataclass(frozen=True)
class UaDetracTarget:
    track_id: int
    xyxy: XYXY
    vehicle_type: str
    truncation_ratio: float


@dataclass(frozen=True)
class UaDetracFrame:
    num: int
    targets: tuple[UaDetracTarget, ...]


@dataclass(frozen=True)
class UaDetracSequence:
    name: str
    camera_state: str
    weather: str
    ignored_regions: tuple[XYXY, ...]
    frames: tuple[UaDetracFrame, ...]


def parse_sequence(path: str | Path) -> UaDetracSequence:
    """Parse one `<sequence>` XML file; raises UaDetracParseError on anything else."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise UaDetracParseError(f"{path}: not well-formed XML: {exc}") from exc
    if root.tag != "sequence":
        raise UaDetracParseError(f"{path}: root element is {root.tag!r}, expected 'sequence'")

    name = (root.get("name") or "").strip()
    if not name:
        raise UaDetracParseError(f"{path}: <sequence> has no name attribute")

    attr = root.find("sequence_attribute")
    if attr is None:
        raise UaDetracParseError(f"{path}: missing <sequence_attribute>")
    camera_state = (attr.get("camera_state") or "").strip()
    weather = (attr.get("sence_weather") or "").strip()
    if not camera_state or not weather:
        raise UaDetracParseError(
            f"{path}: <sequence_attribute> missing camera_state or sence_weather"
        )

    ignored = tuple(
        _box(node, path, "ignored_region") for node in root.findall("ignored_region/box")
    )
    frames = tuple(_frame(node, path) for node in root.findall("frame"))
    if not frames:
        raise UaDetracParseError(f"{path}: no <frame> elements")
    return UaDetracSequence(name, camera_state, weather, ignored, frames)


def parse_directory(dir_path: str | Path) -> tuple[list[UaDetracSequence], list[tuple[Path, str]]]:
    """Parse every ``.xml`` in a directory; returns (successes, [(path, reason), ...] failures)."""
    ok: list[UaDetracSequence] = []
    failed: list[tuple[Path, str]] = []
    for path in sorted(Path(dir_path).glob("*.xml")):
        try:
            ok.append(parse_sequence(path))
        except UaDetracParseError as exc:
            failed.append((path, str(exc)))
    return ok, failed


def _ltwh_to_xyxy(node: ET.Element, path: Path, where: str) -> XYXY:
    try:
        left, top, width, height = (
            float(node.get(tag)) for tag in ("left", "top", "width", "height")
        )
    except (TypeError, ValueError) as exc:
        raise UaDetracParseError(
            f"{path}: {where} has a missing or non-numeric box attribute"
        ) from exc
    if width <= 0 or height <= 0:
        raise UaDetracParseError(f"{path}: {where} has a non-positive width or height")
    return (left, top, left + width, top + height)


def _box(node: ET.Element, path: Path, where: str) -> XYXY:
    return _ltwh_to_xyxy(node, path, where)


def _frame(node: ET.Element, path: Path) -> UaDetracFrame:
    num_text = node.get("num")
    if num_text is None:
        raise UaDetracParseError(f"{path}: <frame> has no num attribute")
    try:
        num = int(num_text)
    except ValueError as exc:
        raise UaDetracParseError(f"{path}: <frame num={num_text!r}> is not an integer") from exc
    targets = tuple(_target(t, path) for t in node.findall("target_list/target"))
    return UaDetracFrame(num, targets)


def _target(node: ET.Element, path: Path) -> UaDetracTarget:
    id_text = node.get("id")
    if id_text is None:
        raise UaDetracParseError(f"{path}: <target> has no id attribute")
    try:
        track_id = int(id_text)
    except ValueError as exc:
        raise UaDetracParseError(f"{path}: <target id={id_text!r}> is not an integer") from exc
    box = node.find("box")
    if box is None:
        raise UaDetracParseError(f"{path}: target {track_id} has no <box>")
    xyxy = _ltwh_to_xyxy(box, path, f"target {track_id}")
    attribute = node.find("attribute")
    if attribute is None:
        raise UaDetracParseError(f"{path}: target {track_id} has no <attribute>")
    vehicle_type = (attribute.get("vehicle_type") or "").strip()
    if not vehicle_type:
        raise UaDetracParseError(f"{path}: target {track_id} has no vehicle_type")
    try:
        truncation_ratio = float(attribute.get("truncation_ratio", "0"))
    except ValueError as exc:
        raise UaDetracParseError(
            f"{path}: target {track_id} has a non-numeric truncation_ratio"
        ) from exc
    return UaDetracTarget(track_id, xyxy, vehicle_type, truncation_ratio)
