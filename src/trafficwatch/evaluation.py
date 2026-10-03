"""Detector evaluation (docs/05 section 2.1): precision/recall at a stated confidence
operating point, mAP50 and mAP50:95, by matching detections against YOLO-format ground truth.

Ultralytics' own ``val()`` (the tool named in docs/05) assumes the model's output class space
already matches the dataset's class list. That holds for a trained model (its head is fine-tuned
directly on ``yolo_ds_v1``'s four classes) but not for the COCO stand-in on its own terms; here it
holds too, because ``VehicleDetector.detect()`` already remaps COCO classes into the unified
``car, van, bus, other`` space before returning (``detection.py``, D-04/D-26). So both ground
truth and predictions live in the same class space, and this module is a standard COCO-style
(101-point interpolated) AP implementation, not a cross-taxonomy bridge. It exists because
``val()`` wants a raw Ultralytics model object, not the wrapped, remapped ``VehicleDetector``.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from trafficwatch.behaviour.kinematics import box_iou
from trafficwatch.detection import VehicleDetector
from trafficwatch.preprocessing import Preprocessor
from trafficwatch.types import XYXY, Frame

IOU_THRESHOLDS_50_95: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class GroundTruthBox:
    cls_id: int
    xyxy: XYXY


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[GroundTruthBox]:
    """Parse a YOLO-format label file (``class xc yc w h``, normalised) back to pixel xyxy.

    A missing label file means the image has no ground-truth boxes (a valid negative), not
    an error, matching how Ultralytics itself treats an absent label.
    """
    if not label_path.is_file():
        return []
    boxes: list[GroundTruthBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id_s, xc_s, yc_s, w_s, h_s = line.split()
        xc, yc = float(xc_s) * width, float(yc_s) * height
        w, h = float(w_s) * width, float(h_s) * height
        boxes.append(
            GroundTruthBox(int(cls_id_s), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        )
    return boxes


Prediction = tuple[str, float, XYXY]  # (image_id, confidence, xyxy)


def _greedy_match(
    ordered: list[Prediction], ground_truth: dict[str, list[XYXY]], iou_thr: float
) -> tuple[list[int], list[int]]:
    """TP/FP flags for predictions already ordered (by confidence, descending).

    Each prediction claims the highest-IoU unmatched ground-truth box in its own image; a
    match below ``iou_thr`` is a false positive. One ground-truth box matches at most once.
    """
    matched = {image_id: [False] * len(boxes) for image_id, boxes in ground_truth.items()}
    tp, fp = [0] * len(ordered), [0] * len(ordered)
    for i, (image_id, _conf, box) in enumerate(ordered):
        gt_boxes = ground_truth.get(image_id, [])
        claims = matched.setdefault(image_id, [False] * len(gt_boxes))
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if claims[j]:
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1
            claims[best_j] = True
        else:
            fp[i] = 1
    return tp, fp


def _cumsum(values: list[int]) -> list[int]:
    out, total = [], 0
    for v in values:
        total += v
        out.append(total)
    return out


def _coco_ap(precisions: list[float], recalls: list[float]) -> float:
    """COCO's 101-point interpolated AP: at each of 101 recall levels, take the maximum
    precision achieved at any recall >= that level (the standard precision envelope)."""
    if not recalls:
        return 0.0
    envelope = precisions[:]
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])
    total = 0.0
    for k in range(101):
        idx = bisect_left(recalls, k / 100.0)
        total += envelope[idx] if idx < len(envelope) else 0.0
    return total / 101.0


def average_precision(
    detections: list[Prediction], ground_truth: dict[str, list[XYXY]], iou_thr: float
) -> tuple[float, list[float], list[float]]:
    """One class's AP at one IoU threshold. Returns (ap, precisions, recalls) over the
    confidence-ordered detections, for plotting the PR curve."""
    n_gt = sum(len(boxes) for boxes in ground_truth.values())
    ordered = sorted(detections, key=lambda d: d[1], reverse=True)
    tp, fp = _greedy_match(ordered, {k: list(v) for k, v in ground_truth.items()}, iou_thr)
    cum_tp, cum_fp = _cumsum(tp), _cumsum(fp)
    precisions = [t / (t + f) if (t + f) > 0 else 1.0 for t, f in zip(cum_tp, cum_fp, strict=True)]
    recalls = [t / n_gt if n_gt > 0 else 0.0 for t in cum_tp]
    return _coco_ap(precisions, recalls), precisions, recalls


def precision_recall_at_conf(
    detections: list[Prediction],
    ground_truth: dict[str, list[XYXY]],
    iou_thr: float,
    conf_thr: float,
) -> dict[str, float | int]:
    """TP/FP/FN counts and precision/recall at one fixed confidence operating point."""
    kept = [d for d in detections if d[1] >= conf_thr]
    ordered = sorted(kept, key=lambda d: d[1], reverse=True)
    n_gt = sum(len(boxes) for boxes in ground_truth.values())
    gt_copy = {k: list(v) for k, v in ground_truth.items()}
    tp_flags, fp_flags = _greedy_match(ordered, gt_copy, iou_thr)
    tp, fp = sum(tp_flags), sum(fp_flags)
    fn = n_gt - tp
    return {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 1.0,
        "recall": tp / n_gt if n_gt > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@dataclass(frozen=True)
class SplitEvalResult:
    """Precision/recall/mAP for one split, overall and per class.

    Classes absent from this split's ground truth are excluded from the per-class mean
    (an AP is not meaningfully defined for a class with zero ground-truth boxes), matching
    standard COCO/Ultralytics practice; ``classes_present`` records which ones counted.
    """

    n_images: int
    operating_conf: float
    classes_present: tuple[str, ...]
    precision_at_conf: dict[str, float | int]
    per_class_precision_recall: dict[str, dict[str, float | int]]
    map50: float
    map50_95: float
    per_class_ap50: dict[str, float]
    per_class_pr_curve: dict[str, dict[str, list[float]]]  # one P/R curve per class, at IoU 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "operating_conf": self.operating_conf,
            "classes_present": list(self.classes_present),
            "precision_at_conf": self.precision_at_conf,
            "per_class_precision_recall": self.per_class_precision_recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "per_class_ap50": self.per_class_ap50,
            "per_class_pr_curve": self.per_class_pr_curve,
        }


def _by_class(
    detections_by_class: dict[int, list[Prediction]],
    ground_truth_by_class: dict[int, dict[str, list[XYXY]]],
    classes: tuple[str, ...],
) -> tuple[int, ...]:
    """Classes with at least one real ground-truth box in this split.

    ``ground_truth_by_class[c]`` may hold an entry per image with an empty box list (recorded
    so recall/false-negative accounting sees every image), so presence must count actual boxes,
    not just check the dict is non-empty.
    """
    return tuple(
        c
        for c in range(len(classes))
        if sum(len(boxes) for boxes in ground_truth_by_class.get(c, {}).values()) > 0
    )


def summarise_split(
    detections_by_class: dict[int, list[Prediction]],
    ground_truth_by_class: dict[int, dict[str, list[XYXY]]],
    classes: tuple[str, ...],
    n_images: int,
    operating_conf: float,
) -> SplitEvalResult:
    present = _by_class(detections_by_class, ground_truth_by_class, classes)

    overall_dets: list[Prediction] = [d for c in present for d in detections_by_class.get(c, [])]
    overall_gt: dict[str, list[XYXY]] = defaultdict(list)
    for c in present:
        for image_id, boxes in ground_truth_by_class.get(c, {}).items():
            overall_gt[image_id].extend(boxes)
    precision_at_conf = precision_recall_at_conf(overall_dets, overall_gt, 0.5, operating_conf)

    per_class_pr = {
        classes[c]: precision_recall_at_conf(
            detections_by_class.get(c, []), ground_truth_by_class.get(c, {}), 0.5, operating_conf
        )
        for c in present
    }

    per_class_ap50: dict[str, float] = {}
    per_class_pr_curve: dict[str, dict[str, list[float]]] = {}
    for c in present:
        ap, precisions, recalls = average_precision(
            detections_by_class.get(c, []), ground_truth_by_class.get(c, {}), 0.5
        )
        per_class_ap50[classes[c]] = ap
        per_class_pr_curve[classes[c]] = {"precision": precisions, "recall": recalls}

    map50 = sum(per_class_ap50.values()) / len(per_class_ap50) if per_class_ap50 else 0.0

    per_threshold_maps = []
    for iou_thr in IOU_THRESHOLDS_50_95:
        aps = [
            average_precision(
                detections_by_class.get(c, []), ground_truth_by_class.get(c, {}), iou_thr
            )[0]
            for c in present
        ]
        per_threshold_maps.append(sum(aps) / len(aps) if aps else 0.0)
    map50_95 = sum(per_threshold_maps) / len(per_threshold_maps) if per_threshold_maps else 0.0

    return SplitEvalResult(
        n_images=n_images,
        operating_conf=operating_conf,
        classes_present=tuple(classes[c] for c in present),
        precision_at_conf=precision_at_conf,
        per_class_precision_recall=per_class_pr,
        map50=map50,
        map50_95=map50_95,
        per_class_ap50=per_class_ap50,
        per_class_pr_curve=per_class_pr_curve,
    )


def evaluate_split(
    detector: VehicleDetector,
    preprocessor: Preprocessor,
    image_dir: Path,
    label_dir: Path,
    classes: tuple[str, ...],
    *,
    operating_conf: float,
    groups: dict[str, str] | None = None,
) -> tuple[SplitEvalResult, dict[str, SplitEvalResult]]:
    """Run the detector over every image in ``image_dir`` and score against ``label_dir``.

    ``detector`` must already be configured at a near-zero confidence threshold (conf=0.001,
    docs/05) so the full precision-recall curve is captured in one pass; ``operating_conf`` is
    applied afterwards by filtering, not by re-running the detector. ``groups`` optionally maps
    an image stem (or its sequence prefix) to a stratification label (for example weather);
    each group gets its own precision/recall/mAP breakdown, returned alongside the overall one.
    """
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    detections_by_class: dict[int, list[Prediction]] = defaultdict(list)
    ground_truth_by_class: dict[int, dict[str, list[XYXY]]] = defaultdict(dict)
    group_images: dict[str, set[str]] = defaultdict(set)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        image_id = image_path.stem
        prepared = preprocessor.apply(Frame(0, 0.0, image))
        for pred in detector.detect(prepared):
            detections_by_class[pred.cls_id].append((image_id, pred.conf, pred.xyxy))
        for c in range(len(classes)):
            ground_truth_by_class[c].setdefault(image_id, [])
        for gt in load_yolo_labels(label_dir / f"{image_id}.txt", width, height):
            ground_truth_by_class[gt.cls_id].setdefault(image_id, []).append(gt.xyxy)
        if groups is not None:
            group_images[groups.get(image_id, "unknown")].add(image_id)

    overall = summarise_split(
        detections_by_class, ground_truth_by_class, classes, len(image_paths), operating_conf
    )

    per_group: dict[str, SplitEvalResult] = {}
    for group, image_ids in group_images.items():
        g_dets = {
            c: [d for d in dets if d[0] in image_ids] for c, dets in detections_by_class.items()
        }
        g_gt = {
            c: {img: boxes for img, boxes in gt.items() if img in image_ids}
            for c, gt in ground_truth_by_class.items()
        }
        per_group[group] = summarise_split(g_dets, g_gt, classes, len(image_ids), operating_conf)

    return overall, per_group
