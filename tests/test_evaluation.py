"""Detector evaluation metrics (docs/05 section 2.1): hand-computed white-box cases for AP,
precision/recall at an operating point, and the YOLO-label round trip, plus one small
integration test of evaluate_split against a scripted detector."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from trafficwatch import evaluation as ev

CLASSES = ("car", "van", "bus", "other")


def write_label(path, lines):
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def test_load_yolo_labels_round_trips_a_known_box(tmp_path):
    # A 100x50 box centred at (200, 100) in a 400x200 image.
    path = tmp_path / "x.txt"
    write_label(path, ["0 0.500000 0.500000 0.250000 0.250000"])
    (box,) = ev.load_yolo_labels(path, width=400, height=200)
    assert box.cls_id == 0
    assert box.xyxy == pytest.approx((150.0, 75.0, 250.0, 125.0))


def test_load_yolo_labels_missing_file_is_a_valid_negative(tmp_path):
    assert ev.load_yolo_labels(tmp_path / "missing.txt", 100, 100) == []


def test_load_yolo_labels_skips_blank_lines(tmp_path):
    path = tmp_path / "x.txt"
    write_label(path, ["0 0.5 0.5 0.2 0.2", "", "1 0.1 0.1 0.1 0.1"])
    boxes = ev.load_yolo_labels(path, 100, 100)
    assert [b.cls_id for b in boxes] == [0, 1]


# --- average_precision: hand-computed cases -------------------------------------------------


def test_ap_is_one_for_a_single_perfect_detection():
    detections = [("img1", 0.9, (10.0, 10.0, 20.0, 20.0))]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    ap, precisions, recalls = ev.average_precision(detections, gt, iou_thr=0.5)
    assert ap == pytest.approx(1.0)
    assert precisions == [1.0]
    assert recalls == [1.0]


def test_ap_is_zero_with_detections_but_no_ground_truth_match():
    detections = [("img1", 0.9, (1000.0, 1000.0, 1010.0, 1010.0))]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    ap, _, _ = ev.average_precision(detections, gt, iou_thr=0.5)
    assert ap == 0.0


def test_ap_is_zero_with_ground_truth_but_no_detections():
    ap, precisions, recalls = ev.average_precision([], {"img1": [(0.0, 0.0, 10.0, 10.0)]}, 0.5)
    assert ap == 0.0
    assert precisions == []
    assert recalls == []


def test_ap_two_detections_one_correct_one_false_positive_after_it():
    # Confident correct detection first, then a lower-confidence false positive.
    # TP at rank 1 (P=1.0, R=1.0 since the single GT is already matched);
    # FP at rank 2 (cum TP=1, cum FP=1, P=0.5, R=1.0, same recall level as rank 1).
    # The COCO envelope keeps precision at 1.0 for every recall point up to 1.0: FP after
    # the only GT is already matched cannot lower AP, since recall cannot increase further
    # and the 101-point envelope takes the max precision at or above each recall level.
    detections = [
        ("img1", 0.9, (10.0, 10.0, 20.0, 20.0)),  # matches the GT exactly
        ("img1", 0.8, (500.0, 500.0, 510.0, 510.0)),  # nowhere near the GT
    ]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    ap, precisions, recalls = ev.average_precision(detections, gt, iou_thr=0.5)
    assert precisions == pytest.approx([1.0, 0.5])
    assert recalls == pytest.approx([1.0, 1.0])
    assert ap == pytest.approx(1.0)


def test_ap_false_positive_before_the_true_positive_lowers_ap():
    # Higher-confidence false positive is scored first, so recall is still 0 at that rank;
    # AP must come out strictly below 1.0 because the envelope cannot recover that FP away
    # from recall levels at or below the true positive's own recall of 1.0.
    detections = [
        ("img1", 0.95, (500.0, 500.0, 510.0, 510.0)),  # false positive, scored first
        ("img1", 0.9, (10.0, 10.0, 20.0, 20.0)),  # the true positive, scored second
    ]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    ap, precisions, recalls = ev.average_precision(detections, gt, iou_thr=0.5)
    assert precisions == pytest.approx([0.0, 0.5])
    assert recalls == pytest.approx([0.0, 1.0])
    # Envelope is [0.5, 0.5] (max-from-the-right); every one of the 101 recall points is
    # covered by precision 0.5, giving AP exactly 0.5.
    assert ap == pytest.approx(0.5)


def test_ap_two_ground_truth_boxes_one_recovered_gives_half_recall():
    detections = [("img1", 0.9, (10.0, 10.0, 20.0, 20.0))]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0), (200.0, 200.0, 210.0, 210.0)]}
    ap, precisions, recalls = ev.average_precision(detections, gt, iou_thr=0.5)
    assert precisions == pytest.approx([1.0])
    assert recalls == pytest.approx([0.5])
    # Envelope is 1.0 up to recall 0.5, then 0 beyond it: exactly 51 of 101 points (0.00..0.50)
    # score 1.0, the rest score 0 -> AP = 51/101.
    assert ap == pytest.approx(51 / 101)


def test_ap_a_second_detection_cannot_steal_an_already_matched_ground_truth_box():
    # Two detections both overlapping the same single GT box; only the higher-confidence one
    # (scored first) may claim it, the second is a false positive against an already-used box.
    detections = [
        ("img1", 0.9, (10.0, 10.0, 20.0, 20.0)),
        ("img1", 0.8, (11.0, 11.0, 21.0, 21.0)),
    ]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    _ap, precisions, recalls = ev.average_precision(detections, gt, iou_thr=0.5)
    assert precisions == pytest.approx([1.0, 0.5])
    assert recalls == pytest.approx([1.0, 1.0])


def test_ap_respects_the_iou_threshold():
    # A detection offset just enough that IoU falls below 0.5.
    detections = [("img1", 0.9, (15.0, 10.0, 25.0, 20.0))]  # shifted 5px right of a 10x10 box
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    iou = ev.box_iou((15.0, 10.0, 25.0, 20.0), (10.0, 10.0, 20.0, 20.0))
    assert iou < 0.5  # sanity check on the fixture itself
    ap, _, _ = ev.average_precision(detections, gt, iou_thr=0.5)
    assert ap == 0.0


# --- precision_recall_at_conf ----------------------------------------------------------------


def test_precision_recall_at_conf_filters_below_threshold():
    detections = [
        ("img1", 0.9, (10.0, 10.0, 20.0, 20.0)),  # kept, correct
        ("img1", 0.2, (500.0, 500.0, 510.0, 510.0)),  # dropped by the conf filter
    ]
    gt = {"img1": [(10.0, 10.0, 20.0, 20.0)]}
    result = ev.precision_recall_at_conf(detections, gt, iou_thr=0.5, conf_thr=0.35)
    assert result == {"precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0}


def test_precision_recall_at_conf_counts_a_miss_as_a_false_negative():
    result = ev.precision_recall_at_conf([], {"img1": [(0.0, 0.0, 10.0, 10.0)]}, 0.5, 0.35)
    assert result == {"precision": 1.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 1}


# --- summarise_split: per-class exclusion and mean -------------------------------------------


def test_summarise_split_excludes_classes_absent_from_ground_truth():
    detections_by_class = {0: [("img1", 0.9, (10.0, 10.0, 20.0, 20.0))]}
    ground_truth_by_class = {
        0: {"img1": [(10.0, 10.0, 20.0, 20.0)]},
        1: {},  # class "van" never appears in this split
    }
    result = ev.summarise_split(detections_by_class, ground_truth_by_class, CLASSES, 1, 0.35)
    assert result.classes_present == ("car",)
    assert result.map50 == pytest.approx(1.0)
    assert "van" not in result.per_class_ap50


def test_summarise_split_map50_is_the_mean_of_present_classes():
    detections_by_class = {
        0: [("img1", 0.9, (10.0, 10.0, 20.0, 20.0))],  # perfect: AP 1.0
        1: [],  # missed entirely: AP 0.0
    }
    ground_truth_by_class = {
        0: {"img1": [(10.0, 10.0, 20.0, 20.0)]},
        1: {"img2": [(0.0, 0.0, 5.0, 5.0)]},
    }
    result = ev.summarise_split(detections_by_class, ground_truth_by_class, CLASSES, 2, 0.35)
    assert set(result.classes_present) == {"car", "van"}
    assert result.map50 == pytest.approx(0.5)


# --- evaluate_split: small real pass with a scripted detector ---------------------------------


@dataclass
class ScriptedModel:
    """Minimal stand-in for VehicleDetector's inner model: returns one fixed box per call."""

    boxes_by_call: list[list[tuple]]
    call: int = 0

    def predict(self, image, **kwargs):
        from types import SimpleNamespace

        entries = self.boxes_by_call[self.call] if self.call < len(self.boxes_by_call) else []
        self.call += 1
        xyxy = np.array([e[0] for e in entries]) if entries else np.zeros((0, 4))
        conf = np.array([e[1] for e in entries]) if entries else np.zeros((0,))
        cls = np.array([e[2] for e in entries]) if entries else np.zeros((0,))
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy, conf=conf, cls=cls))]


def _write_image(path, width, height):
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def test_evaluate_split_runs_detector_over_a_tiny_image_set(tmp_path):
    import dataclasses

    from trafficwatch.config import load_config
    from trafficwatch.detection import VehicleDetector
    from trafficwatch.preprocessing import Preprocessor

    image_dir, label_dir = tmp_path / "images", tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    _write_image(image_dir / "a.jpg", 640, 640)
    write_label(label_dir / "a.txt", ["0 0.5 0.5 0.1 0.1"])  # a 64x64 car centred on the image
    _write_image(image_dir / "b.jpg", 640, 640)
    write_label(label_dir / "b.txt", [])  # clean negative

    base = load_config("configs/pipeline.yaml", env={})
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"coco")
    cfg = dataclasses.replace(
        base.detector, weights=str(weights), weights_kind="coco_standin", conf=0.001
    )
    # Image a: one correctly placed "car" (coco id 2) detection. Image b: none.
    model = ScriptedModel(boxes_by_call=[[((288.0, 288.0, 352.0, 352.0), 0.9, 2)], []])
    detector = VehicleDetector(cfg, model=model)
    preprocessor = Preprocessor(base.preprocess, cfg)

    overall, groups = ev.evaluate_split(
        detector, preprocessor, image_dir, label_dir, base.detector.classes, operating_conf=0.35
    )
    assert overall.n_images == 2
    assert overall.classes_present == ("car",)
    assert overall.map50 == pytest.approx(1.0)
    assert overall.precision_at_conf["tp"] == 1
    assert overall.precision_at_conf["fp"] == 0
    assert groups == {}
