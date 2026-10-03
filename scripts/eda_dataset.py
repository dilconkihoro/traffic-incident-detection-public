"""Exploratory data analysis over the raw datasets (docs/03 section 1, step 2).

Writes data/processed/eda_insights.json and a handful of plots to docs/evidence/, covering
every item in docs/03 section 8's checklist: sequences and frames, boxes per class, boxes per
frame, box-size distribution, weather and night share, ignored-region share, class imbalance.
Reads only the annotation archives and DhakaAI's images (for size); never reads UA-DETRAC's
image archive, so this runs in seconds, not the minutes the full conversion needs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trafficwatch.data.ua_detrac import parse_directory as parse_ua_detrac_dir
from trafficwatch.data.voc import parse_voc_annotation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UA_TRAIN = REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Train-Annotations-XML"
DEFAULT_UA_TEST = REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Test-Annotations-XML"
DEFAULT_DHAKAAI_DIR = REPO_ROOT / "data" / "raw" / "dhakaai" / "train" / "Final Train Dataset"
DEFAULT_OUT_JSON = REPO_ROOT / "data" / "processed" / "eda_insights.json"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence"


def box_wh(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return x2 - x1, y2 - y1


def ua_detrac_insights(train_dir: Path, test_dir: Path) -> dict:
    train_ok, train_failed = parse_ua_detrac_dir(train_dir)
    test_ok, test_failed = parse_ua_detrac_dir(test_dir)
    if train_failed or test_failed:
        raise RuntimeError(f"unparseable UA-DETRAC file(s): {[*train_failed, *test_failed]}")

    classes: Counter[str] = Counter()
    boxes_per_frame: list[int] = []
    widths: list[float] = []
    heights: list[float] = []
    weather: Counter[str] = Counter()
    ignored_region_counts: list[int] = []
    frame_count = 0

    for seq in train_ok:
        weather[seq.weather] += 1
        ignored_region_counts.append(len(seq.ignored_regions))
        for frame in seq.frames:
            frame_count += 1
            boxes_per_frame.append(len(frame.targets))
            for target in frame.targets:
                classes[target.vehicle_type] += 1
                w, h = box_wh(target.xyxy)
                widths.append(w)
                heights.append(h)

    return {
        "train_sequences": len(train_ok),
        "test_sequences": len(test_ok),
        "train_frames": frame_count,
        "total_boxes": sum(classes.values()),
        "boxes_per_class": dict(classes),
        "boxes_per_frame": {
            "mean": _mean(boxes_per_frame),
            "min": min(boxes_per_frame),
            "max": max(boxes_per_frame),
        },
        "box_size_px": {
            "width_mean": _mean(widths),
            "width_median": _median(widths),
            "height_mean": _mean(heights),
            "height_median": _median(heights),
        },
        "weather_share_by_sequence": dict(weather),
        "night_sequence_share": weather.get("night", 0) / len(train_ok) if train_ok else 0.0,
        "ignored_regions_per_sequence": {
            "mean": _mean(ignored_region_counts),
            "sequences_with_none": sum(1 for n in ignored_region_counts if n == 0),
        },
        "_raw_for_plots": {
            "box_widths": widths,
            "box_heights": heights,
            "boxes_per_frame": boxes_per_frame,
        },
    }


def dhakaai_insights(dhakaai_dir: Path) -> dict:
    classes: Counter[str] = Counter()
    image_sizes: Counter[tuple[int, int]] = Counter()
    boxes_per_image: list[int] = []
    unparseable = 0
    zero_size = 0

    for xml_path in sorted(dhakaai_dir.glob("*.xml")):
        try:
            ann = parse_voc_annotation(xml_path)
        except Exception:  # noqa: BLE001 - the real archive's own anomaly, counted not crashed on
            unparseable += 1
            continue
        if ann.width <= 0 or ann.height <= 0:
            zero_size += 1
            continue
        image_sizes[(ann.width, ann.height)] += 1
        boxes_per_image.append(len(ann.objects))
        for obj in ann.objects:
            classes[obj.name] += 1

    return {
        "usable_images": sum(image_sizes.values()),
        "unparseable_files": unparseable,
        "zero_size_files": zero_size,
        "distinct_image_sizes": len(image_sizes),
        "most_common_sizes": image_sizes.most_common(5),
        "total_boxes": sum(classes.values()),
        "boxes_per_class": dict(classes.most_common()),
        "boxes_per_image": {
            "mean": _mean(boxes_per_image),
            "min": min(boxes_per_image) if boxes_per_image else 0,
            "max": max(boxes_per_image) if boxes_per_image else 0,
        },
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def write_plots(ua: dict, dhaka: dict, evidence_dir: Path) -> list[str]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(6, 4))
    raw = ua["_raw_for_plots"]
    ax.scatter(raw["box_widths"][::50], raw["box_heights"][::50], s=4, alpha=0.4)
    ax.set_xlabel("box width (px)")
    ax.set_ylabel("box height (px)")
    ax.set_title("F5.3a: UA-DETRAC train box size distribution (every 50th box)")
    path = evidence_dir / "F5.3a_ua_detrac_box_sizes.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["boxes_per_frame"], bins=30)
    ax.set_xlabel("vehicles per frame")
    ax.set_ylabel("frame count")
    ax.set_title("F5.3b: UA-DETRAC train boxes-per-frame distribution")
    path = evidence_dir / "F5.3b_ua_detrac_boxes_per_frame.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    items = sorted(dhaka["boxes_per_class"].items(), key=lambda kv: -kv[1])
    ax.bar([k for k, _ in items], [v for _, v in items])
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items], rotation=60, ha="right")
    ax.set_ylabel("box count")
    ax.set_title("F5.3c: DhakaAI class imbalance (train.rar)")
    path = evidence_dir / "F5.3c_dhakaai_class_imbalance.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ua-train-annotations", type=Path, default=DEFAULT_UA_TRAIN)
    parser.add_argument("--ua-test-annotations", type=Path, default=DEFAULT_UA_TEST)
    parser.add_argument("--dhakaai-dir", type=Path, default=DEFAULT_DHAKAAI_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    print("Analysing UA-DETRAC annotations...")
    ua = ua_detrac_insights(args.ua_train_annotations, args.ua_test_annotations)
    print("Analysing DhakaAI annotations...")
    dhaka = dhakaai_insights(args.dhakaai_dir)

    plots = [] if args.no_plots else write_plots(ua, dhaka, args.evidence_dir)

    insights = {
        "ua_detrac": {k: v for k, v in ua.items() if k != "_raw_for_plots"},
        "dhakaai": dhaka,
        "figures": plots,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(insights, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.out}")
    for p in plots:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
