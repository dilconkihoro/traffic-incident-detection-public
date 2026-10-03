"""T0 zero-shot detector evaluation (docs/05 section 2.1, Gate G2): precision/recall at the
configured confidence operating point, mAP50, mAP50:95, for the COCO stand-in detector against
``data/processed/yolo_ds_v1``'s ``val`` and ``dhaka_val`` splits. The UA-DETRAC ``val`` result is
additionally stratified by the sequence's own weather attribute.

Scope: this script is for the stand-in (``coco_standin``) case specifically, where
``VehicleDetector`` already remaps COCO classes into the unified taxonomy (D-26) before this
script ever sees a prediction. Once a trained model exists (T1 onward), its class head matches
``yolo_ds_v1`` directly and Ultralytics' own `val()` is the right tool (decision D-18); this
script would be a needless reimplementation of it at that point and is not meant to replace it.

Stand-in numbers never enter a final result table (the project rules ML rule 6); this run is logged and
reported as the explicit T0 zero-shot baseline, not as a final detector result.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trafficwatch.config import PipelineConfig, behaviour_hash, config_hash, load_config
from trafficwatch.data.manifest import load_manifest, manifest_hash
from trafficwatch.data.ua_detrac import parse_sequence
from trafficwatch.detection import VehicleDetector
from trafficwatch.evaluation import evaluate_split
from trafficwatch.preprocessing import Preprocessor
from trafficwatch.run_manifest import git_state, hardware_string, package_versions

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "processed" / "yolo_ds_v1"
DEFAULT_UA_TRAIN_ANNOTATIONS = (
    REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Train-Annotations-XML"
)
DEFAULT_OUT = REPO_ROOT / "reports"
DATASET_MANIFEST = REPO_ROOT / "data" / "MANIFEST.json"
EVAL_CONF = 0.001  # near-zero so the full precision-recall curve is captured in one detector pass


def sequence_weather(annotations_dir: Path, sequence_names: list[str]) -> dict[str, str]:
    """Sequence name to its ``sence_weather`` attribute, parsing only the named sequences."""
    weather: dict[str, str] = {}
    for name in sequence_names:
        path = annotations_dir / f"{name}.xml"
        if not path.is_file():
            continue
        weather[name] = parse_sequence(path).weather
    return weather


def image_groups(image_dir: Path, weather_by_sequence: dict[str, str]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        sequence = "_".join(path.stem.split("_")[:-1])
        if sequence in weather_by_sequence:
            groups[path.stem] = weather_by_sequence[sequence]
    return groups


def build_eval_manifest(
    cfg: PipelineConfig,
    *,
    run_id: str,
    weights_kind: str,
    weights_run_id: str,
    weights_sha256: str,
    seed: int,
    dataset_manifest_hash: str | None,
    split_manifest_hash: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "weights_kind": weights_kind,
        "weights_run_id": weights_run_id,
        "weights_sha256": weights_sha256,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "behaviour_hash": behaviour_hash(cfg.behaviour),
        "dataset_manifest_hash": dataset_manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        **git_state(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "hardware": hardware_string(),
    }


def evaluate(
    *,
    config_path: Path,
    dataset_dir: Path,
    out_dir: Path,
    ua_train_annotations: Path,
) -> dict[str, Any]:
    cfg = load_config(config_path, standin=True)
    eval_detector_cfg = dataclasses.replace(cfg.detector, conf=EVAL_CONF)
    detector = VehicleDetector(eval_detector_cfg)
    preprocessor = Preprocessor(cfg.preprocess, eval_detector_cfg)
    classes = cfg.detector.classes
    operating_conf = cfg.detector.conf

    split_manifest_path = dataset_dir / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    val_sequences = split_manifest["sequences"]["val"]
    weather_by_sequence = sequence_weather(ua_train_annotations, val_sequences)

    results: dict[str, Any] = {}

    val_images, val_labels = dataset_dir / "images" / "val", dataset_dir / "labels" / "val"
    groups = image_groups(val_images, weather_by_sequence)
    overall, by_weather = evaluate_split(
        detector,
        preprocessor,
        val_images,
        val_labels,
        classes,
        operating_conf=operating_conf,
        groups=groups,
    )
    results["val"] = {
        "overall": overall.to_dict(),
        "by_weather": {k: v.to_dict() for k, v in by_weather.items()},
    }

    dhaka_images = dataset_dir / "images" / "dhaka_val"
    dhaka_labels = dataset_dir / "labels" / "dhaka_val"
    if dhaka_images.is_dir():
        dhaka_overall, _ = evaluate_split(
            detector,
            preprocessor,
            dhaka_images,
            dhaka_labels,
            classes,
            operating_conf=operating_conf,
            groups=None,
        )
        results["dhaka_val"] = {"overall": dhaka_overall.to_dict()}

    dataset_manifest_hash = (
        manifest_hash(load_manifest(DATASET_MANIFEST)) if DATASET_MANIFEST.is_file() else None
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_eval_t0_{detector.weights_run_id}"
    manifest = build_eval_manifest(
        cfg,
        run_id=run_id,
        weights_kind=detector.weights_kind,
        weights_run_id=detector.weights_run_id,
        weights_sha256=detector.weights_sha256,
        seed=cfg.run.seed,
        dataset_manifest_hash=dataset_manifest_hash,
        split_manifest_hash=manifest_hash(split_manifest),
    )

    report = {
        "manifest": manifest,
        "note": (
            "T0 zero-shot baseline (coco_standin weights): excluded from any final result "
            "table (the project rules ML rule 6). mAP values are this project's own Ultralytics-style "
            "(101-point interpolated) implementation, not the official DETRAC toolkit score, "
            "and are not leaderboard-comparable (docs/05 section 2.1)."
        ),
        "eval_conf_for_curve": EVAL_CONF,
        "operating_conf": operating_conf,
        "splits": results,
    }

    out_path = out_dir / run_id
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_out_path"] = str(out_path / "eval_report.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ua-train-annotations", type=Path, default=DEFAULT_UA_TRAIN_ANNOTATIONS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        ua_train_annotations=args.ua_train_annotations,
    )
    for split, data in report["splits"].items():
        overall = data["overall"]
        print(
            f"{split}: n={overall['n_images']} mAP50={overall['map50']:.4f} "
            f"mAP50:95={overall['map50_95']:.4f} "
            f"P={overall['precision_at_conf']['precision']:.4f} "
            f"R={overall['precision_at_conf']['recall']:.4f} (conf={report['operating_conf']})"
        )
    print(f"wrote {report['_out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
