"""Convert UA-DETRAC and DhakaAI raw archives into one YOLO-format training dataset (docs/03).

Builds data/processed/yolo_ds_v1/ end to end: parses both datasets, applies the v1 class
taxonomy (D-04, docs/07), splits UA-DETRAC by whole sequence (leakage-checked against the
official test list) and DhakaAI by perceptual-hash cluster, masks UA-DETRAC's ignored regions,
strides UA-DETRAC frames, writes YOLO labels and images, and packages a deterministic tar.

Scope for this run, deliberate: official UA-DETRAC **test** sequence images are parsed only for
the leakage check and never converted or written here. Building that split is Phase 5 work, the
one time the test split is actually touched (the project rules rule 2); converting it now would be
needless test-data handling with no use until then.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from trafficwatch.data.config import DataConfig, load_data_config
from trafficwatch.data.manifest import sha256_file
from trafficwatch.data.package import deterministic_tar
from trafficwatch.data.split import (
    SequenceInfo,
    assign_clusters,
    assign_splits,
    check_hash_leakage,
    check_leakage,
    choose_val_sequences,
    split_manifest,
)
from trafficwatch.data.ua_detrac import UaDetracSequence
from trafficwatch.data.ua_detrac import parse_directory as parse_ua_detrac_dir
from trafficwatch.data.voc import VocParseError, parse_voc_annotation
from trafficwatch.data.yolo import map_class, mask_regions, overlap_fraction, strided, to_yolo_line

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"
DEFAULT_UA_TRAIN_ANNOTATIONS = (
    REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Train-Annotations-XML"
)
DEFAULT_UA_TEST_ANNOTATIONS = (
    REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Test-Annotations-XML"
)
DEFAULT_UA_IMAGES_ZIP = REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Images.zip"
DEFAULT_DHAKAAI_DIR = REPO_ROOT / "data" / "raw" / "dhakaai" / "train" / "Final Train Dataset"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "yolo_ds_v1"
DATASET_VERSION = "yolo_ds_v1"


class ConversionError(RuntimeError):
    pass


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_dirs(out: Path, splits: list[str]) -> None:
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)


def class_index(cfg: DataConfig, unified: str) -> int:
    return cfg.unified_classes.index(unified)


# ----- UA-DETRAC -----


def load_ua_detrac(
    train_dir: Path, test_dir: Path
) -> tuple[list[UaDetracSequence], list[SequenceInfo], set[str]]:
    train_ok, train_failed = parse_ua_detrac_dir(train_dir)
    test_ok, test_failed = parse_ua_detrac_dir(test_dir)
    if train_failed or test_failed:
        bad = ", ".join(str(p) for p, _ in (*train_failed, *test_failed))
        raise ConversionError(f"unparseable UA-DETRAC annotation file(s): {bad}")

    sequences = [*train_ok, *test_ok]
    infos = [SequenceInfo(s.name, "train", s.weather) for s in train_ok] + [
        SequenceInfo(s.name, "test", s.weather) for s in test_ok
    ]
    official_test_ids = {s.name for s in test_ok}
    return sequences, infos, official_test_ids


def convert_ua_detrac_sequence(
    *,
    seq: UaDetracSequence,
    split: str,
    cfg: DataConfig,
    images_zip: zipfile.ZipFile,
    out: Path,
    dry_run: bool,
    stats: Counter,
) -> None:
    frame_nums = strided([f.num for f in seq.frames], cfg.stride)
    frames_by_num = {f.num: f for f in seq.frames}
    class_map = cfg.class_map["ua-detrac"]

    for num in frame_nums:
        frame = frames_by_num[num]
        zip_path = f"DETRAC-Images/{seq.name}/img{num:05d}.jpg"
        try:
            raw = images_zip.read(zip_path)
        except KeyError as exc:
            raise ConversionError(
                f"{seq.name} frame {num}: {zip_path} not found in images zip"
            ) from exc
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ConversionError(f"{seq.name} frame {num}: could not decode {zip_path}")
        height, width = image.shape[:2]

        lines: list[str] = []
        for target in frame.targets:
            stats["ua_detrac_boxes_total"] += 1
            unified = map_class(target.vehicle_type, class_map, "ua-detrac")
            if unified is None:
                stats["ua_detrac_boxes_dropped_class"] += 1
                continue
            if (
                overlap_fraction(target.xyxy, seq.ignored_regions, (height, width))
                >= cfg.ignored_box_overlap
            ):
                stats["ua_detrac_boxes_dropped_ignored_region"] += 1
                continue
            line = to_yolo_line(class_index(cfg, unified), target.xyxy, width, height)
            if line is None:
                stats["ua_detrac_boxes_dropped_degenerate"] += 1
                continue
            lines.append(line)
            stats["ua_detrac_boxes_kept"] += 1

        stats[f"ua_detrac_frames_{split}"] += 1
        if dry_run:
            continue
        masked, masked_share = mask_regions(image, seq.ignored_regions, tuple(cfg.mask_colour))
        stats["ua_detrac_masked_pixel_share_sum"] += masked_share
        stem = f"{seq.name}_{num:06d}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), masked)
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


# ----- DhakaAI -----


def phash_of(path: Path) -> int:
    with Image.open(path) as img:
        return int(str(imagehash.phash(img)), 16)


def find_image(dhakaai_dir: Path, stem: str) -> Path | None:
    """The archive's actual image file for an XML stem, whatever its real extension and case
    (confirmed on the real files: mixed .jpg/.JPG/.png/.PNG/.jpeg). The XML's own <filename> is
    the uploader's original name, not the file actually shipped in the archive; every image is
    instead matched to its label purely by sharing the same stem as the XML, which the full real
    archive (3,003 files) confirms is always unique and always present."""
    for candidate in dhakaai_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() != ".xml":
            return candidate
    return None


def convert_dhakaai(
    *, cfg: DataConfig, dhakaai_dir: Path, out: Path, dry_run: bool, stats: Counter
) -> dict[str, str]:
    hashes: dict[str, int] = {}
    by_stem = {}
    image_path_by_stem: dict[str, Path] = {}
    for xml_path in sorted(dhakaai_dir.glob("*.xml")):
        try:
            ann = parse_voc_annotation(xml_path)
        except VocParseError as exc:
            stats["dhakaai_files_unparseable"] += 1
            print(f"  skipping {xml_path.name}: {exc}", file=sys.stderr)
            continue
        if ann.width <= 0 or ann.height <= 0:
            # A real upstream defect (confirmed: Pias (359).xml, Pias (360).xml both have
            # <width>0</width><height>0</height>), the same class of error as 231.xml: detect
            # and skip, do not let a degenerate size crash the run or get silently coerced.
            stats["dhakaai_files_zero_size"] += 1
            print(f"  skipping {xml_path.name}: zero width or height in <size>", file=sys.stderr)
            continue
        stem = xml_path.stem
        image_path = find_image(dhakaai_dir, stem)
        if image_path is None:
            stats["dhakaai_images_missing"] += 1
            continue
        hashes[stem] = phash_of(image_path)
        by_stem[stem] = ann
        image_path_by_stem[stem] = image_path

    cluster_of = {}
    ids = sorted(hashes)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1 :]:
            if (hashes[a] ^ hashes[b]).bit_count() <= cfg.phash_max_distance:
                parent[find(b)] = find(a)
    roots = sorted({find(i) for i in ids})
    numbered = {root: n for n, root in enumerate(roots)}
    cluster_of = {i: numbered[find(i)] for i in ids}

    assignment = assign_clusters(cluster_of, cfg.dhaka_fractions, cfg.seed)
    violations = check_hash_leakage(hashes, assignment, cfg.phash_max_distance)
    if violations:
        raise ConversionError("DhakaAI perceptual-hash leakage:\n  " + "\n  ".join(violations))

    class_map = cfg.class_map["dhakaai"]
    folder_name = {"train": "dhaka_train", "dhaka_val": "dhaka_val", "dhaka_test": "dhaka_test"}
    for stem, split in assignment.items():
        ann = by_stem[stem]
        folder = folder_name[split]
        lines = []
        for obj in ann.objects:
            stats["dhakaai_boxes_total"] += 1
            unified = map_class(obj.name, class_map, "dhakaai")
            if unified is None:
                stats["dhakaai_boxes_dropped_class"] += 1
                continue
            line = to_yolo_line(class_index(cfg, unified), obj.bndbox, ann.width, ann.height)
            if line is None:
                stats["dhakaai_boxes_dropped_degenerate"] += 1
                continue
            lines.append(line)
            stats["dhakaai_boxes_kept"] += 1
        stats[f"dhakaai_images_{folder}"] += 1
        if dry_run:
            continue
        (out / "labels" / folder / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        image_path = image_path_by_stem[stem]
        (out / "images" / folder / f"{stem}{image_path.suffix.lower()}").write_bytes(
            image_path.read_bytes()
        )
    return {stem: folder_name[split] for stem, split in assignment.items()}


# ----- orchestration -----


def convert(
    *,
    data_config: Path,
    ua_train_annotations: Path,
    ua_test_annotations: Path,
    ua_images_zip: Path,
    dhakaai_dir: Path,
    out: Path,
    dry_run: bool,
) -> int:
    cfg = load_data_config(data_config)
    stats: Counter = Counter()

    print("Parsing UA-DETRAC annotations...")
    sequences, infos, official_test_ids = load_ua_detrac(ua_train_annotations, ua_test_annotations)

    val_ids = choose_val_sequences(infos, cfg.val_sequences, cfg.seed)
    assignment = assign_splits(infos, val_ids)
    violations = check_leakage(assignment, infos, official_test_ids)
    if violations:
        print("LEAKAGE CHECK FAILED, nothing written:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 2
    n_train = sum(1 for s in assignment.values() if s == "train")
    n_test = sum(1 for s in assignment.values() if s == "test")
    print(
        f"  {len(sequences)} sequences: {n_train} train, {len(val_ids)} val, "
        f"{n_test} test (not converted)"
    )

    splits_to_write = ["train", "val"]
    if not dry_run:
        ensure_dirs(out, splits_to_write)

    by_name = {s.name: s for s in sequences}
    with zipfile.ZipFile(ua_images_zip) as images_zip:
        for seq_id, split in sorted(assignment.items()):
            if split == "test":
                continue  # parsed for the leakage check only; see module docstring
            print(f"  converting {seq_id} ({split})...")
            convert_ua_detrac_sequence(
                seq=by_name[seq_id],
                split=split,
                cfg=cfg,
                images_zip=images_zip,
                out=out,
                dry_run=dry_run,
                stats=stats,
            )

    print("Parsing and converting DhakaAI...")
    if not dry_run:
        ensure_dirs(out, ["dhaka_train", "dhaka_val", "dhaka_test"])
    dhaka_assignment = convert_dhakaai(
        cfg=cfg, dhakaai_dir=dhakaai_dir, out=out, dry_run=dry_run, stats=stats
    )

    sha = git_sha()
    manifest = split_manifest(assignment, seed=cfg.seed, stride=cfg.stride, git_sha=sha)
    manifest["dhakaai"] = {
        "fractions": cfg.dhaka_fractions,
        "phash_max_distance": cfg.phash_max_distance,
        "images_by_split": {
            split_name: sorted(f for f, s in dhaka_assignment.items() if s == split_name)
            for split_name in ("dhaka_train", "dhaka_val", "dhaka_test")
        },
    }

    class_map_out = {
        "unified_classes": list(cfg.unified_classes),
        "index": {i: name for i, name in enumerate(cfg.unified_classes)},
        "source_class_map": {k: dict(v) for k, v in cfg.class_map.items()},
        "version": DATASET_VERSION,
    }

    stats_out = {
        "unified_classes": list(cfg.unified_classes),
        "stride": cfg.stride,
        **dict(stats),
        "ua_detrac_masked_pixel_share_mean": (
            stats["ua_detrac_masked_pixel_share_sum"] / stats["ua_detrac_frames_train"]
            if stats.get("ua_detrac_frames_train")
            else 0.0
        ),
    }
    stats_out.pop("ua_detrac_masked_pixel_share_sum", None)

    print(f"Writing stats, class map, split manifest to {out}...")
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "class_map.json").write_text(json.dumps(class_map_out, indent=2), encoding="utf-8")
        (out / "stats.json").write_text(
            json.dumps(stats_out, indent=2, default=str), encoding="utf-8"
        )
        (out / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        (out / "data.yaml").write_text(
            "path: /content/data/" + DATASET_VERSION + "\n"
            "train:\n  - images/train\n  - images/dhaka_train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(cfg.unified_classes)),
            encoding="utf-8",
        )
    else:
        print(json.dumps(stats_out, indent=2, default=str))

    if dry_run:
        print("Dry run: no images, tar, or hash written.")
        return 0

    tar_path = out.parent / f"{DATASET_VERSION}.tar"
    print(f"Packaging {tar_path}...")
    deterministic_tar(out, tar_path)
    digest = sha256_file(tar_path)
    (out.parent / f"{DATASET_VERSION}.sha256").write_text(
        f"{digest}  {DATASET_VERSION}.tar\n", encoding="utf-8"
    )
    print(f"sha256 {digest}  {tar_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--ua-train-annotations", type=Path, default=DEFAULT_UA_TRAIN_ANNOTATIONS)
    parser.add_argument("--ua-test-annotations", type=Path, default=DEFAULT_UA_TEST_ANNOTATIONS)
    parser.add_argument("--ua-images-zip", type=Path, default=DEFAULT_UA_IMAGES_ZIP)
    parser.add_argument("--dhakaai-dir", type=Path, default=DEFAULT_DHAKAAI_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="compute stats, write nothing")
    args = parser.parse_args(argv)

    try:
        return convert(
            data_config=args.data_config,
            ua_train_annotations=args.ua_train_annotations,
            ua_test_annotations=args.ua_test_annotations,
            ua_images_zip=args.ua_images_zip,
            dhakaai_dir=args.dhakaai_dir,
            out=args.out,
            dry_run=args.dry_run,
        )
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
