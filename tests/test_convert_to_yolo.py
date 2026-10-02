"""scripts/convert_to_yolo.py: the orchestration specific to this script (class mapping, stride,
ignored-region masking, image resolution, leakage abort). split.py, yolo.py, and voc.py's own
logic is already covered by their own test files; this file does not re-test it."""

from __future__ import annotations

import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_to_yolo as c  # noqa: E402
from trafficwatch.data.config import load_data_config  # noqa: E402
from trafficwatch.data.ua_detrac import (  # noqa: E402
    UaDetracFrame,
    UaDetracSequence,
    UaDetracTarget,
)

MINIMAL_DATA_CONFIG = {
    "seed": 0,
    "stride": 2,
    "val_sequences": 1,
    "unified_classes": ["car", "van", "bus", "other"],
    "mask_colour": [114, 114, 114],
    "ignored_box_overlap": 0.5,
    "dhaka": {
        "phash_max_distance": 4,
        "fractions": {"train": 0.5, "dhaka_val": 0.25, "dhaka_test": 0.25},
    },
    "class_map": {
        "ua-detrac": {"car": "car", "van": "van", "bus": "bus", "others": "other"},
        "dhakaai": {"car": "car", "bus": "bus", "rickshaw": None},
    },
}


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(MINIMAL_DATA_CONFIG), encoding="utf-8")
    return load_data_config(path)


# ----- find_image -----


def test_find_image_resolves_regardless_of_case_or_extension(tmp_path):
    (tmp_path / "07.xml").write_text("<annotation/>", encoding="utf-8")
    (tmp_path / "07.PNG").write_bytes(b"not a real png, just bytes")
    found = c.find_image(tmp_path, "07")
    assert found == tmp_path / "07.PNG"


def test_find_image_returns_none_when_no_image_matches_the_stem(tmp_path):
    (tmp_path / "09.xml").write_text("<annotation/>", encoding="utf-8")
    assert c.find_image(tmp_path, "09") is None


# ----- UA-DETRAC conversion -----


def _frame_jpeg_bytes(colour: tuple[int, int, int], size=(64, 48)) -> bytes:
    image = np.full((size[1], size[0], 3), colour, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_convert_ua_detrac_sequence_strides_maps_classes_and_masks_ignored_regions(tmp_path, cfg):
    targets_frame1 = (
        UaDetracTarget(
            1, (0.0, 0.0, 10.0, 10.0), "car", 0.0
        ),  # inside the ignored region -> dropped
        UaDetracTarget(2, (30.0, 30.0, 50.0, 46.0), "van", 0.0),  # kept
    )
    targets_frame3 = (UaDetracTarget(2, (30.0, 30.0, 50.0, 46.0), "van", 0.0),)
    seq = UaDetracSequence(
        name="MVI_TEST",
        camera_state="stable",
        weather="sunny",
        ignored_regions=((0.0, 0.0, 15.0, 15.0),),
        frames=(
            UaDetracFrame(1, targets_frame1),
            UaDetracFrame(2, ()),  # skipped by stride=2 (every 2nd of [1, 2, 3] -> frames 1, 3)
            UaDetracFrame(3, targets_frame3),
        ),
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("DETRAC-Images/MVI_TEST/img00001.jpg", _frame_jpeg_bytes((50, 50, 50)))
        zf.writestr("DETRAC-Images/MVI_TEST/img00003.jpg", _frame_jpeg_bytes((50, 50, 50)))
    buf.seek(0)

    out = tmp_path / "out"
    c.ensure_dirs(out, ["train"])  # convert() always does this before calling the function below
    stats = Counter()
    with zipfile.ZipFile(buf) as zf:
        c.convert_ua_detrac_sequence(
            seq=seq, split="train", cfg=cfg, images_zip=zf, out=out, dry_run=False, stats=stats
        )

    assert stats["ua_detrac_frames_train"] == 2  # frames 1 and 3 only, never frame 2
    assert stats["ua_detrac_boxes_dropped_ignored_region"] == 1  # track 1 in frame 1
    assert stats["ua_detrac_boxes_kept"] == 2  # track 2 in frames 1 and 3

    label_1 = (out / "labels" / "train" / "MVI_TEST_000001.txt").read_text(encoding="utf-8")
    assert label_1.strip().startswith("1 ")  # class index 1 = van
    assert not (out / "labels" / "train" / "MVI_TEST_000002.txt").exists()
    assert (out / "images" / "train" / "MVI_TEST_000001.jpg").is_file()

    masked = cv2.imread(str(out / "images" / "train" / "MVI_TEST_000001.jpg"))
    assert tuple(int(v) for v in masked[5, 5]) == (114, 114, 114)  # ignored region painted over
    assert tuple(int(v) for v in masked[40, 40]) == (50, 50, 50)  # elsewhere untouched


def test_convert_ua_detrac_sequence_dry_run_writes_nothing(tmp_path, cfg):
    seq = UaDetracSequence(
        "MVI_DRY",
        "stable",
        "sunny",
        (),
        (UaDetracFrame(1, (UaDetracTarget(1, (0.0, 0.0, 10.0, 10.0), "car", 0.0),)),),
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("DETRAC-Images/MVI_DRY/img00001.jpg", _frame_jpeg_bytes((1, 2, 3)))
    buf.seek(0)
    out = tmp_path / "out"
    stats = Counter()
    with zipfile.ZipFile(buf) as zf:
        c.convert_ua_detrac_sequence(
            seq=seq, split="train", cfg=cfg, images_zip=zf, out=out, dry_run=True, stats=stats
        )
    assert stats["ua_detrac_boxes_kept"] == 1
    assert not out.exists()


# ----- DhakaAI conversion -----


def _write_voc(
    path: Path, *, filename: str, width: int, height: int, objects: list[tuple[str, tuple]]
):
    objs = "".join(
        f"<object><name>{name}</name><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox>"
        f"</object>"
        for name, b in objects
    )
    path.write_text(
        f"<annotation><filename>{filename}</filename>"
        f"<size><width>{width}</width><height>{height}</height><depth>3</depth></size>"
        f"{objs}</annotation>",
        encoding="utf-8",
    )


def _write_image(path: Path, colour=(10, 20, 30), size=(64, 48)) -> None:
    Image.new("RGB", size, colour).save(path)


def test_convert_dhakaai_matches_images_by_stem_not_embedded_filename(tmp_path, cfg):
    """The real archive's <filename> is the uploader's original name, never the actual file on
    disk (confirmed against all 3,003 real files); matching must go by the XML's own stem."""
    _write_voc(
        tmp_path / "01.xml",
        filename="totally_unrelated_name.jpg",
        width=64,
        height=48,
        objects=[("car", (0, 0, 20, 20))],
    )
    _write_image(tmp_path / "01.jpg")

    stats = Counter()
    assignment = c.convert_dhakaai(
        cfg=cfg, dhakaai_dir=tmp_path, out=tmp_path / "out", dry_run=True, stats=stats
    )
    assert "01" in assignment
    assert stats["dhakaai_boxes_kept"] == 1


def test_convert_dhakaai_skips_a_zero_size_annotation_without_crashing(tmp_path, cfg):
    """A real upstream defect in the archive: Pias (359).xml and Pias (360).xml both embed
    <width>0</width><height>0</height>."""
    _write_voc(
        tmp_path / "bad.xml", filename="x.jpg", width=0, height=0, objects=[("car", (0, 0, 1, 1))]
    )
    _write_image(tmp_path / "bad.jpg")
    _write_voc(
        tmp_path / "good.xml",
        filename="y.jpg",
        width=64,
        height=48,
        objects=[("car", (0, 0, 20, 20))],
    )
    _write_image(tmp_path / "good.jpg")

    stats = Counter()
    assignment = c.convert_dhakaai(
        cfg=cfg, dhakaai_dir=tmp_path, out=tmp_path / "out", dry_run=True, stats=stats
    )
    assert stats["dhakaai_files_zero_size"] == 1
    assert "bad" not in assignment
    assert "good" in assignment


def test_convert_dhakaai_drops_classes_mapped_to_null_without_folding_into_other(tmp_path, cfg):
    _write_voc(
        tmp_path / "01.xml",
        filename="x.jpg",
        width=64,
        height=48,
        objects=[("car", (0, 0, 20, 20)), ("rickshaw", (0, 0, 20, 20))],
    )
    _write_image(tmp_path / "01.jpg")

    stats = Counter()
    c.convert_dhakaai(
        cfg=cfg, dhakaai_dir=tmp_path, out=tmp_path / "out", dry_run=True, stats=stats
    )
    assert stats["dhakaai_boxes_kept"] == 1
    assert stats["dhakaai_boxes_dropped_class"] == 1


def test_convert_dhakaai_writes_real_files_with_the_images_real_extension(tmp_path, cfg):
    _write_voc(
        tmp_path / "01.xml",
        filename="x.jpg",
        width=64,
        height=48,
        objects=[("car", (0, 0, 20, 20))],
    )
    _write_image(tmp_path / "01.PNG", colour=(7, 8, 9))

    out = tmp_path / "out"
    c.ensure_dirs(out, ["dhaka_train", "dhaka_val", "dhaka_test"])

    stats = Counter()
    assignment = c.convert_dhakaai(
        cfg=cfg, dhakaai_dir=tmp_path, out=out, dry_run=False, stats=stats
    )
    folder = assignment["01"]
    assert (out / "images" / folder / "01.png").read_bytes() == (tmp_path / "01.PNG").read_bytes()
    assert (out / "labels" / folder / "01.txt").is_file()


# ----- leakage abort -----


def test_convert_aborts_and_writes_nothing_when_the_leakage_check_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "check_leakage", lambda *a, **k: ["fabricated violation for this test"])
    (tmp_path / "ua_train").mkdir()
    (tmp_path / "ua_test").mkdir()
    (tmp_path / "dhaka").mkdir()
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump(MINIMAL_DATA_CONFIG), encoding="utf-8")

    def fake_load_ua_detrac(_train, _test):
        seq = UaDetracSequence(
            "MVI_X",
            "stable",
            "sunny",
            (),
            (UaDetracFrame(1, (UaDetracTarget(1, (0, 0, 10, 10), "car", 0.0),)),),
        )
        from trafficwatch.data.split import SequenceInfo

        return [seq], [SequenceInfo("MVI_X", "train", "sunny")], set()

    monkeypatch.setattr(c, "load_ua_detrac", fake_load_ua_detrac)

    out = tmp_path / "out"
    result = c.convert(
        data_config=data_config,
        ua_train_annotations=tmp_path / "ua_train",
        ua_test_annotations=tmp_path / "ua_test",
        ua_images_zip=tmp_path / "images.zip",
        dhakaai_dir=tmp_path / "dhaka",
        out=out,
        dry_run=False,
    )
    assert result == 2
    assert not out.exists()
