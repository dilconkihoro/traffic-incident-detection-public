"""Conversion helpers and the deterministic Colab archive."""

from __future__ import annotations

import os
import subprocess
import tarfile

import numpy as np
import pytest

from trafficwatch.data import yolo
from trafficwatch.data.package import deterministic_tar, write_sha256_file


def test_yolo_line_is_normalised_and_hand_computed():
    # Box (100, 200, 300, 400) in a 1000 x 500 image: centre (200, 300), size 200 x 200.
    assert yolo.to_yolo_line(2, (100, 200, 300, 400), 1000, 500) == (
        "2 0.200000 0.600000 0.200000 0.400000"
    )


def test_yolo_line_clips_to_the_image_and_drops_empty_boxes():
    # Clipped to (0, 0, 50, 100) in a 100 x 100 image: centre (25, 50), size 50 x 100.
    assert yolo.to_yolo_line(0, (-20, -5, 50, 130), 100, 100) == (
        "0 0.250000 0.500000 0.500000 1.000000"
    )
    assert yolo.to_yolo_line(0, (120, 10, 150, 40), 100, 100) is None
    assert yolo.to_yolo_line(0, (10, 10, 10, 40), 100, 100) is None


def test_yolo_line_rejects_bad_image_size():
    with pytest.raises(ValueError, match="image size"):
        yolo.to_yolo_line(0, (0, 0, 1, 1), 0, 100)


def test_class_map_maps_drops_and_refuses_the_unknown():
    mapping = {"car": "car", "others": "other", "bicycle": None}
    assert yolo.map_class("car", mapping, "ds") == "car"
    assert yolo.map_class("bicycle", mapping, "ds") is None
    with pytest.raises(yolo.UnmappedClassError, match="tuk-tuk"):
        yolo.map_class("tuk-tuk", mapping, "ds")


@pytest.mark.parametrize(
    ("indices", "stride", "expected"),
    [(range(1, 12), 5, [1, 6, 11]), ([3, 1, 2, 5, 4], 2, [1, 3, 5]), (range(4), 1, [0, 1, 2, 3])],
)
def test_stride_counts_from_the_first_frame(indices, stride, expected):
    assert yolo.strided(indices, stride) == expected


def test_stride_must_be_positive():
    with pytest.raises(ValueError, match="stride"):
        yolo.strided(range(3), 0)


def test_masking_fills_regions_and_reports_the_share():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masked, share = yolo.mask_regions(image, [(0, 0, 5, 4), (2, 2, 6, 6)], (114, 114, 114))
    # Union area: 5*4 + 4*4 - overlap (2..5 x 2..4 = 3*2) = 30 of 100 pixels.
    assert share == pytest.approx(0.30)
    assert (masked[0, 0] == 114).all()
    assert (masked[9, 9] == 0).all()
    assert (image == 0).all()


def test_overlap_fraction_of_a_box_inside_ignored_regions():
    # Box 4x4 at (0, 0); region covers its left half.
    assert yolo.overlap_fraction((0, 0, 4, 4), [(0, 0, 2, 10)], (10, 10)) == pytest.approx(0.5)
    assert yolo.overlap_fraction((5, 5, 8, 8), [(0, 0, 2, 10)], (10, 10)) == 0.0


def make_tree(root, payload=b"a"):
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images" / "train" / "b.jpg").write_bytes(payload * 100)
    (root / "images" / "train" / "a.jpg").write_bytes(b"x" * 10)
    (root / "data.yaml").write_text("names: {0: car}\n", encoding="utf-8")
    return root


def test_tar_is_byte_identical_for_the_same_tree_despite_mtimes(tmp_path):
    first = make_tree(tmp_path / "one" / "yolo_ds_v1")
    second = make_tree(tmp_path / "two" / "yolo_ds_v1")
    os.utime(second / "data.yaml", (12345, 12345))
    h1 = deterministic_tar(first, tmp_path / "one.tar")
    h2 = deterministic_tar(second, tmp_path / "two.tar")
    assert h1 == h2
    third = make_tree(tmp_path / "three" / "yolo_ds_v1", payload=b"z")
    assert deterministic_tar(third, tmp_path / "three.tar") != h1


def test_tar_layout_metadata_and_sha256sum_compatibility(tmp_path):
    tree = make_tree(tmp_path / "yolo_ds_v1")
    tar_path = tmp_path / "yolo_ds_v1.tar"
    digest = deterministic_tar(tree, tar_path)
    with tarfile.open(tar_path) as archive:
        names = archive.getnames()
        assert names[0] == "yolo_ds_v1"
        assert "yolo_ds_v1/images/train/a.jpg" in names
        assert all(m.mtime == 0 and m.uid == 0 and m.gid == 0 for m in archive.getmembers())
    checksum_file = write_sha256_file(tar_path)
    assert checksum_file.read_text(encoding="utf-8") == f"{digest}  yolo_ds_v1.tar\n"
    result = subprocess.run(
        ["sha256sum", "-c", checksum_file.name], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tar_of_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        deterministic_tar(tmp_path / "nope", tmp_path / "x.tar")
