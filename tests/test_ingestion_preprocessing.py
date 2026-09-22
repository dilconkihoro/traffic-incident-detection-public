"""TC-UT-01 (letterbox and inverse mapping) and ingestion behaviour on real mp4 files."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from synth import grey_ramp, write_video
from trafficwatch.config import load_config
from trafficwatch.ingestion import IngestError, VideoIngestor
from trafficwatch.preprocessing import LETTERBOX_PAD_VALUE, Preprocessor
from trafficwatch.types import Frame

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


@pytest.fixture
def clip(tmp_path):
    return write_video(tmp_path / "ramp.mp4", grey_ramp(20), fps=10.0)


def test_frames_carry_index_time_and_bgr_uint8_images(clip):
    ingestor = VideoIngestor(clip)
    frames = list(ingestor.frames())
    assert ingestor.fps == pytest.approx(10.0)
    assert (ingestor.width, ingestor.height) == (64, 48)
    assert [f.idx for f in frames] == list(range(20))
    assert frames[7].t_s == pytest.approx(0.7)
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[0].image.dtype == np.uint8
    means = [float(f.image.mean()) for f in frames]
    assert means == sorted(means)


def test_frame_stride_keeps_every_nth_frame_with_its_own_timestamp(clip):
    frames = list(VideoIngestor(clip, frame_stride=3).frames())
    assert [f.idx for f in frames] == [0, 3, 6, 9, 12, 15, 18]
    assert frames[1].t_s == pytest.approx(0.3)


def test_frames_can_be_read_twice(clip):
    ingestor = VideoIngestor(clip)
    assert len(list(ingestor.frames())) == len(list(ingestor.frames())) == 20


def test_missing_and_unreadable_files_are_refused(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        VideoIngestor(tmp_path / "nope.mp4")
    fake = tmp_path / "fake.mp4"
    fake.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(IngestError):
        VideoIngestor(fake)


def test_stride_must_be_positive(clip):
    with pytest.raises(IngestError, match="frame_stride"):
        VideoIngestor(clip, frame_stride=0)


def preprocessor(**changes):
    cfg = load_config(SHIPPED, env={})
    return Preprocessor(dataclasses.replace(cfg.preprocess, **changes), cfg.detector)


def frame_of(image):
    return Frame(0, 0.0, image)


def test_letterbox_of_a_1080p_frame_matches_hand_computed_geometry():
    # scale 640/1920, resized 640 x 360, vertical pad (640 - 360) / 2 = 140 each side.
    image = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    prepared = preprocessor().apply(frame_of(image))
    assert prepared.image640.shape == (640, 640, 3)
    assert prepared.image640.dtype == np.uint8
    assert prepared.scale == pytest.approx(640 / 1920)
    assert prepared.pad == (0.0, 140.0)
    assert prepared.original_wh == (1920, 1080)
    canvas = prepared.image640
    assert (canvas[139] == LETTERBOX_PAD_VALUE).all()
    assert (canvas[140] == 200).all()
    assert (canvas[499] == 200).all()
    assert (canvas[500] == LETTERBOX_PAD_VALUE).all()


def test_portrait_frame_is_padded_left_and_right():
    prepared = preprocessor().apply(frame_of(np.zeros((1280, 720, 3), dtype=np.uint8)))
    assert prepared.scale == pytest.approx(0.5)
    assert prepared.pad == (140.0, 0.0)


def test_square_frame_of_detector_size_is_unchanged():
    image = np.random.default_rng(0).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    prepared = preprocessor().apply(frame_of(image))
    assert prepared.scale == 1.0
    assert prepared.pad == (0.0, 0.0)
    assert np.array_equal(prepared.image640, image)


@pytest.mark.parametrize(
    "shape", [(1080, 1920), (720, 1280), (480, 854), (3, 1000), (1000, 3), (541, 961)]
)
def test_output_is_always_the_detector_size(shape):
    prepared = preprocessor().apply(frame_of(np.zeros((*shape, 3), dtype=np.uint8)))
    assert prepared.image640.shape == (640, 640, 3)


def test_boxes_invert_within_a_pixel_after_the_forward_mapping():
    prepared = preprocessor().apply(frame_of(np.zeros((1080, 1920, 3), dtype=np.uint8)))
    original = (100.0, 200.0, 500.0, 700.0)
    (left, top), s = prepared.pad, prepared.scale
    letterboxed = (
        original[0] * s + left,
        original[1] * s + top,
        original[2] * s + left,
        original[3] * s + top,
    )
    assert prepared.to_original(letterboxed) == pytest.approx(original, abs=1.0)


def test_boxes_are_clipped_to_the_original_frame():
    prepared = preprocessor().apply(frame_of(np.zeros((1080, 1920, 3), dtype=np.uint8)))
    assert prepared.to_original((-50.0, -50.0, 700.0, 700.0)) == (0.0, 0.0, 1920.0, 1080.0)


def test_low_light_enhancement_raises_contrast_and_keeps_shape_and_type():
    ramp = np.tile(np.linspace(20, 60, 640).astype(np.uint8), (640, 1))
    dark = np.dstack([ramp, ramp, ramp])
    plain = preprocessor().apply(frame_of(dark)).image640
    enhanced = preprocessor(low_light_enhance=True).apply(frame_of(dark)).image640
    assert enhanced.shape == plain.shape
    assert enhanced.dtype == np.uint8
    assert enhanced.std() > plain.std()
