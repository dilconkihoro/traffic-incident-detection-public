"""Loggers (TC-UT-02), run manifest, latency report, and the pipeline on synthetic clips."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from synth import ScriptedDetector, abrupt_stop_paths, render_scene, write_video
from trafficwatch import cli
from trafficwatch.config import behaviour_hash, load_config
from trafficwatch.incident_log import INCIDENT_COLUMNS, TRACK_COLUMNS, IncidentLogger, TrackLogger
from trafficwatch.ingestion import VideoIngestor
from trafficwatch.pipeline import (
    STAGES,
    WARMUP_FRAMES,
    Pipeline,
    annotate,
    latency_report,
    seed_everything,
)
from trafficwatch.run_manifest import build_run_manifest
from trafficwatch.types import Incident, IncidentType, Track

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "configs" / "pipeline.yaml"


def cfg_with(tmp_path, **output):
    cfg = load_config(SHIPPED, env={})
    out = dataclasses.replace(cfg.output, dir=str(tmp_path / "reports"), **output)
    return dataclasses.replace(cfg, output=out)


def run_scene(tmp_path, paths, cfg=None):
    frames, detections = render_scene(paths)
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    return Pipeline(cfg or cfg_with(tmp_path), ScriptedDetector(detections)).run(clip)


@pytest.fixture(scope="module")
def abrupt_result(tmp_path_factory):
    """One shared end-to-end run; the embedder makes each scene run cost several seconds."""
    return run_scene(tmp_path_factory.mktemp("abrupt"), abrupt_stop_paths())


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_incident_row_is_formatted_exactly_and_flushed_per_write(tmp_path):
    incident = Incident(
        IncidentType.COLLISION_LIKE, (3, 7), 40, 44, 4.0, 4.4, (10.04, 20.0, 110.0, 120.06), 0.5
    )
    path = tmp_path / "incidents.csv"
    logger = IncidentLogger(
        path, "RUN1", "clipA", rule_params_hash="abc", weights_run_id="R9", weights_kind="trained"
    )
    logger.write(incident)
    lines = path.read_text(encoding="utf-8").splitlines()  # read before close: flushed already
    assert lines[0] == ",".join(INCIDENT_COLUMNS)
    assert lines[1] == ",".join(
        ["1", "RUN1", "clipA", "COLLISION_LIKE", "3;7", "40", "44", "4.000", "4.400"]
        + ["10.0", "20.0", "110.0", "120.1", "0.500", "abc", "R9", "trained"]
    )
    logger.write(incident)
    logger.close()
    assert [r["incident_id"] for r in read_rows(path)] == ["1", "2"]


def test_track_row_is_formatted_exactly(tmp_path):
    path = tmp_path / "tracks.csv"
    with TrackLogger(path, "clipA") as logger:
        logger.write(Track(5, (1.25, 2.5, 30.0, 40.0), "bus", 4, 4, True), 12, 1.2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [",".join(TRACK_COLUMNS), "clipA,12,1.200,5,bus,1.2,2.5,30.0,40.0"]


def test_run_manifest_carries_the_reproducibility_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config(SHIPPED, env={})
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 10)
    manifest = build_run_manifest(
        cfg, run_id="R", video_path=video, weights_kind="coco_standin",
        weights_run_id="coco_standin", weights_sha256="a" * 64, seed=cfg.run.seed,
    )  # fmt: skip
    assert manifest["weights_kind"] == "coco_standin"
    assert manifest["behaviour_hash"] == behaviour_hash(cfg.behaviour)
    assert len(manifest["config_hash"]) == 64 and len(manifest["dataset_manifest_hash"]) == 64
    assert manifest["video"]["size_bytes"] == 10
    assert manifest["packages"]["numpy"] and "threads" in manifest["hardware"]
    assert manifest["git_sha"] is None or len(manifest["git_sha"]) == 40
    assert isinstance(manifest["git_dirty"], bool | type(None))


def test_latency_report_excludes_warmup_and_computes_hand_values():
    samples = {stage: [0.001] * (WARMUP_FRAMES + 10) for stage in STAGES}
    report = latency_report(samples, source_fps=25.0)
    assert report["warmup_frames_excluded"] == WARMUP_FRAMES
    assert report["frames_measured"] == 10
    assert report["total"]["mean_ms"] == pytest.approx(5.0)
    assert report["fps"] == pytest.approx(200.0)
    assert report["stages"]["detection"]["p95_ms"] == pytest.approx(1.0)


def test_latency_report_keeps_short_runs_whole():
    report = latency_report({stage: [0.002] * 10 for stage in STAGES}, source_fps=25.0)
    assert (report["warmup_frames_excluded"], report["frames_measured"]) == (0, 10)


def test_seeding_is_reproducible():
    seed_everything(7)
    first = (random.random(), float(np.random.rand()))
    seed_everything(7)
    assert (random.random(), float(np.random.rand())) == first


def test_annotation_draws_boxes_banner_and_weights_kind_without_touching_the_input():
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    track = Track(1, (20.0, 30.0, 80.0, 70.0), "car", 3, 3, True)
    incident = Incident(
        IncidentType.ABRUPT_STOP, (1,), 0, 1, 0.0, 0.1, (20.0, 30.0, 80.0, 70.0), 1.0
    )
    shown = annotate(image, [track], [incident], "coco_standin")
    assert (image == 0).all()
    assert tuple(shown[30, 40]) == (0, 0, 255)  # incident box edge is red, drawn over the track box
    assert shown[100:, :150].any()  # weights_kind text along the bottom edge


def test_abrupt_stop_end_to_end_matches_the_hand_computed_frame(abrupt_result):
    result = abrupt_result
    assert result.frames_processed == 51
    assert [(i.type, i.track_ids, i.frame_flag) for i in result.incidents] == [
        (IncidentType.ABRUPT_STOP, (1,), 39)
    ]
    files = {p.name for p in result.out_dir.iterdir()}
    assert files == {
        "incidents.csv",
        "tracks.csv",
        "annotated.mp4",
        "latency.json",
        "run_manifest.json",
    }
    (row,) = read_rows(result.out_dir / "incidents.csv")
    assert (row["incident_type"], row["track_ids"], row["frame_flag"]) == ("ABRUPT_STOP", "1", "39")
    assert (row["weights_kind"], row["weights_run_id"]) == ("coco_standin", "scripted")
    assert len(read_rows(result.out_dir / "tracks.csv")) > 100
    manifest = json.loads((result.out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["weights_kind"] == "coco_standin" and manifest["seed"] == 0
    assert json.loads((result.out_dir / "latency.json").read_text(encoding="utf-8"))["fps"] > 0


def test_on_frame_and_on_incident_hooks_are_called_correctly(tmp_path):
    """Optional progress hooks (added for the webapp dashboard, decision D-25); the CLI never
    passes them, so this only tests the additive behaviour, not a change to the default path."""
    frame_calls: list[tuple[int, int | None]] = []
    incident_calls: list[IncidentType] = []
    frames, detections = render_scene(abrupt_stop_paths())
    clip = write_video(tmp_path / "scene.mp4", frames, fps=10.0)
    result = Pipeline(cfg_with(tmp_path), ScriptedDetector(detections)).run(
        clip,
        on_frame=lambda done, total: frame_calls.append((done, total)),
        on_incident=lambda incident: incident_calls.append(incident.type),
    )
    assert frame_calls == [(n, 51) for n in range(1, 52)]
    assert incident_calls == [IncidentType.ABRUPT_STOP]
    assert len(result.incidents) == 1


def test_annotated_video_has_every_frame_at_the_source_size(abrupt_result):
    annotated = VideoIngestor(abrupt_result.out_dir / "annotated.mp4")
    assert (annotated.width, annotated.height) == (640, 480)
    assert len(list(annotated.frames())) == 51


def test_free_flow_produces_no_incidents_and_a_header_only_csv(tmp_path):
    paths = {vid: p for vid, p in abrupt_stop_paths().items() if vid != 1}
    result = run_scene(tmp_path, paths)
    assert result.incidents == []
    assert (result.out_dir / "incidents.csv").read_text(encoding="utf-8").splitlines() == [
        ",".join(INCIDENT_COLUMNS)
    ]


def test_same_clip_gives_the_same_incident_rows(abrupt_result, tmp_path):
    def key(result):
        return [
            (i.type, i.track_ids, i.frame_onset, i.frame_flag, i.score) for i in result.incidents
        ]

    assert key(run_scene(tmp_path, abrupt_stop_paths())) == key(abrupt_result)


def test_outputs_can_be_switched_off_in_config(tmp_path):
    cfg = cfg_with(tmp_path, annotated_video=False, tracks_csv=False)
    result = run_scene(tmp_path, abrupt_stop_paths(), cfg)
    assert {p.name for p in result.out_dir.iterdir()} == {
        "incidents.csv", "latency.json", "run_manifest.json",
    }  # fmt: skip


def test_lane_obstruction_end_to_end_with_a_shortened_hold_time(tmp_path):
    cfg = cfg_with(tmp_path)
    lane = dataclasses.replace(cfg.behaviour.lane_obstruction, min_stationary_s=3.0)
    cfg = dataclasses.replace(
        cfg, behaviour=dataclasses.replace(cfg.behaviour, lane_obstruction=lane)
    )
    paths = abrupt_stop_paths(n_frames=70, stop_frame=20)
    result = run_scene(tmp_path, paths, cfg)
    obstruction = [i for i in result.incidents if i.type is IncidentType.LANE_OBSTRUCTION]
    assert [i.track_ids for i in obstruction] == [(1,)]
    assert obstruction[0].t_flag_s - obstruction[0].t_onset_s == pytest.approx(3.0, abs=0.11)


def write_cli_config(tmp_path):
    data = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    data["output"]["dir"] = str(tmp_path / "reports")
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_cli_refuses_missing_video_stand_in_in_a_final_run_and_missing_weights(tmp_path, capsys):
    config = write_cli_config(tmp_path)
    clip = write_video(tmp_path / "c.mp4", render_scene(abrupt_stop_paths(3))[0], fps=10.0)
    assert cli.main(["run", str(tmp_path / "nope.mp4"), "--config", str(config), "--standin"]) == 2
    assert cli.main(["run", str(clip), "--config", str(config), "--standin", "--final"]) == 2
    assert cli.main(["run", str(clip), "--config", str(config)]) == 2
    err = capsys.readouterr().err
    assert "not allowed in a final run" in err and "weights contract violated" in err


@pytest.mark.standin
def test_cli_runs_the_real_standin_model_and_labels_the_output(tmp_path, capsys):
    weights = ROOT / "weights" / "coco_standin" / "yolov8n.pt"
    if not weights.is_file():
        pytest.skip("run scripts/fetch_standin_weights.py first")
    frames = render_scene(abrupt_stop_paths(12))[0]
    clip = write_video(tmp_path / "c.mp4", frames, fps=10.0)
    assert (
        cli.main(["run", str(clip), "--config", str(write_cli_config(tmp_path)), "--standin"]) == 0
    )
    (run_dir,) = (tmp_path / "reports").iterdir()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["weights_kind"] == "coco_standin"
    assert manifest["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert "incidents" in capsys.readouterr().out
