"""The package imports without pulling in the model stack, and every skeleton class builds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from trafficwatch.behaviour import (
    AbruptStopRule,
    BehaviourAnalyzer,
    CollisionRule,
    LaneObstructionRule,
)
from trafficwatch.config import load_config
from trafficwatch.detection import VehicleDetector
from trafficwatch.incident_log import IncidentLogger
from trafficwatch.ingestion import VideoIngestor
from trafficwatch.preprocessing import Preprocessor
from trafficwatch.tracking import TrajectoryStore, VehicleTracker

SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def test_import_does_not_load_model_libraries():
    code = (
        "import sys, trafficwatch, trafficwatch.behaviour, trafficwatch.config, "
        "trafficwatch.detection, trafficwatch.tracking, trafficwatch.data.manifest\n"
        "loaded = {'torch', 'ultralytics', 'cv2'} & set(sys.modules)\n"
        "assert not loaded, loaded"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_skeleton_classes_construct_from_the_shipped_config(tmp_path):
    cfg = load_config(SHIPPED, env={})
    VideoIngestor(tmp_path / "v.mp4", cfg.run.frame_stride)
    Preprocessor(cfg.preprocess, cfg.detector)
    VehicleDetector(cfg.detector)
    VehicleTracker(cfg.tracker)
    TrajectoryStore()
    IncidentLogger(tmp_path / "incidents.csv", "run", "video")
    BehaviourAnalyzer(cfg)
    for rule in (AbruptStopRule, LaneObstructionRule, CollisionRule):
        rule(cfg.behaviour, cfg.tracker.n_init)
