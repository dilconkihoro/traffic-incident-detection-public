"""The package imports without pulling in the model stack, and every skeleton class builds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from trafficwatch.behaviour import (
    AbruptStopRule,
    BehaviourAnalyzer,
    CollisionRule,
    LaneObstructionRule,
)
from trafficwatch.config import load_config
from trafficwatch.detection import VehicleDetector, WeightsError
from trafficwatch.incident_log import IncidentLogger
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


def test_pipeline_classes_construct_from_the_shipped_config(tmp_path):
    cfg = load_config(SHIPPED, env={})
    Preprocessor(cfg.preprocess, cfg.detector)
    VehicleTracker(cfg.tracker)
    TrajectoryStore(max_samples=10)
    IncidentLogger(
        tmp_path / "incidents.csv",
        "run",
        "video",
        rule_params_hash="h",
        weights_run_id="w",
        weights_kind="trained",
    ).close()
    BehaviourAnalyzer(cfg, period_s=0.1)
    for rule in (AbruptStopRule, LaneObstructionRule, CollisionRule):
        rule(cfg.behaviour, cfg.tracker.n_init)


def test_shipped_config_refuses_to_start_until_real_weights_exist():
    cfg = load_config(SHIPPED, env={})
    with pytest.raises(WeightsError, match="weights file not found"):
        VehicleDetector(cfg.detector)
