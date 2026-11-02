from __future__ import annotations

import importlib.metadata
from pathlib import Path

from fastapi import APIRouter, Request

from trafficwatch.config import ConfigError, load_config
from trafficwatch.detection import WeightsError, check_weights_contract

router = APIRouter(prefix="/api/models")


def trained_model_status(pipeline_config_path: Path) -> dict[str, object]:
    """Whether a run without stand-in weights would pass the weights contract, checked the same
    way the pipeline checks it at load (``TW_WEIGHTS`` or ``detector.weights``, then
    model_card.json)."""
    try:
        cfg = load_config(pipeline_config_path)
        card = check_weights_contract(cfg.detector, importlib.metadata.version("ultralytics"))
    except (ConfigError, WeightsError) as exc:
        return {"available": False, "run_id": None, "problem": str(exc)}
    if cfg.detector.weights_kind != "trained" or card is None:
        return {
            "available": False,
            "run_id": None,
            "problem": "detector.weights_kind is not 'trained' in the pipeline config",
        }
    return {"available": True, "run_id": card["run_id"], "problem": None}


@router.get("")
def list_models(request: Request) -> dict[str, object]:
    return {"trained": trained_model_status(request.app.state.settings.pipeline_config_path)}
