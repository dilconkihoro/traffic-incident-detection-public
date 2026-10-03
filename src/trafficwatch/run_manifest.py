"""Reproducibility manifest written with every run (project ML rule 5)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trafficwatch.config import PipelineConfig, behaviour_hash, config_hash
from trafficwatch.data.manifest import load_manifest, manifest_hash

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "opencv-python",
    "torch",
    "torchvision",
    "ultralytics",
    "deep-sort-realtime",
    "PyYAML",
)
DATASET_MANIFEST = Path("data/MANIFEST.json")


def git_state() -> dict[str, Any]:
    """Commit SHA and whether the working tree has uncommitted changes; None outside a repo."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gib() -> str:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal"):
                return f"{int(line.split()[1]) / 1024 / 1024:.0f} GiB RAM"
    except (OSError, ValueError):
        pass
    return "RAM unknown"


def hardware_string() -> str:
    gpu = "no GPU"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return " | ".join(
        [_cpu_model(), f"{os.cpu_count()} threads", _ram_gib(), gpu, platform.platform()]
    )


def dataset_manifest_hash() -> str | None:
    try:
        return manifest_hash(load_manifest(DATASET_MANIFEST))
    except Exception:  # noqa: BLE001 - absent or invalid manifest is recorded as unknown
        return None


def build_run_manifest(
    cfg: PipelineConfig,
    *,
    run_id: str,
    video_path: Path,
    weights_kind: str,
    weights_run_id: str,
    weights_sha256: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "video": {"path": str(video_path), "size_bytes": video_path.stat().st_size},
        "weights_kind": weights_kind,
        "weights_run_id": weights_run_id,
        "weights_sha256": weights_sha256,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "behaviour_hash": behaviour_hash(cfg.behaviour),
        "dataset_manifest_hash": dataset_manifest_hash(),
        **git_state(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "hardware": hardware_string(),
    }
