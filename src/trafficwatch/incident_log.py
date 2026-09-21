from __future__ import annotations

from pathlib import Path

from trafficwatch.types import Incident


class IncidentLogger:
    """Appends incidents to incidents.csv, flushing after each write."""

    def __init__(self, csv_path: str | Path, run_id: str, video_id: str) -> None:
        self.csv_path = Path(csv_path)
        self.run_id = run_id
        self.video_id = video_id

    def write(self, incident: Incident) -> None:
        raise NotImplementedError("Phase 2")

    def close(self) -> None:
        raise NotImplementedError("Phase 2")
