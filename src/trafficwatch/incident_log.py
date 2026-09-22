from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import IO

from trafficwatch.types import Incident, Track

INCIDENT_COLUMNS = (
    "incident_id",
    "run_id",
    "video_id",
    "incident_type",
    "track_ids",
    "frame_onset",
    "frame_flag",
    "t_onset_s",
    "t_flag_s",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "rule_params_hash",
    "weights_run_id",
    "weights_kind",
)
TRACK_COLUMNS = ("video_id", "frame_idx", "t_s", "track_id", "cls_name", "x1", "y1", "x2", "y2")


class _CsvWriter:
    def __init__(self, path: Path, columns: tuple[str, ...]) -> None:
        self._handle: IO[str] = open(path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.writer(self._handle, lineterminator="\n")
        self._writer.writerow(columns)
        self._handle.flush()

    def _write(self, row: list[object]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class IncidentLogger(_CsvWriter):
    """Appends incidents to incidents.csv, flushing after each write."""

    def __init__(
        self,
        csv_path: str | Path,
        run_id: str,
        video_id: str,
        *,
        rule_params_hash: str,
        weights_run_id: str,
        weights_kind: str,
    ) -> None:
        super().__init__(Path(csv_path), INCIDENT_COLUMNS)
        self._constants = (run_id, video_id)
        self._provenance = (rule_params_hash, weights_run_id, weights_kind)
        self._next_id = 1

    def write(self, incident: Incident) -> None:
        x1, y1, x2, y2 = incident.bbox_union
        run_id, video_id = self._constants
        self._write(
            [
                self._next_id,
                run_id,
                video_id,
                incident.type.value,
                ";".join(str(t) for t in incident.track_ids),
                incident.frame_onset,
                incident.frame_flag,
                f"{incident.t_onset_s:.3f}",
                f"{incident.t_flag_s:.3f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
                f"{incident.score:.3f}",
                *self._provenance,
            ]
        )
        self._next_id += 1


class TrackLogger(_CsvWriter):
    """Optional tracks.csv used by tracker evaluation and white-box replay."""

    def __init__(self, csv_path: str | Path, video_id: str) -> None:
        super().__init__(Path(csv_path), TRACK_COLUMNS)
        self._video_id = video_id

    def write(self, track: Track, frame_idx: int, t_s: float) -> None:
        x1, y1, x2, y2 = track.xyxy
        self._write(
            [
                self._video_id,
                frame_idx,
                f"{t_s:.3f}",
                track.track_id,
                track.cls_name,
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
            ]
        )
