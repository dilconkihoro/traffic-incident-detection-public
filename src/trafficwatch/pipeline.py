"""Orchestration: ingestion, preprocessing, detection, tracking, behaviour analysis, outputs."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from trafficwatch.behaviour.analyzer import BehaviourAnalyzer
from trafficwatch.config import PipelineConfig, behaviour_hash
from trafficwatch.incident_log import IncidentLogger, TrackLogger
from trafficwatch.ingestion import VideoIngestor
from trafficwatch.preprocessing import Preprocessor
from trafficwatch.run_manifest import build_run_manifest, hardware_string
from trafficwatch.tracking import VehicleTracker
from trafficwatch.types import Detection, Incident, Prepared, Track

STAGES = ("ingestion", "preprocessing", "detection", "tracking", "behaviour")
# Measurement protocol from docs/05 section 2.4: the first frames include model warm-up.
WARMUP_FRAMES = 50
# Display only: how long an incident overlay stays on the annotated video.
BANNER_S = 2.0
GREEN, RED, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)


class Detector(Protocol):
    weights_kind: str
    weights_run_id: str
    weights_sha256: str

    def detect(self, prepared: Prepared) -> list[Detection]: ...


@dataclass(frozen=True)
class RunResult:
    run_id: str
    out_dir: Path
    incidents: list[Incident]
    frames_processed: int
    latency: dict[str, Any]


class _Clock:
    """perf_counter that first waits for queued GPU work, so GPU stages are timed correctly."""

    def __init__(self) -> None:
        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self._torch = None

    def now(self) -> float:
        if self._torch is not None:
            self._torch.cuda.synchronize()
        return time.perf_counter()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def latency_report(samples: dict[str, list[float]], source_fps: float) -> dict[str, Any]:
    """Per-stage and total milliseconds after excluding warm-up frames (when there are enough)."""
    n = len(next(iter(samples.values()), []))
    skip = WARMUP_FRAMES if n > WARMUP_FRAMES else 0
    total = np.sum([samples[s] for s in STAGES], axis=0)[skip:] * 1000.0

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }

    report: dict[str, Any] = {
        "frames_measured": int(total.size),
        "warmup_frames_excluded": skip,
        "source_fps": source_fps,
        "total": stats(total) if total.size else None,
        "fps": float(1000.0 / np.mean(total)) if total.size else None,
        "stages": {
            s: stats(np.asarray(samples[s][skip:]) * 1000.0) if total.size else None for s in STAGES
        },
        "hardware": hardware_string(),
    }
    return report


def annotate(
    image: np.ndarray, tracks: list[Track], banners: list[Incident], weights_kind: str
) -> np.ndarray:
    out = image.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(
            out, f"#{track.track_id} {track.cls_name}", (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA,
        )  # fmt: skip
    for row, incident in enumerate(banners):
        x1, y1, x2, y2 = (int(v) for v in incident.bbox_union)
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(
            out, incident.type.value, (10, 24 + 24 * row),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA,
        )  # fmt: skip
    cv2.putText(
        out, f"weights_kind: {weights_kind}", (10, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
    )  # fmt: skip
    return out


class Pipeline:
    """Runs one video through the seven-module chain and writes the run directory."""

    def __init__(self, cfg: PipelineConfig, detector: Detector | None = None) -> None:
        self.cfg = cfg
        if detector is None:
            from trafficwatch.detection import VehicleDetector

            detector = VehicleDetector(cfg.detector)
        self.detector = detector

    def run(self, video: str | Path, *, display: bool = False) -> RunResult:
        cfg, video = self.cfg, Path(video)
        seed_everything(cfg.run.seed)
        ingestor = VideoIngestor(video, cfg.run.frame_stride)
        period_s = cfg.run.frame_stride / ingestor.fps
        preprocessor = Preprocessor(cfg.preprocess, cfg.detector)
        tracker = VehicleTracker(cfg.tracker)
        analyzer = BehaviourAnalyzer(cfg, period_s)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{self.detector.weights_run_id}_{video.stem}"
        out_dir = Path(cfg.output.dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=False)
        manifest = build_run_manifest(
            cfg,
            run_id=run_id,
            video_path=video,
            weights_kind=self.detector.weights_kind,
            weights_run_id=self.detector.weights_run_id,
            weights_sha256=self.detector.weights_sha256,
            seed=cfg.run.seed,
        )
        (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        incident_log = IncidentLogger(
            out_dir / "incidents.csv",
            run_id,
            video.stem,
            rule_params_hash=behaviour_hash(cfg.behaviour),
            weights_run_id=self.detector.weights_run_id,
            weights_kind=self.detector.weights_kind,
        )
        track_log = (
            TrackLogger(out_dir / "tracks.csv", video.stem) if cfg.output.tracks_csv else None
        )
        writer: cv2.VideoWriter | None = None
        samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
        incidents: list[Incident] = []
        banners: list[Incident] = []
        clock, frames = _Clock(), iter(ingestor.frames())
        try:
            while True:
                t0 = clock.now()
                frame = next(frames, None)
                if frame is None:
                    break
                t1 = clock.now()
                prepared = preprocessor.apply(frame)
                t2 = clock.now()
                detections = self.detector.detect(prepared)
                t3 = clock.now()
                tracks = tracker.update(detections, frame)
                t4 = clock.now()
                found = analyzer.step(tracks, frame.idx, frame.t_s)
                t5 = clock.now()
                marks = (t0, t1, t2, t3, t4, t5)
                for stage, start, end in zip(STAGES, marks[:-1], marks[1:], strict=True):
                    samples[stage].append(end - start)

                for incident in found:
                    incident_log.write(incident)
                    incidents.append(incident)
                if track_log is not None:
                    for track in tracks:
                        track_log.write(track, frame.idx, frame.t_s)
                banners = [i for i in [*banners, *found] if frame.t_s - i.t_flag_s <= BANNER_S]
                if cfg.output.annotated_video or display:
                    shown = annotate(frame.image, tracks, banners, self.detector.weights_kind)
                    if cfg.output.annotated_video:
                        writer = writer or self._open_writer(out_dir, ingestor, shown)
                        writer.write(shown)
                    if display:
                        cv2.imshow("trafficwatch", shown)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            incident_log.close()
            if track_log is not None:
                track_log.close()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        latency = latency_report(samples, ingestor.fps)
        (out_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
        return RunResult(run_id, out_dir, incidents, len(samples[STAGES[0]]), latency)

    def _open_writer(self, out_dir: Path, ingestor: VideoIngestor, sample: np.ndarray):
        height, width = sample.shape[:2]
        fps = ingestor.fps / self.cfg.run.frame_stride
        path = out_dir / "annotated.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open a video writer for {path}")
        return writer
