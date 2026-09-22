from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from trafficwatch.config import ConfigError, load_config
from trafficwatch.detection import WeightsError
from trafficwatch.ingestion import IngestError
from trafficwatch.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trafficwatch")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="process one video into incidents.csv and annotated.mp4")
    run.add_argument("video", type=Path)
    run.add_argument(
        "--config", type=Path, default=Path(os.environ.get("TW_CONFIG", "configs/pipeline.yaml"))
    )
    run.add_argument("--display", action="store_true", help="show an OpenCV window while running")
    run.add_argument("--final", action="store_true", help="refuse stand-in weights")
    run.add_argument("--standin", action="store_true", help="use the COCO stand-in weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, final=args.final, standin=args.standin)
        result = Pipeline(cfg).run(args.video, display=args.display)
    except (ConfigError, WeightsError, IngestError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = Counter(i.type.value for i in result.incidents)
    fps = result.latency["fps"]
    print(
        f"run {result.run_id}: {result.frames_processed} frames, {len(result.incidents)} incidents"
    )
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")
    print(f"  outputs: {result.out_dir}" + (f"  ({fps:.1f} FPS)" if fps else ""))
    return 0
