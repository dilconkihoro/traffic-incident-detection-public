"""Generate a Tier C synthetic demo clip and run it through the real pipeline.

Uses ScriptedDetector (docs/05 Tier C, decision D-03): ground-truth boxes are fed straight into
the real DeepSORT tracker and the real behaviour rules, so this proves tracking, rule logic, and
the output files are correct. It does NOT test detection accuracy: no YOLO model runs. Never cite
a number from this script's output as a measured result; it is a demonstration and a test fixture,
not an evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from trafficwatch.config import load_config
from trafficwatch.demo.scenarios import SCENARIOS, ScriptedDetector, render_scene, write_video
from trafficwatch.pipeline import Pipeline

DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def make_clip(name: str, out_dir: Path) -> Path:
    scenario = SCENARIOS[name]
    frames, detections = render_scene(scenario.paths, scenario.size)
    clip_path = out_dir / f"{name}.mp4"
    write_video(clip_path, frames, scenario.fps)
    return clip_path, detections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=[*SCENARIOS, "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=Path("data/demo_clips"))
    args = parser.parse_args(argv)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, standin=True)
    cfg = dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, dir="reports"))

    for name in names:
        scenario = SCENARIOS[name]
        clip_path, detections = make_clip(name, args.out_dir)
        result = Pipeline(cfg, ScriptedDetector(detections)).run(clip_path)
        kinds = sorted({i.type.value for i in result.incidents})
        print(f"{name} ({scenario.description}): {clip_path}")
        print(f"  {len(result.incidents)} incident(s): {', '.join(kinds) or 'none'}")
        print(f"  outputs: {result.out_dir}")
    print(
        "\nGround-truth-detection demo (ScriptedDetector), not a real AI detection result. "
        "Never cite these numbers as a measured outcome."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
