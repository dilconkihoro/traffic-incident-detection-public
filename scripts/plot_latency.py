"""Per-stage latency figure (docs/06 F5.11) from a run's latency.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latency_json", type=Path)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    args = parser.parse_args(argv)

    report = json.loads(args.latency_json.read_text(encoding="utf-8"))
    stages = [s for s, v in report["stages"].items() if v]
    means = [report["stages"][s]["mean_ms"] for s in stages]
    p95s = [report["stages"][s]["p95_ms"] for s in stages]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(stages))
    ax.bar([i - 0.2 for i in x], means, 0.4, label="mean")
    ax.bar([i + 0.2 for i in x], p95s, 0.4, label="p95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("milliseconds per frame")
    ax.set_title(
        f"F5.11: Per-stage latency, {report['frames_measured']} frames, "
        f"{report['fps']:.2f} FPS (CPU only)"
    )
    ax.legend()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    path = args.evidence_dir / "F5.11_latency_per_stage.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
