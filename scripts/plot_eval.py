"""Plots from a `reports/<run_id>/eval_report.json` (docs/05 section 8): F5.6 (PR curve) and
F5.13 (precision/recall per weather condition), saved to docs/evidence/ with their figure IDs.
"""

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


def plot_pr_curve(report: dict, evidence_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for split in ("val", "dhaka_val"):
        if split not in report["splits"]:
            continue
        curves = report["splits"][split]["overall"].get("per_class_pr_curve", {})
        for cls_name, curve in curves.items():
            if not curve.get("recall"):
                continue
            ax.plot(
                curve["recall"], curve["precision"], label=f"{split}/{cls_name}",
                marker=".", markersize=3,
            )  # fmt: skip
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"F5.6: Precision-recall at IoU 0.5 ({report['manifest']['weights_kind']})")
    ax.legend()
    path = evidence_dir / "F5.6_pr_curve_t0.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_per_weather(report: dict, evidence_dir: Path) -> str | None:
    by_weather = report["splits"].get("val", {}).get("by_weather")
    if not by_weather:
        return None
    weathers = sorted(by_weather)
    precisions = [by_weather[w]["precision_at_conf"]["precision"] for w in weathers]
    recalls = [by_weather[w]["precision_at_conf"]["recall"] for w in weathers]
    x = range(len(weathers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], precisions, width, label="precision")
    ax.bar([i + width / 2 for i in x], recalls, width, label="recall")
    ax.set_xticks(list(x))
    ax.set_xticklabels(weathers)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"at conf={report['operating_conf']}")
    ax.set_title("F5.13: Precision/recall per weather condition (val, T0 zero-shot)")
    ax.legend()
    path = evidence_dir / "F5.13_precision_recall_per_weather.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_report", type=Path)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    args = parser.parse_args(argv)

    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    written = [plot_pr_curve(report, args.evidence_dir)]
    weather_plot = plot_per_weather(report, args.evidence_dir)
    if weather_plot:
        written.append(weather_plot)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
