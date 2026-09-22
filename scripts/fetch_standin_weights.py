"""Fetch the COCO-pretrained YOLOv8n stand-in weights and record their SHA-256."""

from __future__ import annotations

from pathlib import Path

from trafficwatch.data.manifest import sha256_file

TARGET = Path("weights/coco_standin/yolov8n.pt")


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(TARGET)
    if not TARGET.is_file():
        print(f"error: {TARGET} was not downloaded")
        return 1
    digest = sha256_file(TARGET)
    (TARGET.parent / "yolov8n.pt.sha256").write_text(f"{digest}  yolov8n.pt\n", encoding="utf-8")
    print(f"{TARGET} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
