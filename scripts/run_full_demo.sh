#!/usr/bin/env bash
# A guided, narrated walkthrough of the whole pipeline: run it top to bottom and read along.
# Two parts:
#   1. The real AI model (YOLOv8n stand-in) on a real UA-DETRAC video.
#   2. The Tier C synthetic scenarios (known ground truth, tests the rule engine and tracker).
# See docs/PROJECT_GUIDE.md for what each stage means, and docs/05_EVALUATION_AND_TESTING.md
# section 1.1 for why Tier C uses ground-truth boxes instead of the real model.
set -euo pipefail
cd "$(dirname "$0")/.."

pause() { echo; read -r -p "Press Enter to continue... " _ || true; echo; }
say() { echo; echo "=== $* ==="; echo; }

say "0. Checking the environment"
echo "This uses the project's own virtual environment (.venv), not the system Python."
if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv not found. Run: python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
PY=.venv/bin/python
"$PY" --version
pause

say "1. Making sure the stand-in AI model is downloaded"
echo "This is YOLOv8n exactly as Ultralytics released it (not retrained on our datasets yet)."
echo "Every output made with it is labelled weights_kind: coco_standin and is never a measured result."
if [ ! -f weights/coco_standin/yolov8n.pt ]; then
  "$PY" scripts/fetch_standin_weights.py
else
  echo "already present: weights/coco_standin/yolov8n.pt"
fi
pause

say "2. Running the real pipeline on a real UA-DETRAC video"
echo "data/samples/MVI_20011_short.mp4 is the first 75 real frames (3 seconds at the dataset's"
echo "native 25 fps) of UA-DETRAC's own MVI_20011 sequence, assembled from the images you"
echo "downloaded (960x540, the dataset's native resolution). This is genuine footage, going"
echo "through the real detector, the real tracker, and the real rules."
echo
echo "Takes a minute or two without a GPU. Do not close the terminal or press Ctrl+C while it"
echo "runs, or the output video will be left incomplete and unplayable (it needs to finish to"
echo "close the file properly). The full, un-trimmed sequence (data/samples/MVI_20011.mp4, 664"
echo "frames) is also here if you want to run it yourself later; on CPU alone it is slow enough"
echo "that it is not part of this walkthrough."
if [ ! -f data/samples/MVI_20011_short.mp4 ]; then
  echo "ERROR: data/samples/MVI_20011_short.mp4 not found. Ask the assistant to extract another sequence,"
  echo "or point this script at any other video: $PY -m trafficwatch run <your_video.mp4> --standin"
  exit 1
fi
"$PY" -m trafficwatch run data/samples/MVI_20011_short.mp4 --standin
echo
echo "That printed how many vehicles were detected and tracked, and any incidents flagged."
echo "Look for the newest folder under reports/ (the run ID is a timestamp) and open its"
echo "annotated.mp4 in a video player: it will show boxes drawn on real traffic."
pause

say "3. Running the four Tier C scenarios (known ground truth, proves the rule engine)"
echo "These use ground-truth boxes, not the AI model, on purpose: they prove the tracker and the"
echo "three rules are correct, independent of detection accuracy. See docs/05 section 1.1."
"$PY" scripts/make_demo_clip.py all
pause

say "Done"
echo "What you just saw, in order:"
echo "  1. The real YOLOv8n stand-in model detecting real vehicles in real UA-DETRAC footage."
echo "  2. All three behaviour rules firing correctly on known synthetic scenarios."
echo
echo "Every output is under reports/<run_id>/: incidents.csv, annotated.mp4, latency.json,"
echo "run_manifest.json. Nothing here is a measured result yet (the model is not fine-tuned and"
echo "the synthetic clips are not real evaluation data); it demonstrates that the system works."
