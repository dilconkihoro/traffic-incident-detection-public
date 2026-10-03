# Traffic Incident Detection (YOLOv8 + DeepSORT on CCTV video)

An automated prototype that reads a pre-recorded fixed-camera CCTV video, detects and tracks vehicles, flags three incident types with transparent rules, and writes a time-stamped incident log. Evaluated on precision, recall, mAP, false-alarm rate, and detection latency.

**Status:** Foundation, data preparation and the pipeline are built and tested on COCO stand-in weights. A zero-shot baseline is measured. Training on the project's own classes runs on Colab and is not part of this repository.

## What it does

```
video file -> ingestion -> preprocessing (640 letterbox) -> YOLOv8 detection -> DeepSORT tracking
           -> behaviour rules (abrupt stop | lane obstruction | collision-like) -> incidents.csv + annotated video
                                                            evaluation (separate) -> five-metric report
```

| Input | Output |
|---|---|
| Fixed-camera video file | `incidents.csv`, annotated video, latency report, evaluation report |

Data: UA-DETRAC and DhakaAI (public). Core scope is offline video files on a desktop; no live streams, plates, or pedestrians. A supplementary web dashboard (`webapp/`) sits on top of the same pipeline for demonstration. It calls the CLI pipeline and does not change the core scope or how it is evaluated. See `webapp/README.md`.

## Documentation

| File | Purpose |
|---|---|
| `docs/02_ARCHITECTURE_AND_RULES.md` | Modules, contracts, config, behaviour rule specifications, CSV schema, draft diagrams |
| `docs/03_DATA_PIPELINE.md` | Acquisition, YOLO conversion, class mapping, sequence-level splits, EDA |
| `docs/05_EVALUATION_AND_TESTING.md` | Metric definitions, three-tier incident ground truth, calibration, test pyramid |
| `docs/08_SETUP_AND_PORTABILITY.md` | Setting up a fresh checkout on another machine, Linux or Windows |
| `webapp/README.md` | The supplementary web dashboard: stack, setup, running it end to end |

## Process

1. **Foundation:** environment, package skeleton, decisions.
2. **Data:** acquire, verify, convert, split by sequence, exploratory analysis.
3. **Pipeline:** seven modules on COCO stand-in weights, rules with white-box tests.
4. **Training (Colab):** zero-shot baseline, then fine-tuned runs and model selection on validation.
5. **Integration:** trained weights, threshold calibration on validation, freeze.
6. **Evaluation:** five metrics, stratified results, latency, tests, one test-split evaluation.

## Quickstart

Linux/macOS shown; for Windows or a second machine see `docs/08_SETUP_AND_PORTABILITY.md`.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # local CPU pins (inference, tests, EDA); training runs on Colab
pip install -e . --no-deps
cp .env.example .env                     # then set your own values; never commit .env
pytest -m "not target" -q                # must be green
pytest -m target -q                      # white-box rule tests
ruff check . && ruff format --check .
```

`requirements.txt` is the local CPU install. `requirements.lock` is the Colab `pip freeze` of the run that produced the reported weights. No model training runs on the local machine.

Run the pipeline on a video with the COCO stand-in weights (output is labelled `coco_standin` and never enters result tables):

```bash
python scripts/fetch_standin_weights.py            # once; downloads yolov8n.pt, records its SHA-256
python -m trafficwatch run path/to/video.mp4 --standin
pytest -m standin -q                               # real-model smoke tests
```

Each run writes `reports/<run_id>/` with `incidents.csv`, `tracks.csv`, `annotated.mp4`, `latency.json` and `run_manifest.json`. `--final` refuses stand-in weights.

## Training on Colab

Local training is out of scope. `notebooks/colab_train.ipynb` runs on Colab with a GPU: it fetches this repository, installs `requirements-colab.txt`, stages the dataset from Google Drive, runs a smoke test, then the run matrix (yolov8n and yolov8s only), compares validation metrics, and exports each run to Drive. The notebook narrates each step.

## Adding a trained model

Drop the exported run into `weights/<RUN_ID>/` (needs `best.pt` and `model_card.json`), set `detector.weights` in `configs/pipeline.yaml`, set `weights_kind: trained`, and run the integration checks.

## Web dashboard (supplementary)

A browser UI (`webapp/`: FastAPI and SQLite backend, Vite, React, TypeScript and Tailwind frontend) on top of the CLI, with login, video upload, a live processing view, and a run history browser. Setup and running it end to end: `webapp/README.md`.

## Licence and data

The code is licensed under the GNU Affero General Public License v3.0 (see `LICENSE`), matching the AGPL-3.0 licence of Ultralytics YOLOv8, which this project builds on. Datasets are not redistributed; their sources, licences and hashes are recorded in `data/MANIFEST.json`.
