# Traffic Incident Detection (YOLOv8 + DeepSORT on CCTV video)

Final-year project, the programme, the university. An automated prototype that reads a pre-recorded fixed-camera CCTV video, detects and tracks vehicles, flags three incident types with transparent rules, and writes a time-stamped incident log. Evaluated on precision, recall, mAP, false-alarm rate, and detection latency.

**Author:** the author, the author Kuria. **reviewer:** the reviewer. **Status:** planning complete, development starting (Phase 0).

## What it does

```
video file -> ingestion -> preprocessing (640 letterbox) -> YOLOv8 detection -> DeepSORT tracking
           -> behaviour rules (abrupt stop | lane obstruction | collision-like) -> incidents.csv + annotated video
                                                            evaluation (separate) -> five-metric report
```

| Input | Output |
|---|---|
| Fixed-camera video file | `incidents.csv`, annotated video, latency report, evaluation report |

Data: UA-DETRAC and DhakaAI (public). Scope is strictly offline video files on a desktop; no live streams, plates, pedestrians, or web app. Full scope guard: `the project rules` section 3.

## Documentation map

| File | Purpose |
|---|---|
| `the project rules` | Project rules for the development tool sessions (ML rules, Colab rules, weights contract, review checklist) |
| `docs/00_PROJECT_CONTEXT.md` | specification digest: objectives, scope, pipeline, data, limitations |
| `docs/01_MASTER_PLAN.md` | Phases, gates, schedule (Mermaid Gantt), human checkpoints |
| `docs/02_ARCHITECTURE_AND_RULES.md` | Modules, contracts, config, behaviour rule specs, CSV schema, draft diagrams |
| `docs/03_DATA_PIPELINE.md` | Acquisition, YOLO conversion, class mapping, sequence-level splits, EDA |
| `docs/04_COLAB_TRAINING.md` | Colab runbook, run matrix, notebook cells, weights export and integration checklist |
| `docs/05_EVALUATION_AND_TESTING.md` | Metric definitions, three-tier incident ground truth, calibration, test pyramid |
| `docs/06_DOCUMENTATION_PLAN.md` | report sections 4 to 6 map, institutional formatting, diagrams, evidence capture |
| `docs/07_DECISIONS_RISKS.md` | Decision log, risk register, specification defects |
| `docs/EXPERT_KICKOFF_PROMPT.md` | The prompt to start work in the development workspace, plus per-phase prompts |
| `docs/logs/` | Training, evaluation, implementation, and test-access logs |
| `docs/reference/` | The specification (docx and text copy) and institutional guides |

## Process in one screen

1. **Phase 0 Foundation:** environment, package skeleton, decisions.
2. **Phase 1 Data:** acquire, verify, convert, split by sequence, EDA.
3. **Phase 2 Pipeline:** seven modules on COCO stand-in weights, rules with white-box tests.
4. **Phase 3 Training (Colab):** T0 baseline, T1 to T3 runs, model selection on validation.
5. **Phase 4 Integration:** drop trained weights in, calibrate thresholds on validation, freeze.
6. **Phase 5 Evaluation:** five metrics, stratified results, latency, tests, one test-split evaluation.
7. **Phase 6 Documentation** (continuous) and **Phase 7 Demo.**

## Quickstart

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # local CPU pins (inference, tests, EDA); training happens on Colab
pip install -e . --no-deps
cp .env.example .env                     # then set your own values; never commit .env
pytest -m "not target" -q                # must be green
pytest -m target -q                      # white-box rule tests; red until Phase 2 by design
ruff check . && ruff format --check .
```

`requirements.txt` is the local CPU install. `requirements.lock` is the Colab `pip freeze` of the run that produced the reported weights and is added in Phase 3. No model training runs on the local machine.

The pipeline command (`python -m trafficwatch run ...`) arrives in Phase 2.

## Adding the trained model later

Drop the exported run into `weights/<RUN_ID>/` (needs `best.pt` and `model_card.json`), set `detector.weights` in `configs/pipeline.yaml`, set `weights_kind: trained`, and run the integration checklist in `docs/04_COLAB_TRAINING.md` section 9.

## Licence and data notes

Repository is private. Ultralytics YOLOv8 is AGPL-3.0; review before any public release. Datasets are not redistributed; sources, licences, and hashes are recorded in `data/MANIFEST.json`.
