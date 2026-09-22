# 05. Evaluation and testing protocol

This file is the contract for every number in report section 5. The specification names five metrics and gives one-line definitions (1.3.2 v, 3.6); the operational definitions below make them computable and defensible. Items marked PROVISIONAL are ratified at the end of prototype cycle P1.

## 1. The core problem to solve first

UA-DETRAC and DhakaAI label **vehicles**, not **incidents**. Neither contains ground-truth abrupt stops, lane obstructions, or collisions. So false-alarm rate, incident precision, and incident recall cannot be read off the dataset labels, and the specification's wording ("compared to the expected incidents") needs a concrete meaning. Decision D-03 fixes it with a three-tier protocol. Get the reviewer to sign it off before Phase 5.

| Tier | Ground truth | What it proves | What it cannot claim |
|---|---|---|---|
| A. Oracle events | Run the same three rules on **ground-truth trajectories** (from the UA-DETRAC XML) to obtain rule-defined events. Compare pipeline events (detector plus tracker plus rules) against them. | How much detection and tracking error degrades rule outputs; per-rule precision and recall relative to the rule definition | Real-world collision recall |
| B. Adjudicated events | Two-pass manual review of pipeline flags on a fixed set of test clips: each flag labelled true event, false alarm, or ambiguous; plus a scan of the same clips for missed events. Record the review protocol and, if a second reviewer is available, agreement. | Real-footage precision of flags, false alarms per hour, qualitative failure modes | Recall on rare events not present in the clips |
| C. Synthetic scenarios | Hand-built trajectories with known ground truth (white-box) | Rule logic is correct at and around every threshold | Anything about real video |

Optional Tier D (crash clips from public collision datasets) is a scope change; do not do it without D-14 approval. State in report section 5 and 6 that real-collision recall is not established by this study.

### 1.1 Tier C in practice: `scripts/make_demo_clip.py` (added 2026-09-22)

Generates a short synthetic clip for a named scenario and runs it through the real `Pipeline`
class, but with `ScriptedDetector` (`src/trafficwatch/demo/scenarios.py`) standing in for YOLOv8:
it returns the exact ground-truth boxes for each frame instead of running detection. This proves
the tracker, the rule engine, and the output files (`incidents.csv`, `tracks.csv`,
`annotated.mp4`) are correct, and gives something to look at before real footage or a trained
model exist. It proves nothing about detection accuracy, and its numbers never enter a result
table, the same rule as `coco_standin` weights.

```bash
python scripts/make_demo_clip.py all        # or one of: abrupt_stop, free_flow, lane_obstruction, collision
```

| Scenario | Uses the real default thresholds | Expected result |
|---|---|---|
| `abrupt_stop` | Yes | One `ABRUPT_STOP` |
| `free_flow` | Yes | None (the clean-negative case) |
| `lane_obstruction` | Yes, `min_stationary_s: 15.0` in full, not shortened | Both `ABRUPT_STOP` and `LANE_OBSTRUCTION` (a stall is realistically a sudden stop that then persists) |
| `collision` | Yes | One `COLLISION_LIKE` |

Every scenario is covered end to end (not just at the rule level) in `tests/test_demo_scenarios.py`.

## 2. Metric definitions

### 2.1 Detector (component level)

| Metric | Definition | Tool |
|---|---|---|
| Precision, recall | At IoU 0.5 and a stated confidence operating point (report the point) | Ultralytics `val` |
| mAP50, mAP50:95 | Ultralytics COCO-style mAP over unified classes, `conf=0.001` | Ultralytics `val` |

Report on `val` (selection), on `test` once (final), on `dhaka_val` (cross-domain). Stratify by UA-DETRAC weather and night where the sequence attribute exists (specification 2.2 and the day-night gap). State plainly that these are Ultralytics mAP values, not the official DETRAC toolkit score, and are not leaderboard-comparable.

### 2.2 Tracker (supporting)

MOTA, IDF1, ID switches from `py-motmetrics` on val and test sequences with ground-truth tracks. Purpose: show whether tracking quality explains rule errors. Baseline check: same detections through ByteTrack (Ultralytics built-in), labelled as a sanity baseline only; DeepSORT stays the system tracker.

### 2.3 Incident level

- **Event matching:** a pipeline event matches a reference event if types are equal, involved tracks overlap (or, for Tier B, boxes overlap in time and space), and the flag time is within a tolerance window around the reference (default: from reference onset minus 1 s to reference flag plus 2 s, PROVISIONAL). One-to-one matching, greedy by time.
- **Precision** = matched pipeline events / all pipeline events. **Recall** = matched reference events / all reference events. Per rule and overall.
- **False-alarm rate (FAR)**, reported two ways: (a) **per evaluated sequence** = sequences with at least one unmatched flag divided by sequences evaluated (the specification's definition, made explicit); (b) **per hour of video** = unmatched flags divided by video hours. Report both; (b) is the operationally meaningful one.

### 2.4 Latency

| Quantity | Definition |
|---|---|
| Per-frame pipeline latency | Wall time from frame ingestion to the end of behaviour analysis for that frame, in ms, over **1,000 representative frames after 50 warm-up frames**; report mean, p50, p95, and the per-stage breakdown (ingestion, preprocessing, detection, tracking, behaviour). Include FPS. |
| Time-to-flag | Seconds from reference event onset to the flag. For lane obstruction report gross and net of `min_stationary_s`. |
| Hardware | Every latency number carries the machine string (CPU, GPU, RAM, OS, versions). Report Colab T4 and the lab machine separately. The lab machine is the author's development machine (D-22), CPU only. |

Measurement rules: `time.perf_counter`; `torch.cuda.synchronize()` before each clock read on GPU; single process, no other load; video decoded ahead of the timed region if decode is to be excluded, and say which.

## 3. Threshold calibration (Phase 4)

1. Fix the detector weights and `detector.conf` grid (for example 0.25 to 0.6).
2. Grid or random search over rule thresholds on **val** sequences only.
3. Objective: maximise incident recall subject to a false-alarm budget (PROVISIONAL: at most 1 unmatched flag per video hour on val), tie-break on precision.
4. Record the full trade-off table and curve (recall versus FAR). Freeze the chosen point in config, log the config hash.
5. No further changes after the freeze except through a logged decision and a full re-evaluation.

## 4. Test pyramid (specification 3.6)

| Level | Framework and scope | Examples (IDs use `TC-<level>-<n>`) |
|---|---|---|
| Unit | pytest per module | TC-UT-01 letterbox returns 640x640 and inverse-maps boxes; TC-UT-02 CSV row format; TC-UT-03 config validation rejects missing keys; TC-UT-04 abrupt-stop rule on a synthetic trajectory |
| White-box (rules) | Hand-computed trajectories | Positive, negative, and boundary case per threshold; each suppression path (queue, global slowdown, exclusion zone, occlusion versus collision) |
| Integration | Adjacent modules | TC-IT-01 detection boxes initialise tracks; TC-IT-02 trajectories feed the analyzer; TC-IT-03 weights and model-card mismatch aborts startup |
| Black-box system | Whole pipeline on clips; only inputs and outputs inspected | TC-BB-01 known stationary-vehicle clip produces LANE_OBSTRUCTION within tolerance; TC-BB-02 clean free-flow clip produces zero flags |
| Performance | Latency and FPS | TC-PF-01 1,000-frame benchmark on lab machine; TC-PF-02 on Colab GPU |
| Regression | Golden-file test | Fixed clip and fixed weights produce the same `incidents.csv` (hash), guarding refactors |

Stand-in weights are acceptable for unit, white-box, integration, and pipeline smoke tests; not for any reported metric.

## 5. report section 5.5 results table (institutional format)

| Test case | Description | Test data | Experimental outcome | Verdict |
|---|---|---|---|---|
| TC-UT-01 | Preprocessor letterboxes to 640x640 | Sample 1920x1080 frame | Output shape (640,640,3), boxes invert within 1 px | Pass |

Every row links to a pytest node ID or a run ID. Verdicts are Pass, Fail, or Fail with fix and re-test noted.

## 6. Acceptance gates (PROVISIONAL, ratify after P1)

| Gate | Provisional target | Basis |
|---|---|---|
| Detector mAP50 on `val` | At least 0.85 | In-domain fine-tune should beat COCO zero-shot by a wide margin; confirm from T0 and T1 |
| Detector recall at operating conf | At least 0.85 | Missed vehicles become missed incidents |
| False alarms | At most 1 per video hour on val at the frozen point | Operator trust (specification 2.4 gap four) |
| End-to-end speed | Real time at the source frame rate on Colab T4 (for example 25 FPS); lab machine reported as measured, no target | specification "near-real-time" requirement |
| Tests | 100 percent of unit, white-box, integration cases pass | Gate for G5 |

If a target is missed, report the result and analyse why (report section 5.5 and 6); do not move the goalposts after seeing test results.

## 7. Reproducibility manifest (stored in every `eval_report.json`)

git SHA and dirty flag, config hash, `behaviour` block hash, weights run ID and SHA-256, weights kind, dataset manifest hash, split manifest hash, seed, Python and package versions, hardware string, timestamp, command line.

## 8. Report outputs

`reports/<run_id>/eval_report.json` (machine), `reports/<run_id>/latency.json`, plots (PR curves, recall versus FAR, latency distribution, confusion matrix, per-weather bars) generated by a script from the JSON, saved as PNG into `docs/evidence/` with figure IDs. Summaries are copied into `docs/logs/EVAL_LOG.md` with run IDs.
