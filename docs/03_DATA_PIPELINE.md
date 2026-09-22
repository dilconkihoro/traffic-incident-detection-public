# 03. Data pipeline (acquisition to Colab-ready dataset)

Follows a seven-step ML process: collection, EDA, cleaning, preparation, training, evaluation, deployment. This file covers steps 1 to 4. Each step produces an artifact and a written insight for report section 5.3.

Items marked **VERIFY** are recalled facts that must be checked against the downloaded release before they are written into any section.

## 1. Step map

| Step | Script (to build) | Output artifact | Insights to record |
|---|---|---|---|
| 1 Acquisition | `scripts/acquire_datasets.py` (verifies files, hashes, writes manifest; downloads only where a stable source exists) | `data/raw/*`, `data/MANIFEST.json` | Sources, licences, sizes, SHA-256, retrieval date |
| 2 EDA | `scripts/eda_dataset.py` | `data/processed/*/eda_insights.json`, plots | Sequences, frames, boxes per class, boxes per frame, box-size distribution, weather and night share, ignored-region share, class imbalance |
| 3 Convert and split | `scripts/convert_to_yolo.py` | `data/processed/yolo_ds_vN/`, `stats.json` | Dropped items and reasons, per-split counts, class distribution per split, leakage check result |
| 4 Preparation | `configs/data.yaml` template plus training args | `data.yaml`, `class_map.json` | Taxonomy, image size, augmentations, stride |

## 2. Source facts (checked live 2026-09-22; still VERIFY where noted, at acquisition)

| Item | UA-DETRAC | DhakaAI |
|---|---|---|
| Citation | Wen et al. (2020) | Shihavuddin and Rashid (2020), DOI 10.7910/DVN/POREXF |
| Content | 100 sequences of fixed-camera traffic video, frames as JPEG, per-frame XML annotations with track IDs (VERIFY on the real files, Phase 1 step 1.0) | **Confirmed 2026-09-22 (`train.rar` opened, all 3,003 annotation files parsed):** still images only, not sequences, not fixed-camera. Pascal VOC XML, one file per image (`<object><name>...<bndbox><xmin><ymin><xmax><ymax></bndbox></object>`), image `<width>`/`<height>` embedded. Image sizes are highly variable (428 distinct sizes across 3,003 images; common ones 1920x1080, 1280x720, and several phone-camera resolutions up to 4160x3120, both landscape and portrait) and file paths inside the XML (`E:\Datasets\Dataset\Images\...`) point at a Windows folder on the depositor's machine: this is a curated, crowd-sourced photo collection, not frames pulled from a fixed camera. Confirms P-03 |
| Official split | Train sequences and test sequences (expected 60 and 40, VERIFY) | **Confirmed 2026-09-22:** `train.rar` is `Final Train Dataset/`, 3,003 images with matching XML labels (one file, `231.xml`, is not actually XML: it holds YOLO-format label lines, `class_id x y w h` normalised, under a `.xml` name, an upstream data error the converter must detect and skip, not assume away). `test1.rar` is 500 images, **zero annotation files**, Roboflow-style export filenames (`<name>_jpg.rf.<hash>.jpg`). Read together with the Dataverse subtitle ("an international AI-based Dhaka Traffic Detection Challenge"), `test1.rar` is the competition's unlabeled holdout: it has no usable ground truth and is excluded from the pool this project splits. The usable pool is the 3,002 labelled train images |
| Classes | car, bus, van, others (expected, VERIFY) | **Confirmed 2026-09-22, full census over all 3,002 parseable train annotations, 24,368 boxes total, mean 8.12 per image:** 21 distinct classes, exact strings and counts in section 4 below |
| Extras | Sequence attributes (weather such as sunny, cloudy, rainy, night), ignored regions, occlusion and truncation flags (VERIFY) | Per-object `<truncated>` and `<difficult>` flags present (both 0/1); no weather or ignored-region metadata (not applicable to still images from mixed sources) |
| Licence | VERIFY. No terms are shown on the live project page (see Availability); record whatever Google Drive shows when the folder is opened | Confirmed: **CC0 1.0** (`http://creativecommons.org/publicdomain/zero/1.0`), read from the dataset's own Dataverse licence metadata |
| Availability | The original host (`detrac-db.rit.albany.edu`) is confirmed dead: every published download URL now redirects (HTTP 200) to a generic university lab page with no dataset link. The live replacement is a co-author's project page, `https://sites.google.com/view/daweidu/projects/ua-detrac`, with Google Drive links for DETRAC-Images, DETRAC-Annos, DETRAC-Detections, and DETRAC-Toolkits. **Manual**: Google Drive throttles or blocks scripted bulk downloads of files this large, and no licence text is exposed on that page to capture automatically. A Kaggle mirror also exists but needs the author's own Kaggle account and API key. | Confirmed public and unauthenticated: `fileAccessRequest: false` on the dataset's own metadata, no account, login, or request step. **Scripted**: `scripts/acquire_datasets.py fetch-dhakaai` downloads both files from Harvard Dataverse's documented API and records the manifest entry automatically |

Record all of the above in `data/MANIFEST.json`. If DhakaAI turns out to be images only, it feeds detector training and cross-domain detector evaluation only, and cannot be used for tracking or incident evaluation (log as a scope note; specification 3.2.3 says "sequences").

## 3. Acquisition rules

1. Raw archives are immutable. Hash on arrival; never edit in place.
2. Do not commit data. Commit only the manifest.
3. Prefer the official source; if a mirror is used, record it and compare counts against the citation (140,000 frames, 1.21 million boxes, 100 sequences).
4. Check disk before starting: a full UA-DETRAC download plus extracted frames plus the strided YOLO copy needs tens of GB. Measure and record.
5. No scraping of other footage (scope guard).

## 4. Conversion to YOLO format

- **Annotation parsing (UA-DETRAC):** per sequence XML to per-frame label files: `class x_center y_center width height`, normalised to 0..1.
- **Ignored regions:** the protocol marks regions that are not annotated. Mask them (fill with a constant colour in the training copy) so unlabelled vehicles there are not learned as background. Log how many pixels or boxes this affects.
- **Frame stride:** default every 5th frame (`D-06`), keeping roughly 28,000 training candidates from 140,000 frames. Adjacent frames are near-identical, so stride buys speed and Colab feasibility at little accuracy cost. EDA decides the final value; log it.
- **Class taxonomy (D-04, draft, decide after EDA):**

DhakaAI's real class strings and box counts (confirmed 2026-09-22, full census, all 3,002 parseable
train annotations, `train.rar`, exact spelling and spacing as they appear in the XML):

| Class (exact string) | Boxes | Draft unified class |
|---|---|---|
| `car` | 5476 | car |
| `rickshaw` | 3549 | VERIFY, see note below |
| `bus` | 3340 | bus |
| `three wheelers (CNG)` | 2990 | VERIFY, see note below |
| `motorbike` | 2284 | two_wheeler |
| `truck` | 1492 | other |
| `pickup` | 1225 | other |
| `minivan` | 935 | van |
| `suv` | 860 | car |
| `van` | 756 | van |
| `bicycle` | 459 | VERIFY, see note below |
| `auto rickshaw` | 372 | VERIFY, see note below |
| `human hauler` | 169 | other |
| `wheelbarrow` | 120 | VERIFY, see note below |
| `minibus` | 95 | bus |
| `ambulance` | 60 | not in the doc 02 draft taxonomy, VERIFY |
| `taxi` | 60 | car |
| `army vehicle` | 43 | other |
| `scooter` | 38 | two_wheeler |
| `policecar` | 32 | car (one word, not "police car") |
| `garbagevan` | 3 | other (one word, not "garbage van") |

Two open items the census surfaces that the earlier draft (names only, no counts) could not:

1. **Three overlapping three/two-wheeler classes** (`rickshaw` 3549, `three wheelers (CNG)` 2990,
   `auto rickshaw` 372) where the earlier draft assumed one. `rickshaw` in Dhaka is typically
   pedal-powered (closer to `bicycle`), `three wheelers (CNG)` and `auto rickshaw` are both
   motorised three-wheelers and may be the same vehicle type labelled inconsistently by different
   annotators; this needs a human look at a sample of each class's images before deciding whether
   to merge, keep separate, or drop, not a guess from the name alone.
2. **`ambulance` (60 boxes) is a real class DhakaAI labels that was not in the original draft
   taxonomy at all.** Emergency-vehicle detection is not out of scope (only emergency
   *notification* is, specification 1.6); mapping it to `other` is the default unless the author wants
   it separate.

  Label-noise warning: if UA-DETRAC leaves two-wheelers unlabelled while DhakaAI labels them, mixing teaches conflicting definitions. Mitigation candidates: keep two-wheelers out of the taxonomy, or collapse to a class-agnostic `vehicle` detector for the incident pipeline. Decide from EDA, log the reasoning. UA-DETRAC's own classes are still VERIFY (not yet acquired).
- **Stand-in mapping (COCO):** car(2) to car, motorcycle(3) to two_wheeler, bus(5) to bus, truck(7) to other. Used only for T0 and the P0 skeleton.

## 5. Splits

| Split | Rule |
|---|---|
| Test | Official UA-DETRAC test sequences. Frozen. Never used for training or tuning. |
| Train | Official train sequences minus validation sequences. |
| Val | About 10 to 12 official train sequences chosen by fixed seed, stratified across weather and day or night. Split by **sequence**. |
| DhakaAI | Image-level 80/10/10 with near-duplicate check (perceptual hash) so near-identical frames do not straddle splits. Holds a separate `dhaka_val` for cross-domain evaluation. |

Leakage check (script must fail the build if violated): no sequence ID in more than one split; no DhakaAI perceptual-hash collision across splits; test sequence list matches the official list.

## 6. Dataset artifact for Colab

```
data/processed/yolo_ds_v1/
  images/{train,val,test,dhaka_train,dhaka_val}/...
  labels/{train,val,test,dhaka_train,dhaka_val}/...
  data.yaml           # path-relative; rewritten on Colab to /content/data/yolo_ds_v1
  class_map.json
  stats.json          # counts before and after, drops with reasons, per-split class distribution
  split_manifest.json # sequence IDs per split, seed, stride, git SHA
```

Package as one `.tar`, compute SHA-256, copy to Drive under `tid/data/`. Record the hash in `MANIFEST.json`; every training run records this hash.

`data.yaml` template:

```yaml
path: /content/data/yolo_ds_v1
train: images/train
val: images/val
test: images/test
names:
  0: car
  1: van
  2: bus
  3: other
  4: two_wheeler
```

For T3 (UA-DETRAC plus DhakaAI), `train` becomes a list `[images/train, images/dhaka_train]` (the converter writes DhakaAI's training split into `images/dhaka_train` and `labels/dhaka_train`, same YOLO label format); `val` stays the UA-DETRAC val so runs remain comparable, and `dhaka_val` is evaluated separately.

## 7. Incident evaluation assets (built in Phase 5 prep, see doc 05)

- `data/clips/`: short clips cut from val and test sequences for black-box tests and the demo.
- Ground-truth trajectory files exported from the XML for oracle-event generation and tracker evaluation.
- Adjudication sheet template for manual review of flagged events (Tier B).

## 8. Insights checklist for report section 5.3

Sources and licence; retrieval method; sizes; sequences and frames per split; boxes per class; box-size distribution (small and distant vehicles); weather and night share; ignored-region handling; stride and why; class mapping and label-noise decision; leakage check outcome; before and after counts.

## 9. Phase 1 work plan (approved in session 1, executed next)

Real files are inspected before any parser is written; every VERIFY item above is corrected from them.

| Step | Deliverable | Detail |
|---|---|---|
| 1.0 | Source facts | Author places archives in `data/raw/`; `scripts/acquire_datasets.py verify` records hashes; one real XML and the directory listing settle split sizes, class names, weather attribute, ignored-region format, two-wheeler labelling, DhakaAI images versus sequences |
| 1.1 | `data/ua_detrac.py` | XML to typed dataclasses; fail closed on unknown structure; tested on a fixture reproduced from a real file |
| 1.2 | `scripts/eda_dataset.py` | `eda_insights.json` and plots for every item in section 8; figures to `docs/evidence/` with figure IDs |
| 1.3 | `configs/data.yaml` | Stride (default 5), val sequence count, seed, class map path, mask colour, DhakaAI ratios, pHash threshold; validated fail closed |
| 1.4 | `scripts/convert_to_yolo.py` | XML to YOLO labels, stride, ignored-region masking (log affected share and dropped boxes), unmapped source class is an error, versioned `class_map.json`, ground-truth tracks exported to `data/interim/gt_tracks/` |
| 1.5 | `data/split.py` | Val by seeded, weather-and-day-night stratified choice of whole sequences from official train; DhakaAI near-duplicates clustered by perceptual hash before assignment |
| 1.6 | Leakage check | Violations: a sequence in two splits, test list differs from official, DhakaAI cross-split pair within the pHash threshold. Converter exits non-zero and writes no tar on any violation; unit-tested with leaky fixtures |
| 1.7 | Colab artifact | Layout in section 6; deterministic tar (sorted entries, zeroed mtime and owner); `yolo_ds_v1.sha256` as `<hash>  yolo_ds_v1.tar`; hash recorded in `MANIFEST.json`; author copies both to Drive `tid/data/` |
| 1.8 | Gate G1 report | Manifest strict-clean, dataset built, stats and EDA logged, class map versioned, leakage green. Human checkpoint: D-04, D-06, DhakaAI usage |

Pre-committed D-04 rule: if two-wheelers are unlabelled in UA-DETRAC, v1 has four classes (`car, van, bus, other`) and T3 tests a class-agnostic `vehicle` variant instead of mixing conflicting definitions.

### Acquisition checklist (author)

**DhakaAI (scripted, no account needed):**

```bash
.venv/bin/python scripts/acquire_datasets.py fetch-dhakaai --dry-run   # lists files and licence first
.venv/bin/python scripts/acquire_datasets.py fetch-dhakaai --retrieved <YYYY-MM-DD>
```

Downloads `test1.rar` (52 MB) and `train.rar` (1.44 GB) to `data/raw/dhakaai/` and records both in `data/MANIFEST.json` with the confirmed CC0 1.0 licence and the DOI as source. About 1.5 GB total; run it when you have time and bandwidth for it.

**UA-DETRAC (manual, author only):**

1. `mkdir -p data/raw/ua-detrac`.
2. Open `https://sites.google.com/view/daweidu/projects/ua-detrac` and download DETRAC-Images and DETRAC-Annos (and DETRAC-Detections and DETRAC-Toolkits if useful) from the Google Drive links there into `data/raw/ua-detrac/`. Copy whatever licence or terms text Drive or the page shows when you open it; if none is shown, write `VERIFY` and say so.
3. Alternative: a Kaggle mirror exists (`kaggle.com/datasets/bratjay/ua-detrac-orig` or similar); needs your own `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env` (never shared with me) and the `kaggle` CLI.
4. Do not extract yet. For each archive: `.venv/bin/python scripts/acquire_datasets.py verify <archive> --dataset ua-detrac --retrieval <manual|mirror> --source-url <url> --licence "<terms>" --retrieved <YYYY-MM-DD>`.
5. `.venv/bin/python scripts/acquire_datasets.py check --strict` must exit 0 before gate G1.

