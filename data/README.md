# data/

Git-ignored except this file and `MANIFEST.json`. Layout and rules: `docs/03_DATA_PIPELINE.md`.

```
data/
  MANIFEST.json            tracked: source, DOI/URL, licence, retrieval date, SHA-256 per archive
  raw/                     immutable originals (UA-DETRAC, DhakaAI)
  interim/                 extracted frames, parsed annotations
  processed/yolo_ds_vN/    YOLO-format dataset + data.yaml + stats.json (this is what Colab receives as one .tar)
  clips/                   short evaluation and demo clips cut from test sequences
```
