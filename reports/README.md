# reports/

Git-ignored output folder. Per-run artifacts written by the pipeline and evaluation scripts:

```
reports/<run_id>/incidents.csv        time-stamped incident log
reports/<run_id>/annotated.mp4        annotated video (ignored by git)
reports/<run_id>/eval_report.json     five-metric report with reproducibility manifest
reports/<run_id>/latency.json         per-stage timing, p50/p95, hardware string
```

Numbers that go into the report are copied into the evaluation log with the run ID.
