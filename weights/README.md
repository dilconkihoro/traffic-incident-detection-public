# weights/

Git-ignored except this file. The trained model is produced on Colab and added later.

```
weights/<RUN_ID>/best.pt
weights/<RUN_ID>/model_card.json
weights/<RUN_ID>/results.csv, args.yaml, curves/*.png
```

Contract, load-time validation, and the integration checklist: project rules section 6 and the Colab runbook section 9.

Until a trained run is dropped here, the pipeline may use COCO-pretrained `yolov8n.pt` in stand-in mode. Stand-in outputs are labelled `weights_kind: coco_standin` and excluded from final result tables.
