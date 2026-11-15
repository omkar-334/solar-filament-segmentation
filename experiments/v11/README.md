# Pipeline v11

The other packaged version of the pipeline (see `../README.md` for the comparison).

| File | Content |
|---|---|
| `report/report.pdf` | Four-page report of this version |
| `submission.csv` | The file this version produced, as submitted to Kaggle (public board 0.41) |
| `MANIFEST.json` | Names, sizes and SHA-256 hashes of its weight files (the files are not stored in this repository) |
| `rubric_metrics/` | Out-of-fold Dice, IoU and relation statistics |
| `predict_repro/` | Full test-set run of `scripts/predict.py` compared with `submission.csv` |
| `verify_release/` | Clean-environment check of this version, when it was run |
