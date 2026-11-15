# Experiments

Everything that was tried on the way to the submitted pipeline, with the numbers and the logs. The submission itself is the repository root (`README.md`, `notebooks/`, `weights/`, `report/`). Paths in the documents of this folder are relative to the repository root.

| Document | Content |
|---|---|
| `REPORT.md` | Long-form method report: data facts, pipeline, validation protocol, all metrics, error analysis, table of rejected and adopted experiments, reproduction, limitations |
| `CHANGELOG.md` | Dated log of every experiment in the order it was run, with commands, numbers and decisions |
| `MODEL_EXPERIMENTS.md` | The first round (semantic U-Net baselines and thresholds), before the instance pipeline |
| `LITERATURE_REVIEW.md` | Papers and public solutions that were read, and what was taken from them |
| `v11/` | The heavier SAM 2.1 variant of the pipeline: its own four-page report, the file it produced, its weight manifest and its reproduction log |

## How the score was built

All numbers are out-of-fold Panoptic Quality on the same 707 training images (five folds grouped by image, 1,154 annotator views), computed like the competition scorer.

| Version | Change | Out-of-fold PQ | Public board |
|---|---|---|---|
| Semantic U-Net, connected components | first baseline | 0.34 (141 held-out images) | |
| Fixed rule | one YOLO11s-seg detector, crop refiner, detector confidence 0.35 | 0.4290 | 0.37 |
| v3 | learned selector (expected matched IoU over annotators) | 0.4547 | 0.40 |
| v5 | two averaged crop refiners | 0.4583 | 0.40 |
| v6 | candidates of a second detector at 1536 px | 0.4596 | 0.41 |
| v7 (submitted) | stage 2: ResNet34 U-Net on rotation-normalised crops | 0.4626 | 0.41 |
| v9 | stage 2 replaced by fine-tuned SAM 2.1 small and base-plus | 0.4695 | 0.41 |
| v10 | SAM 2.1 base-plus and large | 0.4704 | 0.41 |
| v11 | v10 masks, keep rule uses the fine-tuned SAM mask-quality score | 0.4725 | 0.41 |

## Why v7 is the submission and not v11

| | v7 | v11 |
|---|---|---|
| Out-of-fold PQ (95% bootstrap interval over images) | 0.4626 (0.454 to 0.471) | 0.4725 (0.464 to 0.481) |
| Mean Dice per annotator view | 0.655 | 0.667 |
| Public board | 0.41 | 0.41 |
| GPU time per image | 1.0 s | 11 s |
| Weight files | 12 (1.4 GB) | 22 (4.4 GB) |

Paired over the same images, v11 is ahead by 0.0099 PQ (95% interval 0.0070 to 0.0129, 4,000 bootstrap resamples, `results/report_figures/`). The gain is real and small. It costs eleven times the run time and three times the weights, and the competition lists computational efficiency among its criteria, so the faster configuration was submitted. The v11 code is in this repository (`FilamentPipeline(weights, version="v11")`, `src/solar_filament/sam_refiner.py`, `scripts/sam2_refiner.py`). Its 22 weight files are not stored here; their names, sizes and SHA-256 hashes are in `v11/MANIFEST.json`.

## Experiments that were rejected

Each was measured on the same 707 out-of-fold images against the version current at the time. Details and numbers are in section 6 of `REPORT.md`; the dated entries are in `CHANGELOG.md`.

| Idea | Outcome |
|---|---|
| Soft annotator-consensus targets for the stage-1 crop refiner | no gain (0.4460 against 0.4474) |
| CNN scorer on candidate crops as a selector feature | no gain |
| Overlapping instances with deduplication | worse (0.4446 to 0.4482 against 0.4583) |
| Third detector (YOLO11m at 1280 px), candidates from dense maps | no gain or worse |
| 16 selector variants (context features, larger model, two-part scorer) | within 0.0002 of the reference |
| Labels of temporally neighbouring training images moved by solar rotation, as selector features | no gain (0.4598 against 0.4596) |
| Second selection pass with stage-2 confidence features | no gain |
| ConvNeXt, ResNeXt and DINOv3-pretrained encoders in stage 2 | at most +0.001 |
| Soft-clDice loss term in SAM 2.1 training | worse alone, no gain in the average |
| Learned choice between mask variants per instance; classical grow, trim, dilate, erode | no gain; morphology is much worse |

## Diagnostics worth knowing

| Question | Answer | Log |
|---|---|---|
| How well do annotators agree? | One annotator scored against another reaches PQ 0.340 | `scripts/inter_annotator_ceiling.py` |
| What are the errors? | 18% of false positives match another annotator's label; for 25% of false negatives a good candidate existed and was dropped | `results/error_lab/` |
| What would a perfect keep rule on our candidates reach? | about 0.53 | `REPORT.md` section 5 |
| Is the gap between out-of-fold PQ and the public board sampling noise? | No. Random 90-image subsets have a standard deviation of 0.011 and never fall below 0.415 | `results/board_noise/` |
| Does the polygon rasteriser matter? | Scoring against labels drawn by another routine moves PQ by up to 0.04; retraining recovers at most 0.015 | `CHANGELOG.md`, 2026-11-13 |
| Do splitting and merging matter? | Rarely: 1.4% of labels hold several predictions, 2.5% of predictions hold several labels | `results/rubric_metrics/` |

## Scripts and logs

Every script in `scripts/` writes a summary and per-row files to `results/<script name>/`. Training and the heavier labs ran as GPU jobs defined in `modal_app.py`.

| Step | Script | Log |
|---|---|---|
| Five-fold detectors and crop refiners | `modal_app.py` (`kfold_all_job`, `full_member_job`) | `results/kf5/` |
| Paste rules and selector training | `scripts/postproc_lab.py`, `scripts/selector_lab.py` | `results/postproc_lab/`, `results/selector_lab/` |
| Stage 2 (ResNet34 U-Net and encoder variants) | `scripts/stage2_refiner.py`, `scripts/stage2_apply.py` | `results/stage2_refiner/`, `results/stage2_apply/` |
| Stage 2 with SAM 2.1, mask-quality score | `scripts/sam2_refiner.py`, `scripts/sam_score_lab.py` | `results/sam2_refiner/`, `results/sam_score_lab/` |
| Rubric metrics (Dice, IoU, relations) | `scripts/rubric_metrics.py` | `results/rubric_metrics/` |
| Error analysis | `scripts/error_lab.py` | `results/error_lab/` |
| Temporal neighbours | `scripts/temporal_lab.py`, `scripts/temporal_check.py` | `results/temporal_lab/` |
| Mask variants, overlaps, merging | `scripts/variant_lab.py`, `scripts/overlap_lab.py`, `scripts/merge_lab.py` | `results/variant_lab/`, `results/overlap_lab/` |
| Final weights | `scripts/collect_weights.py` | `results/collect_weights/`, `weights/MANIFEST.json` |
| Full prediction and comparison with the submitted file | `scripts/predict.py` | `results/predict/repro/` |
| Clean-environment check of this repository | `scripts/verify_release.py` | `results/verify_release/` |
| Report figures and bootstrap intervals | `scripts/report_figures.py` | `results/report_figures/` |

## Leaderboard context

The public board has a block of about 110 teams at exactly 0.55. It traces to a public notebook that writes a hard-coded prediction file. The public MAGFiLO test labels, which cover most test images, are prohibited by the organisers. Neither was used here. The best public solutions trained on the training split only score 0.38 to 0.42.
