# Solar filament instance segmentation: method report

Solar Filament Segmentation Challenge 2026 (MAGFiLO 1.0, GONG H-alpha full-disk images, 2048 x 2048 px).

Two versions are packaged for hand-in, each as a self-contained repository built by `scripts/make_release.py` (weights, notebook, `requirements.txt`, the four-page report in `report/`).

| Version | Stage 2 | Out-of-fold PQ (95% interval) | Mean Dice per view | Public board | GPU time per image |
|---|---|---|---|---|---|
| v7 | ResNet34 U-Net, five folds | 0.4626 (0.454 to 0.471) | 0.655 | 0.41 | 1.2 s |
| v11 | SAM 2.1 base-plus and large, five folds each, plus SAM quality score in the keep rule | 0.4725 (0.464 to 0.481) | 0.667 | 0.41 | 11 s |

Paired over the same 707 images (bootstrap over images, 4,000 resamples, `results/report_figures/`), v11 exceeds v7 by 0.0099 PQ (95% interval 0.0070 to 0.0129) and by 0.012 mean Dice (0.008 to 0.017). The rubric script gives 0.4725 for v11 and the lab script that produced the table in section 4.1 gives 0.4724; they differ by one matched pair.

## 1. Task and metric

Each image holds dark, thin filaments on the solar disk. The task is one mask per filament. An image was labelled by one to three annotators, and the labels of every annotator ("view") are scored separately and pooled:

PQ = (sum of IoU over matched pairs) / (TP + 0.5 FP + 0.5 FN), where a prediction and a label match when their IoU exceeds 0.5.

Two properties of the data shape every decision below.

| Fact | Value | Source |
|---|---|---|
| Training images, annotator views, labelled filaments | 707, 1,154, 8,199 | competition annotations |
| PQ of one annotator scored against another | 0.340 | `scripts/inter_annotator_ceiling.py` |
| Share of views that belong to images with several annotators | 64% | competition annotations |

Annotators disagree mostly about which dark structures count as a filament and where one filament ends and the next begins, less about the outline of a filament they both drew. A model that copies one annotator therefore cannot score well. The pipeline instead proposes many candidates and then decides, per candidate, whether the average annotator would have drawn it.

## 2. Pipeline

`src/solar_filament/pipeline.py` (`FilamentPipeline.predict`) is the whole inference path for one image.

| Stage | Model | Detail |
|---|---|---|
| 1. Proposals | Two YOLO11s-seg detectors, 1024 px and 1536 px input | Boxes with confidence 0.05 or more. A low threshold keeps recall high. The decision is taken later |
| 2. Crop refinement | Box-seeded U-Nets (`src/solar_filament/crop_refiner.py`) | Input: gray crop, CLAHE crop, box mask. ResNet18 at 256 px and ResNet34 at 384 px averaged for the 1024 px detector, ResNet18 at 256 px for the 1536 px detector. Four flips |
| 3. Paste | `paste_masks` in `src/solar_filament/postproc.py` | Candidates are pasted in detector-score order at probability 0.65. A pixel belongs to the first candidate that claims it. Instances under 96 px are dropped. Masks never overlap |
| 4. Selection | LightGBM regressor (`rescored_labels`) | Predicts the expected matched IoU of an instance averaged over annotator views, from 17 features (detector score, refiner confidence, size, shape, contrast, agreement of detector mask and refiner mask). Instances at or below 0.24 are dropped |
| 5. Stage 2 | ResNet34 U-Net, five fold models (`src/solar_filament/stage2.py`) | Each kept instance is cut out along its long axis into a 384 x 128 crop with five channels: gray, CLAHE, dense U-Net probability map, its own mask, the other kept instances. The output thresholded at 0.40 replaces the mask |

Version v11 (`FilamentPipeline(weights, version="v11")`) changes stages 4 and 5. Instances down to a selector value of 0.12 go to stage 2. Stage 2 is SAM 2.1 (Ravi et al., ICLR 2025), fully fine-tuned on the same crops, folds and targets: the gray, CLAHE and dense-map channels stretched to 1024 x 1024 are the image and the bounding box of the stage-1 mask is the prompt (`src/solar_filament/sam_refiner.py`, training in `scripts/sam2_refiner.py`). Base-plus and large models, five folds each, are averaged over four flips and thresholded at 0.40. Five further base-plus models have their IoU prediction head trained on the matched IoU of every candidate, the idea of Mask Scoring R-CNN (Huang et al., CVPR 2019). An instance is kept when the mean of the selector value and this score exceeds 0.24. Sections 4.2, 4.3 and 5 below were computed for v7.

Training targets:

- Detectors are trained with every annotator view as a separate sample, so an image labelled three times contributes three label sets. Crop refiners are trained on detector boxes paired with the labelled filament they overlap.
- The selector is trained on out-of-fold candidates. Its target is the candidate's matched IoU (zero when unmatched) averaged over the views of its image, which is its expected contribution to the numerator of PQ.
- Stage 2 is trained on out-of-fold instances. Its target is the mean of the best-overlapping label of every view that has one with IoU above 0.2, which is the probability that an annotator who drew this filament marked each pixel (4,678 crops with a target out of 5,753).

Only the competition training split was used. No external labels and no test labels.

## 3. Validation protocol

All 707 training images are split into five grouped folds. For every fold a detector, its refiners and the stage-2 model are trained on the other four. Every number in this report is therefore out of fold and is computed with a scorer that mirrors the competition scorer (pooled over views, IoU above 0.5).

The selector is trained with a second grouped five-fold split over the out-of-fold candidates.

The models that produce the test predictions are the same recipes trained on all 707 images (detectors, refiners, dense U-Net) or the five fold models averaged (stage 2). The selector is trained on all out-of-fold candidates.

## 4. Results

### 4.1 Panoptic Quality

| Version | Change | Out-of-fold PQ | TP | FP | FN | Public board |
|---|---|---|---|---|---|---|
| Fixed rule | detector confidence 0.35, no selector | 0.4290 | | | | 0.37 |
| v3 | learned selector, one refiner | 0.4547 | 5,189 | 2,230 | 3,010 | 0.40 |
| v5 | 1024 px detector, two averaged refiners, learned selector | 0.4583 | 5,219 | 2,213 | 2,979 | 0.40 |
| v6 | plus candidates of the 1536 px detector | 0.4596 | 5,254 | 2,251 | 2,945 | 0.41 |
| v7 | plus stage 2 | 0.4626 | 5,295 | 2,210 | 2,904 | 0.41 |
| v9 | stage 2 replaced by fine-tuned SAM 2.1 (small and base-plus averaged) | 0.4695 | 5,331 | 2,174 | 2,868 | 0.41 |
| v10 | SAM 2.1 base-plus and large averaged | 0.4704 | 5,341 | 2,164 | 2,858 | 0.41 |
| v11 | v10 masks, kept when the mean of selector value and SAM quality score exceeds 0.24 | 0.4724 | 5,451 | 2,285 | 2,748 | 0.41 |

Random subsets of 90 of the 707 out-of-fold images (the approximate size of the public half of the test set) give a PQ standard deviation of 0.011 for the v11 configuration, with no subset of 20,000 below 0.415 (`results/board_noise/summary.json`). Averaging per view or per image instead of pooling gives 0.463 and 0.462. Sampling noise and the averaging convention therefore do not explain the board value.

For v7, SQ is 0.686 and RQ is 0.674 (`results/rubric_metrics/dump_kf5b_fold0_x5_avg384_pool1536_v7/summary.json`).

The gap between out-of-fold PQ (0.46) and the public board (0.41) is present for every version and for every public solution that reports both numbers. Site, year and annotator-count distributions of train and test match, so we attribute it to annotators in the test set that are absent from training.

### 4.2 Dice and IoU (v7, 1,154 annotator views)

For v11 the same script gives (`results/rubric_metrics/dump_kf5b_fold0_x5_avg384_pool1536_v11/summary.json`): mean Dice per view 0.667, median 0.701, pooled Dice 0.699, mean IoU per view 0.519, pixel precision 0.736, pixel recall 0.666, matched pairs mean IoU 0.690 and mean Dice 0.814, labels holding two or more predictions 1.6%, predictions holding two or more labels 2.5%. The tables below are for v7.

Semantic scores compare the union of predicted instances with the union of one annotator's labels.

| Quantity | Value |
|---|---|
| Mean Dice per view | 0.655 |
| Median Dice per view | 0.688 |
| Dice, 5% / 25% / 75% / 95% quantiles | 0.346 / 0.596 / 0.761 / 0.833 |
| Dice pooled over all pixels | 0.686 |
| Mean IoU per view | 0.505 |
| Pixel precision, pixel recall | 0.728, 0.648 |
| Matched pairs: mean IoU, mean Dice | 0.686, 0.811 |

The Dice per view equals `torchmetrics.segmentation.DiceScore` (foreground class, micro average, one sample per view). `tests/test_rubric_dice.py` checks this.

Distribution of Dice per view:

| Dice bin | 0.0 to 0.1 | 0.1 to 0.2 | 0.2 to 0.3 | 0.3 to 0.4 | 0.4 to 0.5 | 0.5 to 0.6 | 0.6 to 0.7 | 0.7 to 0.8 | 0.8 to 0.9 | 0.9 to 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Views | 28 | 3 | 9 | 38 | 60 | 161 | 314 | 404 | 135 | 2 |

Distribution of IoU per view:

| IoU bin | 0.0 to 0.1 | 0.1 to 0.2 | 0.2 to 0.3 | 0.3 to 0.4 | 0.4 to 0.5 | 0.5 to 0.6 | 0.6 to 0.7 | 0.7 to 0.8 | 0.8 to 0.9 | 0.9 to 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Views | 29 | 24 | 49 | 135 | 259 | 320 | 267 | 68 | 3 | 0 |

Distribution of IoU over the 5,295 matched pairs:

| IoU bin | 0.50 to 0.55 | 0.55 to 0.60 | 0.60 to 0.65 | 0.65 to 0.70 | 0.70 to 0.75 | 0.75 to 0.80 | 0.80 to 0.85 | 0.85 to 0.90 | 0.90 to 0.95 |
|---|---|---|---|---|---|---|---|---|---|
| Pairs | 389 | 581 | 783 | 1,082 | 1,110 | 913 | 380 | 56 | 1 |

The 28 views with Dice below 0.1 are views with few or no labels in which the model predicted filaments, or the reverse.

The operating point is chosen for PQ. Keeping more instances raises Dice and lowers PQ: for v5, a selector threshold of 0.14 gives mean Dice 0.669 at PQ 0.447, against 0.657 at PQ 0.458 for the threshold used.

### 4.3 One-to-many and many-to-one relations (v7)

A prediction lies "inside" a label when more than half of its pixels do, and the reverse.

| Relation | Count | Share |
|---|---|---|
| Labels holding two or more predictions (fragmentation) | 115 of 8,199 | 1.4% of labels |
| Predictions holding two or more labels (merging) | 191 of 7,505 prediction and view pairs | 2.5% |
| Labels holding exactly one prediction | 5,846 | 71.3% |
| Labels holding no prediction | 2,238 | 27.3% |

Splitting and merging are rare. The dominant errors are filaments that are missed and predictions that no annotator of that view drew.

## 5. Error analysis

`scripts/error_lab.py` classifies every false positive and false negative of v5 (`results/error_lab/dump_kf5b_fold0_x5_avg384/summary.json`). The picture is the same for v6 and v7.

| False positives (2,213) | Count |
|---|---|
| Fragment of a larger labelled filament | 706 |
| No label nearby in this view | 686 |
| Matches another annotator's label of the same image | 391 |
| Covers the label but is much larger | 315 |
| IoU just under 0.5 | 115 |

| False negatives (2,979) | Count |
|---|---|
| A good candidate existed and the selector dropped it | 739 |
| Only a poor candidate existed | 630 |
| Covered in part by a kept prediction | 597 |
| Lies inside a larger kept prediction | 474 |
| No candidate | 387 |
| IoU just under 0.5 | 132 |
| Split over several predictions | 20 |

Two readings follow. First, 391 false positives are correct for another annotator of the same image, and 293 of the 706 fragments also match another annotator: a substantial part of the error is annotator disagreement and cannot be removed by a better model. Second, a perfect selector on the existing candidates would reach about 0.53, but the information that separates a filament one annotator draws from one they skip is largely not in the image. An independent team with a different architecture reports the same ceiling.

## 6. Other experiments (the SAM 2.1 rows are the ones adopted in v9 to v11)

Every row was measured on the same 707 out-of-fold images. The reference is the version that was current at the time.

| Idea | Result | Reference |
|---|---|---|
| Soft annotator-consensus targets for the crop refiner | 0.4460 | 0.4474 |
| CNN scorer on candidate crops as selector feature | 0.4480 | 0.4476 |
| Dense U-Net probability map as selector features (v4) | 0.4555 | 0.4547 |
| Allowing overlapping instances, deduplicated | 0.4446 to 0.4482 | 0.4583 |
| 1536 px detector instead of 1024 px | 0.4553 | 0.4547 |
| Third detector (YOLO11m at 1280 px) pooled | 0.4586 | 0.4605 |
| Candidates from dense probability maps pooled | 0.4443 to 0.4600 | 0.4605 |
| 16 selector variants (context features, second-detector agreement, larger model, two-part scorer) | 0.4581 to 0.4598 | 0.4596 |
| Labels of training images taken within four days, moved by solar rotation, as selector features (`scripts/temporal_lab.py`) | 0.4598 | 0.4596 |
| Second selection after stage 2 with stage-2 confidence features (`scripts/rescore2_lab.py`) | 0.4629 | 0.4625 |
| Stage 2 with a ConvNeXt-tiny encoder, averaged with the ResNet34 stage 2 (`scripts/stage2_refiner.py --encoder --blend`) | 0.4634 | 0.4626 |
| Stage 2 with a ResNeXt50 encoder, averaged with the ResNet34 stage 2 | 0.4618 | 0.4626 |
| Stage 2 with a DINOv3-pretrained ConvNeXt-tiny encoder: alone, averaged with ResNet34, and three-model average | 0.4627, 0.4636, 0.4635 | 0.4626 |
| SAM 2.1 as the stage-2 refiner, box prompted and fully fine-tuned (`scripts/sam2_refiner.py`): small, base-plus, both averaged | 0.4678, 0.4692, 0.4695 | 0.4626 |
| SAM 2.1 large alone, and base-plus averaged with large | 0.4690, 0.4704 | 0.4626 |
| Keep/drop by the mean of the selector value and SAM's fine-tuned mask-quality score (`scripts/sam_score_lab.py`, measured on the ResNet34 stage-2 masks) | 0.4644 | 0.4629 |
| The same keep/drop rule on the SAM base-plus plus large masks (the v11 configuration) | 0.4724 | 0.4704 |
| Soft-clDice loss term (Shit et al., CVPR 2021, weight 0.5) in SAM base-plus training: alone, and averaged with base-plus and large | 0.4673, 0.4703 | 0.4692, 0.4704 |
| Learned choice between mask variants per instance (thresholds, morphology, merging, extension) | 0.4585 to 0.4589 | 0.4583 |
| Classical grow, trim, dilate or erode applied to all masks | 0.09 to 0.39 | 0.4583 |

The temporal experiment is informative beyond its score. Test images are interleaved in time with training images (median gap to the nearest training image 1.5 days) and the rotation mapping is accurate (moved labels cover 28% of the next image's labelled pixels, against 1.5% without rotation). Yet the moved labels cover our false positives almost as often as our true positives (mean coverage 0.27 against 0.35). The structures we predict and annotators skip are mostly structures that another annotator, on another day, did label.

An oracle that picks the best mask variant per instance reaches 0.51, so better masks exist among simple variants of ours. A model could not learn which variant to pick (correlation 0.29 with the true gain). Stage 2 is the one mask-level change that held up, at every selector threshold tried (+0.0025 to +0.0035).

Overlapping or duplicated masks raise the score of the public scorer because it counts every matched pair. We did not use this. Our submissions contain disjoint masks only.

## 7. Leaderboard context

The public board has a block of about 110 teams at exactly 0.55. It traces to a public notebook that writes a hard-coded prediction file and does not use its own model. The public test labels of MAGFiLO, which cover 139 of the 180 test images, are prohibited by the organisers. We used neither. The best public solutions that train on the training split only score 0.38 to 0.40.

## 8. Reproduction

| Step | Command or file |
|---|---|
| Environment | `uv venv && uv pip install -r requirements.txt` |
| Weights | twelve files listed in `weights/MANIFEST.json` (1.4 GB) with their SHA-256 hashes |
| Demo notebook | `notebooks/solar_filament_pipeline.ipynb`, built and executed by `scripts/build_notebook.py --execute` |
| Full test prediction | `python scripts/predict.py --images=<test_images> --weights=weights --out=submission.csv` |
| Training | Modal jobs in `modal_app.py`: `--task=kfold` (five folds), `--task=full_member` (full-data detector and refiner), `scripts/stage2_refiner.py` (stage 2), `scripts/postproc_lab.py` (selector) |
| v11 | `uv pip install -r requirements-v11.txt`, then `python scripts/predict.py --version=v11 --weights=<dir>`. 22 weight files (4.4 GB) listed in `results/collect_weights/summary_v11.json`. Checked on all 180 test images against the submitted file (`results/predict/v11_repro/summary.json`): 1,243 of 1,243 rows matched, mean IoU 0.998, 741 masks identical, four extra borderline instances |
| Metrics of this report | `scripts/rubric_metrics.py`, `scripts/error_lab.py` |

The single pipeline was run on all 180 test images on one GPU and compared with the submitted file (`results/predict/v7_repro/summary.json`): 1,214 rows against 1,214, every row matched, 1,190 masks identical, mean IoU of matched masks 0.9999, the same instance count in all 180 images, 1.2 s per image. The 24 masks that differ do so by a few pixels, because the submitted file was built in separate jobs on a different GPU type. On a CPU the outlines move by a pixel or two (best IoU per filament 0.95 to 0.99) and the instance counts are unchanged.

## 9. Limitations

- The selector threshold, the paste threshold and the stage-2 threshold were chosen on the same out-of-fold predictions that are reported. Each was picked from a handful of values, so the bias is small, but the reported PQ is slightly optimistic.
- Stage-2 models saw instances selected at one threshold (0.24). Changing the selector threshold without retraining them is untested beyond 0.20 to 0.28.
- The pipeline has twelve weight files and five stages. A single model would be simpler to maintain. We kept the stages because each one was measured to help and the selector needs candidates from a recall-oriented first stage.
- The drop from out-of-fold PQ to the public board is not explained by anything we could measure.
