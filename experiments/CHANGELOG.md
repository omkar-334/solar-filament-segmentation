# Changelog

## 2026-11-11

- Added `scripts/inter_annotator_ceiling.py`. Annotator vs annotator PQ on 296 multi-view images is 0.340 (SQ 0.634); the oracle best view per image reaches 0.367. Our held-out model (0.425) already exceeds human pairwise agreement.
- Added `dump_proposals_job` (Modal GPU): saves every detector proposal (conf >= 0.05) with its four-flip refiner probability crop, so post-processing is tuned on CPU containers.
- Added `src/solar_filament/postproc.py` and `scripts/postproc_lab.py` (run on Modal via `postproc_lab_job`). The lab reproduces the current rule exactly (0.42489 PQ) and attributes the 716 missed instances: 484 had a proposal with IoU > 0.5, 458 of them below detector confidence 0.35; only 59 had no overlapping proposal.
- Learned instance selector: LightGBM predicts each candidate's expected matched IoU across annotator views; an instance is kept when the prediction exceeds tau. Grouped 5-fold CV on the 141 held-out images: 0.4474 PQ (tau 0.28), +0.0225 over the current rule. Agreement features from YOLO11m and YOLO11x proposals added nothing (0.4461), so they are not used.
- Added full-data training (`full_data_pipeline_job`): YOLO11s and the crop refiner on all 707 images with the validated recipe.
- Added soft-consensus refiner targets (`soft_consensus=True`): each pixel's target is the fraction of annotators that drew the filament there.
- Added a 5-fold grouped OOF pipeline over all 707 images (`kfold_all_job`) to train the selector on 5x more data and to replace the single 141-image validation estimate.
- Added `rescored_submission_job` to build test submissions with the learned selector.
- Initialized git; layout now has `scripts/`, `tests/`, `results/`.
- `submission_rescored_v1.csv` (seed-42 YOLO11s + OOF refiner + learned selector, tau 0.28, 1,081 rows) scored **0.38 public LB**, up from 0.37.
- Soft-consensus refiner: 0.4460 PQ with the selector vs 0.4474 for the existing refiner; SQ unchanged at 0.686. Rejected.
- Re-pasting only kept instances (in predicted-value order) gives the same PQ as dropping them; the simpler drop rule is kept.
- Bug: `full_data` passed `patience` twice to YOLO; the first full-data detector never trained. Fixed and relaunched.
- Bug: 3 of 5 k-fold detectors (folds 0, 1, 2) diverged to NaN at epochs 23, 34 and 7 under AMP with AdamW `lr0=0.01` (Ultralytics' default when AdamW is forced). Added a NaN guard callback and an `lr0` option; folds 0 to 2 are being retrained with `lr0=0.002` (prefix `kf5b`). A backup full-data detector also uses `lr0=0.002`.
- Added `src/solar_filament/cnn_rescorer.py` and `cnn_rescorer_job`: a ResNet18 on 4-channel candidate crops (gray, CLAHE, refiner probability, claimed mask) predicting expected matched IoU, used as an extra selector feature.
- Folds 3 and 4 (282 images, 454 views, independent of the seed-42 split): fixed rule 0.4197 PQ, learned selector 0.4479 (+0.028).
- CNN candidate scorer (OOF correlation 0.69 to 0.74): selector with CNN feature 0.4480 vs 0.4476 without; CNN alone 0.4416. Rejected.
- Oracle selection ceiling on folds 3 and 4 (keep a candidate iff its true mean matched IoU > tau): 0.527 PQ. Part of that gap is annotator-specific and not learnable from the image, so selection is close to saturated.
- Added `insample_proposals_job` and `final_model_job`: the final full-data refiner uses the same recipe as the k-fold refiners so the selector trained on k-fold OOF data stays calibrated. `scripts/watch_and_launch.sh` launches it and the 707-image OOF lab when inputs are ready.
- Train vs test: site and year distributions match; PQ by annotator count on folds 3 and 4 is 0.464 (1 view), 0.438 (2), 0.435 (3). Neither explains the ~0.07 local to LB gap.
- The official scorer code could not be retrieved (Kaggle renders notebooks client-side; the organizers' repo returns 404). The July scorer re-instanced predictions by connected components.
- `scripts/metric_variants.py`: 10% of selected instances (200 of 1,992) have several components. If the scorer splits rows into connected components, PQ drops from 0.4476 to 0.4132; if it re-labels the union, to 0.4269. `scorer_robust` (largest component, 1 px gap between instances) scores 0.4455 under all three scorer variants.
- `submission_rescored_v2_robust.csv`: v1 plus `scorer_robust`, 1,081 rows. Diagnostic for the scorer hypothesis.
- All 8,199 training annotations are single polygons.
- Public solution review (GitHub repos, 2026-11-11). Official scorer mirror (China-gituser `code/pq_official.py`, reportedly mirroring the organiser notebook): no connected-component re-splitting, TP = any pair with IoU > 0.5, pooled over every annotator copy, no area filter. This matches our local metric, so `scorer_robust` is not expected to help.
- The 0.55 to 0.56 public cluster (38 teams) is attributed in shingo257's report to test leakage: 139 of 180 test images match the public MAGFiLO 1.0 annotations (Kaggle discussion 728062). Using those labels is prohibited. Honest public scores in documented repos are 0.37 to 0.40 for OOF PQ 0.44 to 0.45 (e.g. tdogukan: OOF 0.451, public 0.39; shingo257: OOF 0.4426, public 0.39; brosayaying: OOF 0.4386, public 0.38). Our OOF 0.448 and public 0.38 sit in that band.
- Every documented repo shows a similar OOF to public gap (public about 0.86 x OOF); no repo found a scorer cause.
- U-Net family: `unet_family_job` trains a ResNet34 U-Net++ per kf5 fold (held-out probability maps for all 707 images) plus a full-data model (test probability maps). `instance_features` gains 4 U-Net agreement features (mean probability in the mask, fraction above 0.5, IoU with touching U-Net components, ring mean). `train()` accepts `train_files_json`.
- `scripts/finish_v3_v4.sh` builds v3 (full-data detector and refiner, selector trained on 707-image OOF) and v4 (v3 plus U-Net features) automatically. `scorer_robust` is now off by default.
- 707-image grouped OOF (5 folds): fixed rule 0.4290 PQ, learned selector **0.4547** (tau 0.24, 5,189 TP / 2,230 FP / 3,010 FN). With U-Net agreement features: 0.4555 (+0.0008, within noise).
- `submission_rescored_v3.csv`: full-data YOLO11s (lr0 0.002) + full-data refiner (k-fold recipe) + 707-image OOF selector, tau 0.24, 1,217 rows, audit passed.
- `submission_rescored_v4_unet.csv`: v3 + U-Net features, tau 0.24, 1,213 rows, audit passed.
- v3 scored **0.40 public LB** (rank about 200; the leakage cluster sits above).
- Fragment merging after selection (`scripts/merge_lab.py`, 707 OOF images): every radius loses PQ (r=2: 0.4547, r=8: 0.4521, r=16: 0.4452). Rejected; annotators keep nearby filaments separate.
- Refiner encoder and crop size are configurable (`encoder`, `crop_size`); `refiner_variant_job` trains a ResNet34 refiner at 384 px crops on all five folds and on all data.
- Refiner variant on 707 OOF images: ResNet34 at 384 px crops 0.4574 (refiner thr 0.55); average of the ResNet18/256 and ResNet34/384 refiner probabilities **0.4583** (tau 0.24, +0.0036 over v3), SQ 0.686.
- `submission_rescored_v5_avg384.csv`: v3 detector + averaged refiners + selector trained on the averaged OOF, tau 0.24, 1,195 rows, audit passed. Public LB 0.40 (same as v3).

## 2026-11-11 (1536 px detector)

- Miss breakdown on the averaged OOF (8,199 pooled GT, 3,670 FN): 365 have no proposal, 902 only a poor one (IoU 0.1 to 0.5), 2,403 have a good proposal that the selector dropped (2,278 of them with detector score below 0.35).
- Launched the 5-fold run `kf5_1536`: YOLO11s-seg trained and run at 1536 px (batch 4), same refiner and dump recipe. The k-fold jobs now take `image_size`. Call `fc-01M75AHR6EBZGZ7WE7365THEE2`.

## 2026-11-11 (overlap lab, rejected)

- `scripts/overlap_lab.py`: candidates scored without pixel claiming (overlapping instances), averaged refiners, 707 OOF images, 14,869 candidates. The official scorer accepts overlapping rows and counts every IoU > 0.5 pair as a TP.
- Reference (pixel claiming plus selector): 0.4583.

| Variant | Official rule | Strict one-to-one |
|---|---|---|
| Overlap, no dedupe | 0.4763 | 0.3801 |
| Overlap, mask NMS 0.7 | 0.4482 | 0.4358 |
| Overlap, mask NMS 0.5 | 0.4446 | 0.4440 |
| Overlap, mask NMS 0.3 | 0.4477 | 0.4477 |

- The only gain (0.4763) is double counting of near-duplicate masks. Once duplicates are removed, overlap is worse than pixel claiming. Rejected, no submission built.

## 2026-11-11 (1536 px detector result, rejected)

- `kf5_1536` finished: 5 folds, 60 epochs, no divergence. Lab on the 707 OOF images, ResNet18/256 refiner.

| Detector | Fixed rule (conf 0.35, thr 0.65) | With selector | Misses with no proposal | Misses with poor proposal |
|---|---|---|---|---|
| YOLO11s 1024 px (v3 recipe) | 0.4290 | 0.4547 | 365 | 902 |
| YOLO11s 1536 px | 0.4323 | 0.4553 | 372 | 880 |

- The selector score is unchanged (+0.0006) and below v5 (0.4583, averaged refiners). Higher detector resolution does not recover the missed filaments. No full-data model or submission built.
- Public notebooks reviewed (phuongncn, robinamirbahar, hdjojo, huendao, prvsiyan): all 0.34 to 0.40 on the PQ board. The "LB 0.70" title predates the switch from Dice to PQ on 2026-08-07.

## 2026-11-11 (error analysis and mask shape experiments)

- `scripts/error_lab.py`: every FP and FN of the v5 OOF output (707 images, tau 0.24, PQ 0.4583, TP 5,219, FP 2,213, FN 2,979) put in one category, counted per annotator view. Outputs in `results/error_lab/dump_kf5b_fold0_x5_avg384/`.

| False positives | Count | Share |
|---|---|---|
| Fragment (mask lies inside a larger labelled filament) | 706 | 32% |
| False alarm (no annotator drew anything there) | 686 | 31% |
| Other annotator only (a match in another view) | 391 | 18% |
| Oversized (mask covers the label but is much larger) | 315 | 14% |
| Boundary or shift | 115 | 5% |

| False negatives | Count | Share |
|---|---|---|
| Dropped (good candidate existed, selector removed it) | 739 | 25% |
| Poor candidate only | 630 | 21% |
| Partial (our mask covers too little of it) | 597 | 20% |
| Merged by us (inside a much larger mask of ours) | 474 | 16% |
| No candidate | 387 | 13% |
| Boundary or shift | 132 | 4% |
| Split by us | 20 | 1% |

- Of the 706 fragment FPs, 293 are a correct match for another annotator of the same image. In the fragment cases the label is 2.3 times our mask (median), only 51% of it lies inside the detector box, and the refiner probability on the missing part is 0.08.
- What-if ceilings: remove every false alarm 0.479, fix every near miss (IoU 0.3 or more) 0.521, fix every localized error 0.546.
- `scripts/variant_lab.py`: alternative masks per selected instance. Oracle picks the best variant with the labels (upper bound), the learned chooser is a grouped 5-fold LightGBM.

| Family | Oracle PQ | Learned PQ |
|---|---|---|
| Base (v5) | 0.4583 | 0.4583 |
| Per-instance refiner threshold (0.30 to 0.80) | 0.4813 | 0.4585 |
| Dilate or erode | 0.4758 | (in the same chooser) |
| Largest component | 0.4623 | (in the same chooser) |
| Merge adjacent selected instances | 0.4586 | not trained |
| Extend with a dropped adjacent candidate | 0.4767 | 0.4589 |
| All families | 0.5025 | not trained |

- The right shape exists among the variants, but which one is right cannot be predicted from the image: 1,166 extensions help and 2,728 hurt, chooser correlation 0.29. Both learned choosers rejected. No submission built.

## 2026-11-11 (classical grow and trim, selector variants, rejected)

- `scripts/variant_lab.py` gained brightness-based variants: grow the mask into connected dark pixels (three settings), trim it to the dark body, grow on a black-hat response. Applied to every instance (no choice needed), base 0.4583:

| Variant applied to all instances | PQ | Median area change |
|---|---|---|
| Refiner threshold 0.55 | 0.4587 | x1.04 |
| Refiner threshold 0.45 | 0.4563 | x1.09 |
| Refiner threshold 0.30 | 0.4501 | x1.15 |
| Refiner threshold 0.80 | 0.4477 | x0.91 |
| Trim to dark body | 0.3882 | x0.91 |
| Grow into dark pixels (0.5, 16 px) | 0.3813 | x1.18 |
| Erode 1 px | 0.3712 | x0.74 |
| Dilate 2 px | 0.3265 | x1.54 |
| Black-hat grow | 0.0949 | x2.99 |

- Oracle for the classical family 0.4816, all families 0.5108. Learned chooser including the classical variants: 0.4584 at best.
- `scripts/selector_lab.py`: 16 selector variants (4 feature sets x 4 models: image context, agreement with the 1536 px detector, view weighting, larger model, hit probability times IoU). Range 0.4561 to 0.4591 against 0.4583. All within noise. Rejected.

## 2026-11-11 (rubric metrics, pooled detectors)

- `scripts/rubric_metrics.py`: the organisers' quantitative rubric on the 707 OOF images (1,154 annotator views), v5 recipe. Outputs in `results/rubric_metrics/`.

| Metric (v5, refiner threshold 0.65, tau 0.24) | Value |
|---|---|
| PQ | 0.4584 |
| Mean Dice per view (union of instances against union of labels) | 0.657 |
| Global pixel Dice | 0.686 |
| Median Dice per view, 5% to 95% range | 0.692, 0.356 to 0.832 |
| Pixel precision, pixel recall | 0.749, 0.633 |
| Mean IoU, mean Dice of matched pairs | 0.686, 0.811 |
| One-to-many (labels holding 2 or more predictions) | 1.7% of labels |
| Many-to-one (predictions holding 2 or more labels) | 2.6% of predictions |

- PQ against Dice: mean Dice peaks at tau 0.14 (0.669, PQ 0.447). Refiner threshold 0.55 at tau 0.24 gives Dice 0.662 at PQ 0.4582. The trade is small in both directions, so v5 stays as is.
- Pooled candidates of the 1024 px and 1536 px detectors (`postproc_lab.py --pool`): misses with no candidate fall from 365 to 261 and poor candidates from 902 to 720, but the selector result is 0.4605 at best against 0.4587. Within noise. Not built into a submission.

## 2026-11-11 (ensemble of detectors, in progress)

- `modal_app.py`: `full_member_job` (detector, refiner and test dump for one ensemble member on all 707 images), `detector_model` option for the k-fold jobs, `pool_dump_dirs` for `rescored_submission_job`, `--extra` JSON keyword arguments on the entrypoint. `postproc_lab.py --pool` accepts several detectors separated by `+`.
- Launched `final707_1536` (YOLO11s-seg 1536 px on all data, call `fc-01M75JNKF44X5KWYH4HFZDN7D6`) for a pooled submission, and the k-fold run `kf5_m1280` (YOLO11m-seg 1280 px, call `fc-01M75JP3YXEY3CT83ZDB3MYHTH`) as a third member.

## 2026-11-12

- Detector ensemble, third member (YOLO11m-seg at 1280 px, grouped 5-fold OOF, `kf5_m1280`). Pooled with the 1024 px base candidates and the averaged refiners, best selector PQ on 707 images: base only 0.4583, base + 1536 px 0.4605, base + YOLO11m 0.4577, all three 0.4586. A third detector removes the small gain of the second. Detector pooling is rejected as a source of further improvement; no full-data YOLO11m member is trained.
- Bug: the fold 2 dump of `kf5_m1280` had one image with 37 proposal rows and 36 refiner maps (the fold was restarted and its dump ran twice), which crashed both pooled labs. Added `scripts/check_dump.py` (row count against probability map count per image), re-dumped fold 2, and relaunched with `scripts/relaunch_labs.sh`.
- The full-data 1536 px member stalled for 40 minutes waiting for an A10 or L4. `insample_proposals_job`, `crop_refiner_train_job` and `dump_proposals_job` now also accept L40S, A100 and T4. The `dump` and `final_model` CLI tasks take `--extra` keyword arguments.
- Added `scripts/watch_ensemble.sh` (builds v6 when the 1536 px test dump exists).
- `submission_rescored_v6_pool1536.csv`: v5 candidates pooled with the full-data 1536 px member (`final707_1536`), selector trained on the pooled OOF candidates, refiner threshold 0.65, tau 0.24. Audit passed, 1,214 rows over 180 images (v5 has 1,195). OOF PQ at these settings 0.4596 against 0.4583 for v5. The watcher did not fire while the laptop slept, so the build was launched by hand.
- `submission_rescored_v6_pool1536.csv` scored **0.41 public LB** (v5 0.40), rank 152.
- Public notebook review, second pass. The block of about 110 teams at exactly 0.55 traces to `lamhuy8904/solar-filament-unet-segmentation-0-55`, which writes a hard-coded compressed prediction file as its submission and does not use its trained model. Not used. `tushhaaarrrr/solar-filament-segmentation-2026-0-40` (dense ensemble, component tree, hurdle scorer, exact DP, rotation-normalised stage-2 refiner) reports OOF PQ 0.4597 and 0.40 on the board, the same level as our pipeline from a different design.
- Dense-map candidates (`scripts/dense_proposals.py`, `dense_pool_job`): boxes from the OOF U-Net probability maps at five thresholds (10,199 proposals, 14.4 per image), refined by the fold refiners and pooled. Best selector PQ: base + dense 0.4443, v6 pool + dense 0.4503, v6 pool + dense pasted last (`--pool_scale=1,0.1`) 0.4600, against 0.4605 for the v6 pool. Rejected.
- Selector variants on the v6 pool (`scripts/selector_lab.py --pool`): 16 combinations of feature set and model span 0.4581 to 0.4598; the two-part scorer (match chance times IoU given a match) reaches 0.4598 against 0.4596 for the current selector. Rejected.
- Stage-2 refiner (`scripts/stage2_refiner.py`): every pasted candidate is re-segmented in a 384 x 128 crop aligned to its long axis, with gray, CLAHE, dense U-Net map, its own mask and the other selected instances as channels; ResNet34 U-Net, 14 epochs, same grouped 5 folds. OOF PQ with stage-2 masks at threshold 0.40 against v6 masks: 0.4613 against 0.4578 (tau 0.20), 0.4625 against 0.4596 (tau 0.24), 0.4599 against 0.4574 (tau 0.28). True positives at tau 0.24 rise from 5,254 to 5,294. The first printout of this script counted unselected candidates as false positives; the false positive count is fixed in the script and the figures here use the corrected count (`results/stage2_refiner/v6/sweep.csv` still holds the uncorrected `fp` and `pq` columns).
- `submission_rescored_v7_stage2.csv` (`scripts/stage2_apply.py`): v6 rows re-segmented by the mean of the five stage-2 fold models, threshold 0.40. Audit passed, 1,214 rows, no overlapping pixels, no mask under 96 px, mean IoU with the v6 masks 0.928, median area ratio 1.05.
- Added `lab_script_gpu_job` (`--task=script_gpu`); `dump_proposals_job` accepts external proposals; `rescored_submission_job` writes the selector value of every row (`*_pred.json`).
- `submission_rescored_v7_stage2.csv` scored **0.41 public LB**, the same two-decimal value as v6.
- Deliverables. `src/solar_filament/pipeline.py` (`FilamentPipeline`) is the single inference path of v7; `src/solar_filament/stage2.py` holds the stage-2 code shared by training, the test-time script and the pipeline. `scripts/predict.py` writes a submission from a weights folder; `scripts/collect_weights.py` gathers the twelve weight files (1.4 GB) with SHA-256 hashes into `weights/` (`weights/MANIFEST.json`).
- Reproduction check (`results/predict/v7_repro/summary.json`): the single pipeline on one GPU against the submitted v7 file gives 1,214 rows against 1,214, all matched, 1,190 identical masks, mean IoU of matched masks 0.9999, 1.24 s per image. On a CPU the best IoU per filament is 0.95 to 0.99 with the same instance counts.
- Rubric metrics for v6 and v7 (`scripts/rubric_metrics.py --pool --stage2`). v7: PQ 0.4626, SQ 0.686, RQ 0.674, mean Dice per view 0.655, median 0.688, pooled Dice 0.686, mean IoU per view 0.505, labels holding several predictions 1.4%, predictions holding several labels 2.5%. v6: PQ 0.4597, mean Dice 0.654. This independent recomputation confirms the stage-2 gain.
- `tests/test_rubric_dice.py`: the per-view Dice equals `torchmetrics.segmentation.DiceScore` (foreground, micro average).
- `notebooks/solar_filament_pipeline.ipynb` (built and executed by `scripts/build_notebook.py --execute`) and `REPORT.md`. `README.md` now points to them. `requirements.txt` gains `lightgbm` and `matplotlib`.
- Final sprint, two new approaches, both rejected. (1) Temporal neighbours (`src/solar_filament/temporal.py`, `scripts/temporal_check.py`, `scripts/temporal_lab.py`): test images are interleaved in time with training images (median gap to the nearest training image 1.53 days, 1.52 for train to train). Labels of other training images within four days are moved by solar differential rotation; the mapping is verified on 60 pairs (coverage 0.283 forward, 0.015 static, 0.012 backward). As nine selector features on the v6 pool: 0.4598 against 0.4596. Mean coverage of matched candidates 0.35, of unmatched 0.27, so the false positives are mostly structures that annotators of neighbouring days did label. (2) Second selection after stage 2 (`scripts/rescore2_lab.py`) with stage-2 confidence features: 0.4629 against 0.4625 when selecting by the stage-1 value. Both logged in `REPORT.md` section 6.
- Final sprint, second round (2026-11-12). (1) Stronger stage-2 encoders: `scripts/stage2_refiner.py` gained `--encoder` (any segmentation_models_pytorch encoder, via `STAGE2_ENCODER` in `src/solar_filament/stage2.py`) and `--blend` (average with another run). Averaged with the ResNet34 stage 2 at tau 0.24: ConvNeXt-tiny 0.4634, ResNeXt50 0.4618, reference 0.4626. The difference is 8 matches out of 8,199 labels, so it is rejected as noise. (2) The image id prefix is a six-digit code (team, batch, annotator); the annotator part differs between views of one image, mean filaments per image range from 6.3 to 7.7 by annotator, and no part of the code follows the observation date, so annotator style cannot be predicted for a test image. (3) Literature search (multi-annotator instance segmentation, MAGFiLO follow-ups, competition write-ups) found no method that addresses annotator disagreement under a matching metric. Added `scripts/modal_ca.py`: runs the Modal CLI with the CA bundle from `SSL_CERT_FILE`, needed behind a TLS-inspecting gateway.
- Final sprint, third round (2026-11-12). (1) DINOv3-pretrained ConvNeXt-tiny as stage-2 encoder (`--encoder=tu-convnext_tiny.dinov3_lvd1689m`): 0.4627 alone, 0.4636 averaged with the ResNet34 stage 2, 0.4635 for the three-model average with the ImageNet ConvNeXt (`--eval_only`, new). Averaging two stage-2 architectures gives about +0.001 and does not stack, so v7 is unchanged. `scripts/stage2_apply.py` accepts `<encoder>@<checkpoint>` should a blend ever be submitted. (2) PQ by number of annotator views (`scripts/rescore2_lab.py` now writes `view_strata.json`): 0.4729 on 411 one-view images (best tau 0.20), 0.4700 on 145 two-view images (tau 0.26), 0.4539 on 151 three-view images (tau 0.30). Even an all three-view test set would not explain the 0.41 on the public board. Year, site and month distributions of train and test match. (3) Decision-theory check of the keep rule: every kept instance adds 0.5 to the PQ denominator whether or not it matches, so the optimal rule keeps an instance when its expected matched IoU exceeds PQ/2 = 0.23. The tuned tau of 0.24 is already at that optimum, so only better prediction of the expected IoU could help.
- Final sprint, fourth round (2026-11-13). SAM 2.1 replaces the CNN stage-2 refiner (`scripts/sam2_refiner.py`, Modal task `script_sam`): the first three crop channels stretched to 1024 x 1024 are the image, the box of the stage-1 mask is the prompt, all weights are fine-tuned for 8 epochs on the same crops, targets and folds as before (crops exported by `stage2_refiner.py --save_crops`). Out-of-fold PQ at tau 0.24: small 0.4678, base-plus 0.4692, both averaged 0.4695 (threshold 0.50, rule "all"), against 0.4626 for the ResNet34 stage 2. Crop soft Dice 0.74 against 0.726. Averaging SAM with the CNN refiners is worse than SAM alone (0.4675, 0.4663). My earlier argument that a foundation-model refiner could not help was wrong. Built `submission_rescored_v9_sam2.csv` (`stage2_apply.py` with `sam2@<checkpoint>` entries, ten fold models): audit passed, 1,214 rows, mean IoU with the v6 masks 0.895. Also built `submission_rescored_v8_blend.csv` (ResNet34 plus DINOv3 ConvNeXt stage 2, 0.4636). The pipeline class, notebook and weights still describe v7 and need updating if v9 becomes the final selection.
- SAM 2.1 large as stage-2 refiner: 0.4690 alone (threshold 0.40), 0.4704 averaged with base-plus, so the gain saturates with model size. `submission_rescored_v9_sam2.csv` scored 0.41 on the public board, the same displayed value as v6 and v7. Built `submission_rescored_v10_sam2.csv` (base-plus plus large, ten fold models, threshold 0.40): audit passed, 1,214 rows, mean IoU with the v6 masks 0.879.
- SAM as a candidate scorer (`sam2_refiner.py --score`, `scripts/sam_score_lab.py`): the mask-quality head of SAM 2.1 base-plus is trained on the matched IoU of every candidate (mean over views, unmatched = 0) together with the mask loss. Out of fold, selecting by the mean of the stage-1 selector value and this score gives 0.4644 at tau 0.24 against 0.4629 for the selector value alone (52 more matches, 34 more false positives); the score alone gives 0.4620 and adding it to the LightGBM selector gives 0.4623 to 0.4630. `submission_rescored_v10_sam2.csv` scored 0.41 on the public board. Built `submission_rescored_v11_sam2.csv`: stage-1 candidates down to selector value 0.12 (1,489), v10 masks, kept when the mean of selector value and SAM score exceeds 0.24 (`stage2_apply.py --score_models --select`). Audit passed, 1,243 rows; 69 candidates below 0.24 were added and 40 above it dropped. The combination of SAM masks and SAM selection was not measured out of fold together.
- Measured the v11 configuration out of fold (`rescore2_lab.py --stage2=<SAM base-plus plus large maps> --t=0.4`, then `sam_score_lab.py --tag=bplus_sc_on_sam_masks`): 0.4724 at tau 0.24 when keeping by the mean of selector value and SAM score, against 0.4704 for the selector value alone on the same masks. The mask gain and the selection gain add. Sources of the methods adopted in this sprint: SAM 2 (Ravi et al., ICLR 2025; the 2.1 checkpoints are its updated release) as box-prompted refiner, its IoU prediction head as learned mask-quality score in the spirit of Mask Scoring R-CNN (Huang et al., CVPR 2019), and DINOv3 (Meta, 2025 preprint) as encoder initialisation. Note: `results/rescore2_lab/dump_kf5b_fold0_x5_avg384_pool1536/` on the volume now holds the SAM-mask run, the ResNet34 run of the same lab is the one committed earlier.
- `submission_rescored_v11_sam2.csv` scored 0.41 on the public board, like v6, v7, v9 and v10. The out-of-fold gain from v6 to v11 (0.4596 to 0.4724) is not visible at two decimals. v11 is now reproducible from one class: `FilamentPipeline(weights, version="v11")` (`src/solar_filament/pipeline.py`, SAM code in `src/solar_filament/sam_refiner.py`), `scripts/predict.py --version=v11`, weights collected by `scripts/collect_weights.py --version=v11` into `/data/weights_v11` (22 files, 4.4 GB, manifest in `results/collect_weights/summary_v11.json`), extra dependency in `requirements-v11.txt`. Check on a Modal GPU against the submitted file (`results/predict/v11_repro/summary.json`): 1,243 of 1,243 reference filaments matched, mean IoU 0.998, 741 identical, 1,247 rows (four extra borderline instances), 11 s per image. The notebook and the local `weights/` directory still hold v7. Added `scripts/probe_masks.py` and two diagnostic files, `submission_probe_v11_grow1.csv` and `submission_probe_v11_shrink1.csv` (every mask grown or shrunk by one pixel, median area ratio 1.26 and 0.74), to test whether test labels are drawn thicker or thinner than training labels; the out-of-fold score exceeds the board by about 0.06 and train and test match in year, site and month.
- Board noise check (`results/board_noise/summary.json`): PQ of random 90-image subsets of the out-of-fold results has a standard deviation of 0.011 around 0.472, and none of 20,000 subsets falls below 0.415; per-view and per-image averaging give 0.463 and 0.462. The 0.06 gap to the public board is therefore systematic, not sampling noise or an averaging convention. `REPORT.md` now lists v9 to v11 in section 4.1, describes the v11 pipeline in section 2 and its reproduction in section 8.
- Label rasterisation check (2026-11-13). Our validation labels are rasterised with pycocotools. The same training polygons rasterised with OpenCV `fillPoly` or with PIL differ from that by IoU 0.88 to 0.92 and about 10% in area (OpenCV and PIL with outline larger, PIL fill smaller). `data.polygon_mask` reads `SOLAR_RASTER` (diagnostic only) and `rescore2_lab.py --raster` scores the out-of-fold SAM masks (base-plus plus large, tau 0.24) against each convention: pycocotools 0.4704, OpenCV 0.4754, PIL with outline 0.4775, PIL fill 0.4341. Against PIL-fill labels a mask threshold of 0.55 gives 0.4450 and 0.70 gives 0.4422. The differences are half-pixel boundary conventions, not a whole-pixel shift (shifting by one pixel never raises the agreement above 0.89). If the hidden test masks were produced with a thinner convention than pycocotools, that would account for about 0.04 of the 0.06 gap to the board; which routine the organisers used is not known. Building `submission_probe_v11_t055.csv` (v11 with mask threshold 0.55) as a probe of this.
- Follow-up to the rasterisation check: SAM 2.1 base-plus retrained on PIL-fill targets (`stage2_refiner.py --raster=pil --save_crops`, `sam2_refiner.py --tag=bplus_pil`) scores 0.4471 against PIL-fill labels (threshold 0.40), against 0.4324 for the pycocotools-trained base-plus at the same threshold and 0.4446 for it at threshold 0.60. Retraining therefore recovers 0.015 at most and only 0.003 more than raising the threshold, so even if the test masks use a thinner convention the reachable gain is about 0.01, not the 0.04 I first estimated. Built `submission_probe_v11_thinlabels.csv` (masks from the PIL-target models, v11 selection) and `submission_probe_v11_t055.csv` as probes.
- clDice (Shit et al., CVPR 2021), a centre-line loss for thin tubular structures, added to SAM base-plus training with weight 0.5 (`sam2_refiner.py --cldice`, soft skeleton by ten rounds of min and max pooling): 0.4673 alone at threshold 0.50 against 0.4692 without it, and 0.4703 averaged with base-plus and large against 0.4704 for those two alone. Rejected.
- Demo notebook is version aware (`scripts/build_notebook.py`): it runs v11 when the SAM weight files are in `weights/`, otherwise v7, and its results table and text now cover v9 to v11 and the v11 reproduction check. Rebuilt and executed locally with the v7 weights (13 cells, no errors, 127 s on CPU).

## 2026-11-15 (hand-in packages for v7 and v11)

- Requirements read from the competition page and the Google Form on 2026-11-15: a four-page report in the organisers' acmart template (PDF, 10 MB at most), a public git repository with `requirements.txt` (all packages with versions), a notebook showing the entire pipeline, and the model weights stored in the repository and loaded by the notebook. The form also asks for the exact file submitted to Kaggle. Deadline 2026-11-15 23:59 CDT.
- `scripts/make_release.py --version=v7|v11`: builds a self-contained package in `release/<version>/` (source, scripts, tests, pinned `requirements.txt`, weights checked against their hashes and tracked with Git LFS, `submission.csv`, three demo test images, the logs the notebook reads, README, report). `release/` and `dl/` are ignored by git.
- `scripts/verify_release.py` (Modal, `--task=script_sam`): clean-room check of a package. It builds a fresh virtual environment from the package's own `requirements.txt`, checks the weight hashes, runs the tests, predicts all 180 test images with `scripts/predict.py` and compares with `submission.csv`, then executes the notebook. Outputs in `results/verify_release/<version>/`.
- The first clean-room run failed on `import cv2` (`libGL.so.1` missing): ultralytics installs `opencv-python`, which shadowed `opencv-python-headless`. The packages now pin `opencv-python==4.12.0.88` and the README names the two system libraries a minimal Linux image needs.
- `FilamentPipeline(weights)` and `scripts/predict.py` now choose the version from the weight files present (`detect_version`), so each package needs no version flag.
- `scripts/build_notebook.py --version --root`: notebook for the package layout; runs from a fresh clone on `demo_images/` and checks that the weight files are not Git LFS pointers.
- `scripts/rubric_metrics.py --score --floor`: rubric metrics for the v11 selection rule. v11 out of fold: PQ 0.4725 (TP 5,452, FP 2,284, FN 2,747; the lab script gave 0.4724 with one match fewer), SQ 0.690, RQ 0.684, mean Dice per view 0.667 (v7 0.655), median 0.701, mean IoU per view 0.519, labels holding several predictions 1.6%, predictions holding several labels 2.5%.
- `scripts/report_figures.py`: report figures and 95% bootstrap intervals over images (4,000 resamples). v7 PQ 0.4626 (0.454 to 0.471), v11 0.4725 (0.464 to 0.481). Paired difference v11 minus v7: +0.0099 PQ (0.0070 to 0.0129), v11 ahead in every resample; mean Dice +0.012 (0.008 to 0.017).
- `scripts/build_report.py`: writes the report in the template structure (template sentences kept verbatim) and compiles it with tectonic to `report/<version>/report.pdf`. `preamble.tex` and `main.bib` in `report/<version>/` are local stand-ins for the template's files, which could not be exported from the read-only Overleaf link.
