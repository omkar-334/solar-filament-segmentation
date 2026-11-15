# Model experiments

Date: 2026-11-03

## Setup

- Same grouped train/validation split (seed 42) as the existing compact U-Net experiment.
- Validation: 217 annotator views from 141 held-out images.
- Training: 20 epochs, 937 training views from 566 images, two sampled views per image per epoch.
- Models used the same patch training and full-image tiled inference pipeline.
- Compared threshold, minimum component area, and 10-pixel fragment merging on validation.

## Results

| Model | Validation PQ (ratio of sums) | Best tested settings | Test CSV |
| --- | ---: | --- | --- |
| Compact U-Net | 0.3373 | threshold 0.6, area 512, merge 10 | `submission_refined.csv` |
| ResNet34 encoder U-Net | **0.3798** | threshold 0.5, area 400, merge 10 | `submission_resnet34.csv` |
| ResNet34 encoder U-Net++ | **0.3888** | threshold 0.4, area 400, merge 10 | `submission_resnet34_unetpp_t040.csv` |
| U-Net++ + YOLO11s-seg detector fusion | **0.4175** | U-Net 0.5, detector 0.3, proposal growth 4 px, area 200 | `runs/detector_fusion_yolo11s_seed42/submission.csv` — public LB **0.36** |
| 50/50 probability blend: compact U-Net + ResNet34 U-Net | 0.3648 | threshold 0.5, area 512, merge 10 | Not generated |

The ResNet34 model gains 0.0425 absolute validation PQ over the compact U-Net. The tested blend is below the ResNet34 alone, so the compact model is not a useful ensemble member in this run. The ResNet34 validation score at the training-time setting (0.6 / 512 / 10) is 0.3774; the best point in the small postprocessing sweep is 0.3798.

The U-Net++ experiment used the same pretrained ResNet34 encoder, grouped split, and 20-epoch schedule. Its epoch-20 validation PQ was 0.3835 (mean per-view PQ 0.3802) at the training-time settings. An expanded sweep of thresholds 0.4/0.45/0.5 and minimum areas 64/256/400 reached **0.3888** pooled PQ (mean per-view 0.3812) at threshold 0.4, minimum area 400, and merge distance 10. This is 0.0090 above the best tested ResNet34 sweep. The small gain comes with 8 more matched instances and 8 fewer FN than threshold 0.5 at area 400; reducing area to 256 recovered many more instances but lowered PQ due to extra components. The 0.4 test candidate passed both Modal and local RLE audits; its public score remains pending.

## Detector proposals + U-Net pixel refinement

Trained a COCO-pretrained YOLO11s-seg detector for 60 epochs at 1024 px on the 937 annotator views from 566 training observations. Training and validation were split by physical image (seed 42); validation used 217 annotator views from 141 held-out observations. The detector supplies instance proposals. The U-Net++ foreground probability supplies pixel detail: detections claim foreground pixels in descending confidence order, and claimed masks are disjoint. This follows the approach reported in the public [filament-seg solution](https://github.com/ameyypawar/filament-seg).

On the same validation views, detector fusion reached **0.4175 pooled PQ** (mean per-view PQ 0.4079), versus **0.3885** for the connected-component U-Net++ baseline using the same 0.4 / 400 px / 10 px settings. This is +0.0290 absolute. Fusion reduced FP from 834 to 432, while TP fell from 992 to 910 and FN rose from 601 to 683; the PQ gain comes from a large false-positive reduction. The comparison is in-split, so a second grouped holdout should confirm the gain before we use the detector for a submission.

The best tested fusion setting was U-Net threshold 0.5, detector confidence 0.3, detector-mask dilation radius 4 px, and minimum claimed area 200 px. Nearby settings were close: threshold 0.4 with no dilation scored 0.4171, and threshold 0.4 with 4 px dilation scored 0.4163. Lowering detector confidence to 0.15 increased TP to 998 but also raised FP to 750, reducing PQ to 0.4008. Raising minimum area to 400 scored 0.4101.

The detector's final validation mask mAP50 was 0.666 (mAP50-95 0.255). The checkpoint is saved on the Modal volume at `runs/detector_fusion_yolo11s_seed42/train/weights/best.pt`. This was one detector at one scale; the published solution used several YOLO11-seg sizes and cross-fitted fusion settings, so detector diversity and a second validation fold are the next useful checks.

## Larger detector and crop-refiner experiments

Reviewed the public [two-stage/fusion study](https://github.com/HeShen-1/filament-segmentation-2026) and the cited [I079 detector/refiner model](https://www.kaggle.com/models/phuongncn/solar-filament-i079-weights/pyTorch/detector-refiner). Its main transferable choices are YOLO11m at 1536 px, detector proposals followed by a ResNet18 crop refiner conditioned on the proposal box, and four-view flip TTA. Its README reports 0.3593 mean PQ over five folds and 0.35 public LB for its fusion system; it also reports that adding same-family votes often reduced public score. The Kaggle notebook title mentioning 0.70 is not treated as a comparable score without matching metric and evaluation details.

The downloaded public I079 files match the handoff's published MD5 hashes. Those weights were trained on the competition training set, so scoring them on our held-out training split would be contaminated; keep their reported public score separate from our local validation results. Instead, test the transferable model-scale choice with a fresh detector trained on our existing grouped split and compare it against the same U-Net++ refiner.

Trained detached YOLO11m-seg at 1536 px for 50 epochs, batch 2, on the same 937/217 annotator-view grouped split. The first worker was preempted at epoch 43; a retry began from epoch 1, so I stopped it and resumed from the saved epoch-40 checkpoint. The resumed worker reached epoch 50 and commits the Modal volume on model-save callbacks. Final detector mask mAP50 was 0.603 (mAP50–95 0.214).

The same U-Net++ fusion evaluation at detector inference size 1536 reached only **0.3972 PQ** at its best point (foreground threshold 0.4, detector confidence 0.3, mask growth 4 px, minimum area 200); this is **0.0203 below** YOLO11s fusion at 0.4175. Its connected-component U-Net++ baseline was 0.3888, confirming the larger detector adds only +0.0084 here. The reference-inspired solar-disk mask did not help: its four margins scored 0.3459–0.3536. **Do not generate or submit a YOLO11m candidate** from this experiment. Detailed sweep: `runs/detector_fusion_yolo11m_1536_seed42/fusion_validation.json`.

The next experiment was the reference pipeline's distinct second stage: a seed-conditioned ResNet18 crop refiner trained on our training split, with hard negative crops and jittered proposal boxes.

Ran the selected YOLO11s fusion pipeline over all 180 test images and generated a candidate with 1,082 instance rows (6.01 instances/image on average; 3 images had no predictions). The Kaggle CSV/RLE audit passed. This candidate was submitted as `submission.csv` and scored **0.36 public LB** before the crop-refiner submission.

The ResNet34 test prediction file contains 1,465 instances over all 180 test images and passed the submission RLE audit. Its public leaderboard score is **0.32**, compared with **0.29** for the previous refined submission (+0.03 absolute). The YOLO11s fusion submission scored 0.36. Public score and held-out validation PQ are separate measurements.

Trained the seed-conditioned ResNet18 crop refiner for 10 epochs using YOLO11s proposals, jittered positive crops, hard negatives, and four-flip TTA. On the existing 217-view grouped validation split, its best setting (detector confidence 0.35, refiner threshold 0.55, minimum area 96) scored **0.4226 pooled PQ** and **0.4229 mean per-view PQ** (872 TP, 359 FP, 721 FN). The comparable YOLO11s + U-Net++ baseline was 0.4175 pooled PQ and 0.4079 mean per-view PQ. The pooled gain is small (+0.0051); the mean-per-view gain is +0.0150. This is one split, and thresholds were selected on it, so it is not by itself evidence of a major generalization gain.

Generated the crop-refiner test candidate at `runs/seed_refiner_resnet18_seed42/submission.csv`. It has **990 instance rows for 180 test images** and passed the Modal RLE audit. Config is in `runs/seed_refiner_resnet18_seed42/submission_config.json`. The candidate uses detector confidence 0.35, refiner threshold 0.55, minimum area 96, and four-flip TTA. It scored **0.37 public LB**, improving on the YOLO11s fusion score of 0.36 and becoming our current best. This is a +0.01 leaderboard gain; the local validation gain was modest, so the public result is encouraging but still needs validation across more grouped folds.

## ResNet34 false-positive / false-negative review

At the selected validation settings (threshold 0.5, minimum component area 400 px, merge distance 10 px), the instance-matching totals are 956 TP, 817 FP, and 637 FN across 217 held-out annotator views. Of those views, 193 contain at least one FP and 193 contain at least one FN; none are TN-only at the instance level. Pixel totals are 2,244,185 TP, 832,531 FP, 1,324,546 FN, and 905,762,706 TN.

The unmatched predicted instances are usually high confidence (mean probability 0.9505) but have low overlap with any reference (mean best IoU 0.1755). Missed reference instances have lower predicted coverage on average (mean probability 0.4406; mean best predicted IoU 0.1996). This points to both confident off-target structures and incomplete/weak filament coverage; the dashboard supports case-by-case visual inspection. These statistics are for the selected validation configuration, not the Kaggle test set.

## Artifacts

- Candidate submission: `submission_resnet34.csv`
- U-Net++ candidate at 0.4: `submission_resnet34_unetpp_t040.csv`
- U-Net++ candidate at 0.5: `submission_resnet34_unetpp.csv`
- ResNet34 validation review: `resnet_validation_dashboard.html` (keep `resnet_validation_assets/` beside it)
- ResNet34 validation diagnostics bundle: `resnet_validation_diagnostics.zip`
- Per-epoch logs: `resnet34_training_metrics.jsonl`
- U-Net++ per-epoch logs: `resnet34_unetpp_training_metrics.jsonl`
- U-Net++ sweep results: `unetpp_threshold_sweep.json`
- Focused U-Net++ recall sweep: `unetpp_threshold_sweep_recall.json`
- ResNet34 sweep: `resnet34_threshold_sweep.json`
- Compact + ResNet34 blend sweep: `ensemble_threshold_sweep.json`
- Detector/U-Net fusion PQ sweep: `runs/detector_fusion_yolo11s_seed42/fusion_validation.json`
- YOLO11s-seg training log: `runs/detector_fusion_yolo11s_seed42/results.csv`
- Detector/U-Net full-test candidate: `runs/detector_fusion_yolo11s_seed42/submission.csv`
- RF-DETR-Seg PQ and ensemble sweep: `runs/rfdetr_seg_medium_seed42/validation.json`
- Full-test candidate config and per-image counts: `runs/detector_fusion_yolo11s_seed42/submission_config.json`
- Detector/U-Net fusion implementation: `src/solar_filament/detector_fusion.py`
- Model definition and architecture selector: `src/solar_filament/model.py`
- Modal train, sweep, and inference jobs: `modal_app.py`

Checkpoints remain on the `solar-filament-data` Modal Volume at `runs/resnet34_unet_seed42_full/best.pt` and `runs/resnet34_unetpp_seed42_detached/best.pt`.

## DETR and loss-function review (2026-11-04)

The pasted competition notes support three practical constraints: only the provided test H-alpha image may be used at inference; supplied spine/bbox/area fields are allowed as training supervision if the model does not require them at inference; and an ensemble may be used if the submitted inference notebook is reproducible. The notes also explain that historical 0.70-era Dice results are not comparable to our current PQ leaderboard scores. A participant’s forum claims or one casual split are not reliable evidence without an out-of-fold evaluation.

**DETR choice.** RF-DETR is a real-time detector with an official instance-segmentation variant and a pretrained DINOv2 backbone; its upstream docs show segmentation model sizes from Nano through 2XLarge. This makes RF-DETR-Seg a reasonable architecture experiment because the competition labels are instances and our current pipeline’s main weakness is instance proposals. It is not guaranteed to beat YOLO11s, and its documented COCO input sizes are modest relative to 2048 px full-disk imagery, so we must verify how the fine structures survive resizing and/or tiling. Use RF-DETR-Seg, not just a box-only DETR, if the aim is end-to-end instance masks. [Official RF-DETR segmentation docs](https://github.com/roboflow/rf-detr/blob/develop/docs/learn/run/segmentation.md).

“EW-DETR” in the CVPR 2026 paper is *Evolving World Object Detection*: it targets incremental classes, domain adaptation, and unknown-object detection using low-rank adapters and objectness calibration. It is not an architecture for thin-filament segmentation, so it is not the next model to implement for this task. [EW-DETR CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Monga_EW-DETR_Evolving_World_Object_Detection_via_Incremental_Low-Rank_DEtection_Transformer_CVPR_2026_paper.html).

**Current losses.** The ResNet34 U-Net++ foreground model uses `BCE + soft Dice`; the crop refiner uses `0.5 BCE + 0.5 Dice`. These losses mainly optimize pixelwise region agreement and do not explicitly account for object count, matching at IoU 0.5, topology, or thin structures.

**Candidate loss ablation.** Keep the architecture and training split fixed, then compare the existing loss with two controlled alternatives:

1. `BCE + Dice + small clDice term` to test whether thin connected barbs improve. clDice was designed for tubular structures and rewards skeleton/mask connectivity, but can also bridge neighboring filaments; inspect merged-instance errors and PQ, not clDice alone. [clDice CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation_CVPR_2021_paper.pdf).
2. `BCE + recall-weighted Tversky` (or focal Tversky) to test whether emphasizing false negatives recovers missed filament pixels. This can increase false positives, so select the weight on grouped validation using PQ and report TP/FP/FN. The notes’ informal focal-loss suggestions have no demonstrated gain here; focal loss addresses easy-negative imbalance, which is already partly addressed in the crop refiner by hard-negative crops. [Tversky loss](https://arxiv.org/abs/1706.05721), [Focal loss](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).

Optional third arm: add a low-weight boundary-distance loss only if the diagnostic dashboard shows boundary displacement is a major failure mode. Boundary loss improves contour alignment in imbalanced segmentation, but it does not fix missed proposals or instance grouping by itself. [Boundary loss](https://arxiv.org/abs/1812.07032). Lovasz-Softmax is another IoU surrogate, but it targets semantic IoU rather than instance recognition and can optimize poorly when batch composition differs from dataset composition; it is lower priority than the two targeted arms. [Lovasz-Softmax CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf).

A separate Kaggle technical report in the notes says its consensus-target, clDice, and asymmetric-Tversky probe was negative on that team’s evaluation. That result is a caution, not a verdict: the report discusses a different metric/evaluation setup from the current competition PQ leaderboard. Any loss change should therefore earn its place on our own grouped, preferably cross-fitted, PQ validation.

**RF-DETR-Seg experiment (completed).** Fine-tuned RF-DETR-Seg-Medium from the published checkpoint at 576 px for 25 epochs on the existing seed-42 physical-image split. The COCO export contained 937 train annotator views / 566 physical images and 217 validation views / 141 physical images; all source filament categories were collapsed to one scored filament class. Training completed and the epoch-25 COCO segmentation mAP50–95 was only **0.0545**. On the competition instance-PQ sweep, the best RF-DETR-only setting was confidence 0.35 / minimum area 200, at **0.3174 PQ** (751 TP, 607 FP, 842 FN). This is far below the existing YOLO11s + crop-refiner pipeline.

The current YOLO11s + crop-refiner baseline recomputed on this exact sweep scored **0.4231 PQ** (873 TP, 357 FP, 720 FN), consistent with the prior 0.4226 result. The best tested ensemble setting (RF-DETR confidence 0.45 / minimum area 200) reached only **0.3866 PQ** (887 TP, 637 FP, 706 FN). It recovered 14 additional instances and raised matched IoU by 5.28, but added 280 false positives, so PQ fell by 0.0365. No tested RF-DETR threshold/min-area ensemble beat the baseline. **Do not use this RF-DETR checkpoint in a submission or ensemble.** This single grouped split does not establish whether a different RF-DETR resolution or training recipe could work; the current 576 px setup is not competitive.

Artifacts: `runs/rfdetr_seg_medium_seed42/validation.json` contains the complete sweep; qualitative comparison images are saved under the same run on the Modal volume. The inference job took about 21 minutes for 141 physical images. Use the crop-refiner pipeline as the current candidate and prioritize cross-fitted validation and targeted loss/proposal experiments over additional RF-DETR tuning.

**Priority:** keep the 0.4226 local validation / 0.37 public-LB crop-refiner model as the benchmark. Do not retain RF-DETR in an ensemble. Next, test improvements that target missed instances without raising FP, and validate with grouped cross-fitting before spending submissions.

## I-JEPA encoder transfer experiment (2026-11-04)

Tested the public ImageNet-1K I-JEPA ViT-H/14 encoder inside a tile-based segmentation model. The encoder received 448 px grayscale tiles repeated to RGB with ImageNet normalization; positional embeddings were interpolated from its 224 px pretraining size. A local convolutional decoder and high-resolution image skips were added to preserve filament boundaries. The first 28 transformer blocks were frozen; the last four were tuned at 1e-5 while the segmentation decoder used 2e-4. Training used the same grouped seed-42 split for 10 epochs, with 448 px crops and a checkpoint committed after every epoch.

The fixed training cutoff scored **0.2380 PQ**. A 3-threshold × 3-area grouped-validation sweep found a best score of only **0.2760 PQ** at threshold 0.45 / minimum area 400 (904 TP, 1,762 FP, 689 FN; mean per-view PQ 0.2701). On the same split the YOLO11s + crop-refiner baseline is **0.4231 PQ** (873 TP, 357 FP, 720 FN). I-JEPA recovered 31 more references, but added 1,405 false positives and had lower matched IoU. **Do not use this segmentation checkpoint as a candidate or ensemble member.** This rejects the current direct semantic-segmentation transfer setup; it does not test using I-JEPA features inside detector proposals.

Artifacts: `runs/ijepa_vith14_seg_seed42/threshold_sweep.json`; model: `src/solar_filament/ijepa_seg.py`; detached Modal training and evaluation entrypoints: `modal_app.py` (`ijepa_train`, `ijepa_eval`). Training completed in about 7 minutes; the full validation prediction plus threshold sweep took about 16 minutes.

## Grouped OOF proposals for the crop refiner (2026-11-05)

Trained three YOLO11s-seg folds for 60 epochs each at 1024 px. The folds were grouped by physical image within the existing 566-image training split, keeping all annotator views together. Each detector generated proposals only for its held-out fold; all 566 training images received OOF proposals. The existing 141-image / 217-view validation split remained untouched. Fold checkpoints and proposal files were committed to the Modal volume; the merged proposals contain 12,287 detections.

Fine-tuned the existing ResNet18 crop refiner for five epochs (learning rate 1e-4) from its previous best checkpoint, using OOF detector crops, hard negative/merged proposals, and medoid-view ground-truth crops. Best held-out score was **0.4249 pooled PQ / 0.4260 mean per-view PQ** at detector confidence 0.35, refiner threshold 0.65, minimum area 96 (877 TP, 359 FP, 716 FN). The previous refiner's best sweep was **0.4226 pooled PQ** at 0.35 / 0.55 / 96 (872 TP, 359 FP, 721 FN). At the shared 0.35 / 0.55 / 96 setting, OOF fine-tuning scored **0.4239**, a +0.0013 change. The best-sweep gain is only +0.0023 and comes with five more matched instances at the same FP count; treat this as a small local improvement, not evidence for a meaningful public-LB jump.

**Decision:** keep the 0.37 public-LB submission as the benchmark; the local gain is small. At the user's request, generated a test candidate using detector confidence 0.35, refiner threshold 0.65, minimum area 96, and four-flip TTA. The CSV has **990 rows for 180 images** and passed the RLE/overlap audit; the user will submit it. Local files: `submission_oof_refiner.csv` and `submission_oof_refiner_config.json`. Modal artifacts: `/data/runs/oof_yolo11s_seed42_3fold/oof_proposals.json`, `/data/runs/oof_crop_refiner_resnet18_seed42/validation.json`, and the OOF refiner checkpoint on the `solar-filament-data` Modal volume. The OOF implementation is in `modal_app.py` and `src/solar_filament/crop_refiner.py`.

Tested the three OOF YOLO checkpoints as a proposal ensemble on the same untouched validation split, using box NMS at IoU 0.5. Best pooled PQ was **0.4227** at detector confidence 0.35 / refiner threshold 0.55 (907 TP, 428 FP, 686 FN), below the single full-training detector + OOF refiner result of 0.4249. The ensemble gained 30 TP but added 69 FP. **Do not use this ensemble for a submission.** Results: `/data/runs/oof_crop_refiner_resnet18_seed42/ensemble_validation.json`.

## Recommended next experiments

1. Keep the 0.37 crop-refiner submission as the benchmark; do not spend submissions on tiny parameter variations without a validation-backed hypothesis.
2. The transferable next architecture change is a new Mask R-CNN detector trained and validated on our grouped split; the 0.70 recipe's D4 TTA did not improve our current YOLO/refiner stack. Do not scale YOLO11m: it underperformed YOLO11s on this split, and the current RF-DETR/I-JEPA models did not improve the pipeline.

## Active run: YOLO11x native instance segmentation (2026-11-05)

Completed a detached 50-epoch YOLO11x-seg run at 1024 px on the existing grouped seed-42 split, batch size 4 on an NVIDIA A10. Modal call ID: `fc-01M6NJYEV6XNQ1T7AWWGRAVHSQ`; app: `ap-FZGbGM1o7F3tdmS5nDA02V`. Setup passed AMP checks and used all 937 train views / 217 validation views (566 / 141 physical images). Training took 5,677 s (94.6 min). Train losses fell from 2.434 / 2.559 / 2.383 / 1.986 at epoch 1 to 1.477 / 1.719 / 1.212 / 1.367 at epoch 50 (box/seg/class/DFL). Mask mAP50/mAP50–95 peaked at 0.635/0.237 in epoch 44; final was 0.631/0.234. Validation loss showed inf/nan at epochs 1–2 but became finite at epoch 3 and remained finite. `last.pt` and `best.pt` are persisted on the Modal volume.

**Native mask PQ result:** on 141 held-out physical images / 217 views, the best threshold sweep scored **0.3366 pooled PQ** (0.3363 mean per-view) at confidence 0.35 / minimum area 400, with 691 TP / 341 FP / 902 FN. The lower-confidence setting 0.20 raised TP to 853 but also raised FP to 772, leaving PQ at 0.3355. This is below the current crop-refiner's 0.4249 local PQ, despite YOLO's higher COCO mAP; **do not submit native YOLO11x masks**. Full sweep: `runs/yolo11x_seg_seed42_e50/instance_validation.json`.

To check whether the model still helps as a proposal generator, an evaluation pairing YOLO11x proposals with the existing OOF ResNet18 crop refiner is running as Modal call `fc-01M6NXDEQEPY4HBBTAK4PAK9FZ`, app `ap-IUCT39n68zm6fivYNrt9Tw`. Keep the existing 0.37 public submission unless this combination shows a meaningful held-out PQ gain.

## Reference pipeline audit and D4 TTA experiment (2026-11-05)

**Score attribution correction:** the Kaggle notebook named “LB 0.70 | Topology-Safe Solar Filaments Handoff” currently shows **0.34 public / best score**, despite its title. It uses the public I079 detector/refiner weights. The separately documented 0.70 system is Felipe KNBIS's report: Mask R-CNN R50-FPN-v2 at 1024, strict mask NMS, then a ResNet18 crop refiner. It uses eight D4 rotations/reflections and averages refiner logits before sigmoid. Treat the report's score and recipe as that team's report, not as the score of the similarly titled Fususu notebook.

Our crop representation already has grayscale, CLAHE, and proposal seed channels; the refiner uses square 256 px crops with 1.5x context (minimum side 96); and its training uses jittered boxes and hard negatives. These overlap with the public crop-refiner design, so we did not repeat those changes.

**Controlled TTA ablation:** evaluated both the existing four-view probability average (`identity/h/v/hv`) and full eight-view D4 logit average on the same 141 physical-image / 217-view held-out split, OOF-refiner checkpoint, YOLO11s proposals, and threshold grid. Best four-view result: **0.424889 pooled PQ** at detector confidence 0.35 / refiner threshold 0.65 (877 TP, 359 FP, 716 FN). Best D4 result: **0.424810** at 0.35 / 0.55 (877 TP, 361 FP, 716 FN). At the shared 0.35 / 0.65 setting, D4 scored **0.424024** vs **0.424889** for four-view. This is a negligible best-sweep regression and a 0.0009 regression at the existing selected threshold, with no recovered false negatives. **Decision: retain four-view TTA; do not generate or submit a D4 candidate.**

The paired evaluation took 12m28s on Modal for 141 physical images while evaluating both modes. Full results are saved locally at `runs/d4_tta_oof_refiner_seed42/validation.json` and on the Modal volume at `/data/runs/d4_tta_oof_refiner_seed42/validation.json`.

Sources reviewed: [Kaggle notebook metadata](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff), the team's [technical report](https://github.com/FELIPEACASTRO/solar-filament-2026/blob/master/report/TECHNICAL_REPORT.md), and its [competitor notebook audit](https://github.com/FELIPEACASTRO/solar-filament-2026/blob/master/report/COMPETITOR_NOTEBOOKS_ANALYSIS_20260718.md). The report documents the D4 recipe and Mask R-CNN detector; its score claims are self-reported by that team.
