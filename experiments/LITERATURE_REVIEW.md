# MAGFiLO Filament Segmentation: Literature Review and Experiment Plan

**Prepared:** 3 November 2026  
**Task:** Kaggle Solar Filament Segmentation 2026; 2048×2048 GONG H-alpha images; instance masks scored by Panoptic Quality (PQ).

## Executive summary

Our bottleneck is not simply a weak encoder. We trained the model to predict one foreground mask and then used connected components as a stand-in for filament instances. That discards the instance identity, centerline, and branching structure already present in the labels. It explains why ResNet34 reached a 0.32 public score and our best current local validation PQ is 0.3888, while a deeper/stronger semantic segmenter is unlikely to bridge the gap to the reported ~0.70 leaderboard entries by itself.

The highest-value next direction is **instance-aware, spine-conditioned segmentation**:

1. Keep the polygon for each filament as a separate training target.
2. Supervise the annotated spine as an instance-specific centerline signal.
3. Predict per-instance masks or centerline proposals, then assign foreground pixels using the predicted instance structure.
4. Train and select against the exact pooled, per-annotator PQ evaluator on grouped observations, with a second date-grouped fold to check that gains are not split luck.

This is not the same as simply adding a detector before the current model. Detector-plus-U-Net/refiner systems do improve some validation scores, but the gain reported in one public reproduction was modest (0.371 to 0.421 local PQ; 0.37 public). The larger conceptual shift is to make the spine and instance identity part of the learned representation from the beginning.

## Scope

This review covers **NeurIPS, ICML, ICLR, CVPR, and BMVC**, plus closely related work on MAGFiLO and tubular structures. I prioritized primary conference proceedings, publisher papers, and preprints whose methods map to thin, low-contrast, branching, or instance-level segmentation. Claims from preprints and repository reports are labeled and are not treated as equivalent to peer-reviewed results.

“Recent” here means primarily 2024–2026. I prioritized primary conference proceedings, publisher papers, and arXiv/SSRN preprints whose methods map to thin, low-contrast, branching, or instance-level structures. Claims from preprints and GitHub reports are labeled as such and are not treated as equivalent to peer-reviewed evidence.

## Literature findings

### Directly on solar filaments / MAGFiLO

| Work | Method and reported finding | Relevance and limitation for us |
|---|---|---|
| [A Modern ConvNet for Solar Filament Detection (MORDEN, 2026)](https://arxiv.org/abs/2607.24525), arXiv preprint | Proposes a multiscale, oriented, dendritic semantic segmenter; evaluates input scaling and DenseCRF refinement, and uses DBSCAN for fragment integration. The paper explicitly notes that semantic segmentation does not distinguish individual filaments and therefore requires a separate grouping step. | Relevant for scale, orientation, contrast refinement, and classical grouping. It is semantic segmentation followed by postprocessing, so it does not solve our central instance-identity problem by itself. It uses other solar datasets and reports semantic metrics, not our exact pooled PQ. |
| [EdgeAttNet: Towards Barb-Aware Filament Segmentation (2025)](https://arxiv.org/abs/2509.02964), arXiv preprint | Adds a learnable edge map to guide bottleneck attention and reports better filament/barb segmentation than U-Net and U-Net-transformer baselines on MAGFiLO. | Strong domain match and a promising architectural ablation for low-contrast edges and barbs. It is still principally a semantic mask model; the reported improvement does not establish instance PQ improvement. |
| [MAGFiLO dataset paper (Scientific Data, 2024)](https://www.nature.com/articles/s41597-024-03876-y) | MAGFiLO v1.0 contains 10,244 filament annotations across 1,593 observations. Each object annotation includes a polygon, spine, bounding box, and chirality. | Most actionable fact for our project: the per-filament spine is available in our training JSON. We should exploit it directly rather than collapse all polygons into a binary mask. |
| [Detector + U-Net reference implementation](https://github.com/ameyypawar/filament-seg), community reproduction | Reports 0.371 validation PQ for U-Net, 0.383 for detectors alone, 0.418–0.421 for combinations, and 0.37 public. It also reports substantial inter-annotator disagreement. | Useful baseline evidence: proposals help, but the observed lift is not a credible path from our current score to ~0.70 alone. This is a repository/report, not a peer-reviewed paper, and scores depend on its split and pipeline. |

### NeurIPS, ICML, and ICLR

| Work | Method and result | Relevance and limitation |
|---|---|---|
| [GraphMorph: Tubular Structure Extraction by Morphing Predicted Graphs (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e991aa4cd2fdf0014fba2f000f542d0-Abstract-Conference.html) | Uses a graph decoder for branch-level tubular features and a morph module that combines the predicted graph with a centerline probability map; reports topology and segmentation improvements on several datasets. | Closest venue match to our proposed direction. The annotated MAGFiLO spine can anchor a filament's main path; polygon-derived structure can supervise branches. A graph may then help organize each instance before producing its mask. It is not a ready-made 2D multi-instance solution; its source-code status and exact training cost need checking. |
| [Structure-Aware Image Segmentation with Homotopy Warping (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html) | Identifies topologically critical pixels using digital topology and distance transforms, then weights training around those locations. | Established precedent for emphasizing structurally important pixels, including in fine-scale structures. Older and topology-focused; it does not model filament instance identity. Use as background for structural loss design, not the first implementation to port. |
| [Topology-Aware Uncertainty for Image Segmentation (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract.html) | Estimates uncertainty by structural units such as branches and connections rather than only pixel-wise uncertainty. | Useful for a later annotation/error-review tool: rank missed or ambiguous spine segments for visual review. It targets uncertainty estimation and annotation workflows, not directly PQ optimization. |
| [SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation (ICML 2026)](https://proceedings.mlr.press/v306/zhang26bu.html) | A small structure-aware vision model for crack segmentation, using geometry-guided structure transformations and cross-scale features. | Cracks share thin, branching appearance and make this a useful ICML architecture reference. It is semantic crack segmentation, not instance segmentation; transfer is speculative and lower priority than spine-conditioned instance supervision. |
| [When Semantic Segmentation Meets Frequency Aliasing (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bcade016e3004543b289b33e7deb7472-Abstract-Conference.html) | Identifies false responses, merging, and displacement associated with aliasing during encoder downsampling; proposes anti-aliasing filters and frequency mixing. | Directly motivates preserving fine, weak branches through the encoder. Test anti-aliased downsampling or a high-resolution/detail path as a controlled ablation. It improves semantic segmentation, not automatically instance PQ. |
| [Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html) | Builds a graph from prediction and label components, identifies topology-critical regions, and derives a general topology-aware loss with formal guarantees. | More principled than adding a generic skeleton penalty. Potential auxiliary loss after we preserve separate filament labels. It targets topology, however, while PQ also requires correct instance count and mask overlap; topology alone cannot solve merges/splits. |

### CVPR and adjacent top vision venues

| Work | Method and result | Relevance and limitation |
|---|---|---|
| [Dual-level Adapter Boosting Prompt-free Curvilinear Structure Segmentation (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Dual-level_Adapter_Boosting_Prompt-free_Curvilinear_Structure_Segmentation_CVPR_2026_paper.html) | Adapts SAM for prompt-free curve segmentation with internal/external adapters and a two-stage decoder, reporting transfer across 12 curvilinear datasets with limited annotated examples. | A credible low-data transfer experiment. We can test the released implementation as a semantic/curve baseline, but need an instance grouping head to optimize PQ. Check code, licensing, and inference cost before committing GPU time. |
| [DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_DeformCL_Learning_Deformable_Centerline_Representation_for_Vessel_Extraction_in_3D_CVPR_2025_paper.html) | Represents tubular structures as centerline nodes plus edges rather than only a voxel mask; uses a cascaded learning pipeline to improve connectivity and robustness. | Strong conceptual match to ordered filament spines. Adapt the representation to 2D using the manual spine polylines as seeds/targets, then learn per-instance masks around each path. The paper is 3D vessel extraction; its complete architecture should not be copied wholesale. |
| [HarmonySeg: Tubular Structure Segmentation with Deep-Shallow Feature Fusion and Growth-Suppression Balanced Loss (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_HarmonySeg_Tubular_Structure_Segmentation_with_Deep-Shallow_Feature_Fusion_and_Growth-Suppression_Balanced_Loss_ICCV_2025_paper.html) | Fuses shallow and deep features, supplies vesselness cues, and balances target growth against suppression to handle small structures and incomplete labels. | Useful for preserving shallow-detail information and controlling the precision/recall tradeoff. Vesselness should be tested as an input/feature cue, not assumed to transfer from vessels to solar filaments. |
| [SCSegamba: Lightweight Structure-Aware Vision Mamba for Crack Segmentation (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_SCSegamba_Lightweight_Structure-Aware_Vision_Mamba_for_Crack_Segmentation_in_Structures_CVPR_2025_paper.pdf) | Adds structure-aware scanning and a gated convolutional bottleneck for crack morphology. | Crack segmentation is another thin, branching, low-contrast problem. This is a lower-priority architecture comparison than direct spine supervision because it does not address instance identity. |

### BMVC

| Work | Method and result | Relevance and limitation |
|---|---|---|
| [Unsupervised Domain Adaptation for Tubular Structure Segmentation Across Different Anatomical Sources (BMVC 2024)](https://bmvc2024.org/proceedings/342/) | Uses grayscale transformation, target-domain noise, and mask adjustment to handle differences across source domains. | MAGFiLO spans six GONG sites and years of acquisition. First quantify site/year effects on our held-out errors; then test contrast normalization and physically plausible photometric augmentation. Unsupervised adaptation to test images is a separate step and must follow competition rules. |
| [Topology-Preserving Hard Pixel Mining for Tubular Structure Segmentation (BMVC 2023)](https://proceedings.bmvc2023.org/846/) | Mines hard pixels while preserving tubular topology. | A potentially useful way to focus on weak branches and ambiguous boundaries. Older and primarily semantic/tubular, so use only if error review confirms branch omissions are a dominant source of PQ loss. |

### Supporting topology and continuity work

| Work | Method and caveat |
|---|---|
| [Skeleton Recall Loss for Connectivity Conserving Segmentation (ECCV 2024)](https://arxiv.org/abs/2404.03010) | Adds recall supervision on a tubular skeleton and reports gains on several segmentation datasets. It is inexpensive to add as an auxiliary objective, but a later [reassessment](https://arxiv.org/abs/2508.11374) reports inconsistent gains and points out gradient limitations. Treat it as an ablation, not a presumed fix. |
| [Preserving Instance Continuity and Length with Connectivity-Aware Losses (ICCV 2025 Workshop)](https://openaccess.thecvf.com/content/ICCV2025W/BIC/html/Szustakowski_Preserving_instance_continuity_and_length_in_segmentation_through_connectivity-aware_loss_ICCVW_2025_paper.html) | Introduces centerline and topology losses for continuity through weak-signal gaps. Results are on 3D microscopy; use as an idea for missing-spine segments, not direct evidence of a PQ gain here. |
| [Spine-Anchored Branch Recall for Solar Filament Segmentation (SSRN, 2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7441885), preprint | Uses annotated spine endpoints to separate principal paths from lateral skeleton components; reports improved manual-spine coverage on 106 held-out observations, with small Dice/clDice reductions. | Very close to our data and task. It is not peer-reviewed, and reported metrics are spine coverage/Dice rather than competition PQ. It supports running a controlled spine-recall experiment, not adopting the claimed result without replication. |

## What the evidence implies for our pipeline

1. **Preserve instance labels in the training target.** Our present semantic foreground mask throws away which pixels came from which filament and every spine path. This is the clearest mismatch between training objective and competition metric.
2. **Learn structure at high resolution.** Thin barbs are vulnerable to repeated downsampling. Use native-resolution crops or overlapping large crops, retain a high-resolution decoder path, and compare anti-aliased downsampling/detail fusion.
3. **Use spines as structural supervision, not as a substitute for masks.** Spines indicate a filament's main path, but the scored object is its full polygon including branches. The model must learn both.
4. **Keep instance separation explicit.** The output needs a set of non-overlapping per-filament masks. A binary mask plus connected components will continue to merge neighbors and fragment weak filaments.
5. **Treat topology losses as secondary.** clDice/Skeleton Recall/Topograph may improve continuity. They do not by themselves fix wrong instance count or annotator disagreement. Measure PQ and split/merge statistics after each change.
6. **Measure domain shift and annotation noise.** Split by physical observation, not annotator row; inspect performance by GONG site, year, brightness, and annotator multiplicity. Multi-annotator agreement places a real ceiling on exact instance matching.

## Our measured baseline

From the local experiment ledger [MODEL_EXPERIMENTS.md](MODEL_EXPERIMENTS.md):

- ResNet34 public score: **0.32**; best tested local validation PQ: **0.3798**.
- ResNet34 U-Net++ best local validation PQ: **0.3888** at threshold 0.4, minimum component area 400 px, and merge distance 10 px; public result is not recorded there.
- At the ResNet34 selected setting, validation totals were 956 TP, 817 FP, and 637 FN. For the 637 FN, 356 had some predicted overlap below the IoU threshold and 281 had no overlap. This indicates both coverage/continuity failures and the instance grouping problem.
- The detector-plus-refiner community report's 0.421 validation / 0.37 public is a modest gain, not evidence that the architecture swap closes our gap.

## Proposed experiment sequence

### Phase 0: Lock the evaluation and data audit (first)

- Freeze one evaluator matching the organizer's pooled per-annotator PQ and IoU matching rule. Test it against the supplied self-evaluation notebook on a small known subset.
- Keep every annotator view for scoring, but group train/validation by the physical image/date so duplicate annotator drawings cannot cross the split.
- Add one second grouped fold or 3-fold cross-validation for any claimed improvement. Report pooled PQ, mean per-view PQ, TP/FP/FN, fragmentation, merge rate, per-site PQ, and paired bootstrap intervals.
- Inspect annotation polygons and spine polylines together on representative images before generating targets. Confirm coordinate ordering, image dimensions, and transformed geometry under every augmentation.

### Phase 1: Spine-supervised instance baseline (highest priority)

**Question:** Does explicit instance/spine supervision produce a real PQ gain over the same U-Net++ backbone?

- Keep the current backbone and fixed training budget; this isolates the target/loss change from model capacity.
- Create separate object targets from each COCO polygon and its matching spine polyline.
- Add a spine heatmap or centerline-probability head, supervised per object. Add a local affinity/direction target along each spine so adjacent instances are distinguishable.
- Keep the polygon mask head. Train with polygon BCE/Dice plus a modest spine/affinity loss. Avoid a large topology-loss weight until ablations show it helps.
- At inference, use predicted centerline peaks/paths as instance seeds and assign foreground pixels to seeds using learned affinity or a distance/embedding field; enforce disjoint masks.
- Compare three variants under identical seeds/splits: current semantic+components baseline; mask+spine auxiliary head with current postprocess; mask+spine+instance assignment head.

**Success criterion:** Improvement must appear in pooled PQ on both grouped folds, with fewer FN and without a compensating FP explosion. A rise only in skeleton coverage or Dice does not qualify.

### Phase 2: Preserve fine branches

- Test native/large-crop training with overlap-tiled inference and a high-resolution decoder skip.
- Add one anti-aliased downsampling or edge-guided feature variant (ICLR aliasing / EdgeAttNet inspiration).
- Keep augmentation physically plausible: flips/rotations and modest contrast/gamma/noise; verify the spine target is transformed with the image.

**Success criterion:** Fewer missed spine segments and higher TP/PQ at fixed FP. Reject a variant if it only increases semantic recall but creates extra instance fragments.

### Phase 3: Compare alternative grouping / topology methods

- Compare watershed/affinity-based assignment seeded by the spine head against a detector-guided method.
- Add exactly one topology-aware objective (Topograph, centerline recall, or a spine-balanced recall loss) at a time.
- Run SAM-adapter/SACM or EdgeAttNet only after confirming implementation, weights, input resolution, and competition licensing; evaluate as a controlled backbone/feature-transfer comparison.

### Decision rule for submissions

- Do not select by public leaderboard alone. Select from cross-validated PQ and failure-type counts, then make one audited submission from the best frozen configuration.
- Record model hash, training seed, fold membership, checkpoint, postprocessing parameters, and CSV audit for each candidate.
- The official task rules allow external data sources but prohibit using other ground-truth metadata for model training. The public MAGFiLO release includes the competition observations; do not use labels for competition test observations to train or tune. This also means unusually high public scores may not be directly comparable without inspecting the exact pipeline and data provenance.

## Practical priority ranking

1. **Per-instance polygon + spine supervision with explicit instance assignment** — highest expected relevance and uses labels we already own.
2. **High-resolution / anti-aliasing detail preservation** — addresses fine barbs and branch recall.
3. **GONG-site and date robustness** — likely useful if error analysis shows systematic acquisition differences.
4. **Topology-aware loss** — useful targeted ablation after the representation retains instances.
5. **Another plain larger encoder or U-Net++ variant** — low priority; our U-Net++ gain was only 0.009 local PQ over ResNet34.
6. **Detector plus mask refiner alone** — keep as comparator, not the main research direction.

## References

- Ahmadzadeh et al., [A dataset of manually annotated filaments from H-alpha observations](https://www.nature.com/articles/s41597-024-03876-y), Scientific Data, 2024.
- Chen et al., [When Semantic Segmentation Meets Frequency Aliasing](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bcade016e3004543b289b33e7deb7472-Abstract-Conference.html), ICLR, 2024.
- Zhang et al., [GraphMorph: Tubular Structure Extraction by Morphing Predicted Graphs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e991aa4cd2fdf0014fba2f000f542d0-Abstract-Conference.html), NeurIPS, 2024.
- Gupta et al., [Topology-Aware Uncertainty for Image Segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract.html), NeurIPS, 2023.
- Hu, [Structure-Aware Image Segmentation with Homotopy Warping](https://proceedings.neurips.cc/paper_files/paper/2022/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html), NeurIPS, 2022.
- Zhang et al., [SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation](https://proceedings.mlr.press/v306/zhang26bu.html), ICML, 2026.
- Kirchhoff et al., [Skeleton Recall Loss](https://arxiv.org/abs/2404.03010), ECCV, 2024.
- An et al., [Unsupervised Domain Adaptation for Tubular Structure Segmentation Across Different Anatomical Sources](https://bmvc2024.org/proceedings/342/), BMVC, 2024.
- Lux et al., [Topograph](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html), ICLR, 2025.
- Zhao et al., [DeformCL](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_DeformCL_Learning_Deformable_Centerline_Representation_for_Vessel_Extraction_in_3D_CVPR_2025_paper.html), CVPR, 2025.
- Huang et al., [HarmonySeg](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_HarmonySeg_Tubular_Structure_Segmentation_with_Deep-Shallow_Feature_Fusion_and_Growth-Suppression_ICCV_2025_paper.html), ICCV, 2025.
- Liu et al., [SCSegamba](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_SCSegamba_Lightweight_Structure-Aware_Vision_Mamba_for_Crack_Segmentation_in_Structures_CVPR_2025_paper.pdf), CVPR, 2025.
- Zhu et al., [Dual-level Adapter Boosting Prompt-free Curvilinear Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Dual-level_Adapter_Boosting_Prompt-free_Curvilinear_Structure_Segmentation_CVPR_2026_paper.html), CVPR, 2026.
- Hu et al., [A Modern ConvNet for Solar Filament Detection](https://arxiv.org/abs/2607.24525), arXiv preprint, 2026.
- Solomon et al., [EdgeAttNet](https://arxiv.org/abs/2509.02964), arXiv preprint, 2025.
- Hao, [Spine-Anchored Branch Recall for Solar Filament Segmentation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7441885), SSRN preprint, 2026.
- Official competition [evaluation rules and metric](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).
