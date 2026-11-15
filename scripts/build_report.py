r"""Write and compile the four-page technical report for one pipeline version.

The text follows the organisers' Overleaf template (acmart, sigconf): the black template
sentences are kept verbatim, the gray guide text is replaced. Numbers are read from
results/report_figures/<version>/numbers.json and the reproduction logs, figures from
report/<version>/figures (scripts/report_figures.py).

Usage: python scripts/build_report.py --version=v7|v11 [--no_compile]
Writes report/<version>/{main.tex,refs.bib,preamble.tex,main.bib,report.pdf} and
results/build_report/<version>/summary.json. preamble.tex and main.bib are local stand-ins
for the template's own files: in Overleaf keep the template's two files and add main.tex,
refs.bib and figures/.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = r"""% Local stand-in for the template's preamble.tex (same commands, used to compile outside Overleaf).
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\newif\ifguideStyle
\newcommand{\guide}[1]{\ifguideStyle{\color{gray!65}\itshape #1}\fi}
\setcopyright{none}
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagestyle{plain}
\acmConference{}{}{}
\acmDOI{}
\acmISBN{}
"""

MAINBIB = r"""% Local stand-in for the two entries of the template's main.bib.
@misc{filament-segmentation-2026,
  author = {Azim Ahmadzadeh and Dustin J. Kempton and Qin Li and Alexei A. Pevtsov},
  title = {Solar Filament Segmentation Challenge 2026},
  year = {2026},
  howpublished = {\url{https://www.kaggle.com/competitions/filament-segmentation-2026}},
  note = {Kaggle}
}
@article{ahmadzadeh2024dataset,
  author = {Ahmadzadeh, Azim and Adhyapak, Rohan and Chaurasiya, Kartik and Nagubandi, Laxmi Alekya and Aparna, V. and Martens, Petrus C. and Pevtsov, Alexei and Bertello, Luca and Pevtsov, Alexander and Douglas, Naomi and McDonald, Samuel and Bawa, Apaar and Kang, Eugene and Wu, Riley and Kempton, Dustin J. and Abraham, Aya and Landeros, Jeremy and Angryk, Rafal A.},
  title = {A dataset of manually annotated filaments from {H}-alpha observations},
  journal = {Scientific Data},
  volume = {11},
  pages = {1031},
  year = {2024},
  doi = {10.1038/s41597-024-03876-y}
}
"""

REFS = r"""@misc{yolo11,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics {YOLO11}},
  year = {2024},
  howpublished = {\url{https://github.com/ultralytics/ultralytics}},
  note = {Software, version 8.3}
}
@inproceedings{unet,
  author = {Olaf Ronneberger and Philipp Fischer and Thomas Brox},
  title = {{U-Net}: Convolutional Networks for Biomedical Image Segmentation},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year = {2015},
  pages = {234--241}
}
@inproceedings{resnet,
  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title = {Deep Residual Learning for Image Recognition},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016},
  pages = {770--778}
}
@inproceedings{lightgbm,
  author = {Guolin Ke and Qi Meng and Thomas Finley and Taifeng Wang and Wei Chen and Weidong Ma and Qiwei Ye and Tie-Yan Liu},
  title = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2017}
}
@inproceedings{pq,
  author = {Alexander Kirillov and Kaiming He and Ross Girshick and Carsten Rother and Piotr Doll{\'a}r},
  title = {Panoptic Segmentation},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019},
  pages = {9404--9413}
}
@inproceedings{sam2,
  author = {Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman R{\"a}dle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  title = {{SAM} 2: Segment Anything in Images and Videos},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025}
}
@inproceedings{msrcnn,
  author = {Zhaojin Huang and Lichao Huang and Yongchao Gong and Chang Huang and Xinggang Wang},
  title = {Mask Scoring {R-CNN}},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019},
  pages = {6409--6418}
}
"""

HEAD = r"""\documentclass[sigconf]{acmart}
\input{preamble}

% Guide styling: \guideStyletrue = show gray italic hints; \guideStylefalse = omit them
% \guideStyletrue
\guideStylefalse

\begin{document}

\title{@@TITLE@@}
\subtitle{A Solution to the Solar Filament Segmentation Challenge 2026} % DO NOT CHANGE

\author{@@AUTHOR@@}
\affiliation{%
  \country{@@COUNTRY@@}
}
\email{@@EMAIL@@}

\begin{abstract}
This report describes our solution to the Solar Filament Segmentation Challenge~2026, a Kaggle competition on automatic segmentation of solar filaments in GONG H-$\alpha$ observations. @@ABSTRACT@@
\end{abstract}

\maketitle

% ==================================== %
%             INTRODUCTION             %
% ==================================== %
\section{Introduction}
This report goes over the details of our submitted solution to the Solar Filament Segmentation Challenge~2026 \cite{filament-segmentation-2026}. The challenge, launched on July~10,~2026, is organized by the Earth-Space AI Research (ESAIR) Lab\footnote{\href{https://www.esairlab.com/home}{www.esairlab.com/}}. This challenge utilizes the MAGFiLO dataset for evaluation of the segmentation algorithms \cite{ahmadzadeh2024dataset}.
Submissions are ranked by Panoptic Quality (PQ) \cite{pq}: a predicted and a labelled filament match when their IoU exceeds 0.5, and $\mathrm{PQ}=\sum_{TP}\mathrm{IoU}/(|TP|+\tfrac12|FP|+\tfrac12|FN|)$, pooled over the label sets (``views'') of all annotators of an image.

One measurement shaped the design. On the training images labelled by several annotators, one annotator scored against another reaches a PQ of only 0.340. Annotators mostly disagree on \emph{which} dark structures are filaments and where one ends, less on the outline of a filament both drew. A model that imitates one annotator therefore scores poorly. Our pipeline proposes many candidate instances with high recall, refines each one, and then decides per candidate whether the \emph{average} annotator would have drawn it. Adding an instance adds its matched IoU to the numerator of PQ and at least $\tfrac12$ to the denominator, so an instance should be kept when its expected matched IoU exceeds about $\mathrm{PQ}/2\approx 0.23$. We learn exactly this quantity and threshold it. @@INTRO_END@@

% ==================================== %
%             METHODOLOGY              %
% ==================================== %
\section{Methodology}\label{sec:methodology}
The solution combines existing architectures. Its own contributions are the decomposition in Figure~\ref{fig:pipeline}, the selection target derived from the metric, and the training targets built from several annotators. Inference uses the test image only.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_pipeline.pdf}
  \caption{Data pipeline at inference. Blue: segmentation networks, orange: learned keep or drop decisions, green: rule-based step.}
  \label{fig:pipeline}
\end{figure*}

\textbf{Pre-processing.} The 2048$\times$2048 gray image is used as is. Networks that work on crops receive, next to the gray crop, a contrast-limited adaptive histogram equalisation (CLAHE) of it. No limb-darkening correction or disk masking is applied.

\textbf{Stage 1: proposals.} Two YOLO11s-seg detectors \cite{yolo11}, COCO-pretrained and fine-tuned at 1024~px and 1536~px input, propose boxes with confidence of 0.05 or more. The threshold is low on purpose: recall matters here and the keep decision is taken later. Each annotator view is a separate training sample.

\textbf{Stage 1: crop refinement.} The detector masks are coarse (predicted at a quarter of the input resolution), so every box is segmented again in its own crop by a box-seeded U-Net \cite{unet} with a ResNet encoder \cite{resnet}. Its input is the gray crop, the CLAHE crop and the box mask. For the 1024~px detector a ResNet18 model at 256~px and a ResNet34 model at 384~px are averaged, for the 1536~px detector a ResNet18 model at 256~px is used, each over four flips.

\textbf{Paste.} Candidates of both detectors are pasted into one label image in order of detector score at probability 0.65. A pixel belongs to the first candidate that claims it and instances under 96~px are dropped, so instances never overlap and duplicates of the two detectors collapse.

\textbf{Selection value.} A LightGBM regressor \cite{lightgbm} predicts, for every pasted instance, its matched IoU averaged over the annotator views of the image (zero for a view in which it matches nothing). This is the expected contribution of the instance to the numerator of PQ. The 17 features are detector score, refiner confidence, size, shape, contrast and the agreement of detector mask and refiner mask. @@SELECT@@

@@STAGE2@@

\textbf{Post-processing.} Stage-2 masks replace the stage-1 masks under the same pixel-claiming rule, and each instance becomes one COCO run-length row. No morphological operation, merging rule or hole filling is applied: none of the variants we tried improved the validation score.

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_examples.jpg}
  \caption{Predicted instances for test image @@EXAMPLE@@ (left, one colour per instance) and three zooms with instance outlines.}
  \label{fig:examples}
\end{figure*}

\textbf{Data and sampling.} Only the training split of the competition data was used: 707 images, 1,154 annotator views and 8,199 labelled filaments. No external images or labels, no public MAGFiLO test labels and no hard-coded predictions were used. Pretrained initialisations are public (COCO for YOLO11, ImageNet for the ResNets@@PRETRAINED@@). There is no fixed validation set. The 707 images are split into five folds, grouped by image so that all views of an image share a fold. For every fold a detector, its refiners and the stage-2 models are trained on the other four folds, which yields out-of-fold predictions for all 707 images. The selector is trained on these out-of-fold candidates with a second grouped five-fold split, so the candidates it learns from have the quality it will meet at test time. Test predictions come from the same recipes trained on all 707 images (detectors, refiners, dense U-Net), from the five fold models averaged (stage 2) and from the selector trained on all out-of-fold candidates. Figure~\ref{fig:examples} shows predictions on a test image.

% ==================================== %
%              EVALUATION              %
% ==================================== %
\section{Evaluation}\label{sec:evaluation}
\textbf{Strategy.} Every number below is out of fold on all 707 training images and is computed like the competition scorer: each annotator view is scored separately and pooled, with matches at IoU above 0.5. Dice and IoU ``per view'' compare the union of predicted instances with the union of one annotator's labels; this Dice equals \texttt{torchmetrics.segmentation.DiceScore} (a test in the repository checks it). Intervals are 95\% percentile intervals of a bootstrap over images (4,000 resamples).

\begin{table}[t]
  \caption{Out-of-fold results on 707 images and 1,154 views, with 95\% bootstrap intervals.}
  \label{tab:main}
  \centering\small
  \begin{tabular}{lcc}
    \toprule
    Metric & Value & 95\% interval \\
    \midrule
    Panoptic Quality & @@PQ@@ & @@PQCI@@ \\
    Segmentation quality (mean IoU of matches) & @@SQ@@ & @@SQCI@@ \\
    Recognition quality & @@RQ@@ & @@RQCI@@ \\
    Mean Dice per view & @@DICE@@ & @@DICECI@@ \\
    Mean IoU per view & @@IOU@@ & @@IOUCI@@ \\
    \midrule
    Median Dice per view & @@MEDDICE@@ & \\
    Dice pooled over all pixels & @@GDICE@@ & \\
    Pixel precision / recall & @@PREC@@ / @@REC@@ & \\
    TP / FP / FN & \multicolumn{2}{c}{@@TP@@ / @@FP@@ / @@FN@@} \\
    Public leaderboard PQ & 0.41 & \\
    \bottomrule
  \end{tabular}
\end{table}

\textbf{Results.} Table~\ref{tab:main} gives the main scores and Figure~\ref{fig:metrics} the distributions. Dice per view has quartiles @@Q25@@ and @@Q75@@; the @@LOWDICE@@ views below 0.1 are views with few or no labels in which the model predicted filaments, or the reverse. Matched pairs have a mean IoU of @@PAIRIOU@@ (mean Dice @@PAIRDICE@@). Table~\ref{tab:ablation} shows how the score was built up; every row was measured on the same images. The largest single step is the learned selection.

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig_metrics.pdf}
  \caption{Distributions over the 1,154 annotator views (Dice, IoU) and over the @@TP@@ matched pairs (IoU).}
  \label{fig:metrics}
\end{figure}

\begin{table}[t]
  \caption{Out-of-fold PQ of successive versions of the pipeline.}
  \label{tab:ablation}
  \centering\small
  \begin{tabular}{lc}
    \toprule
    Configuration & PQ \\
    \midrule
    One detector and refiner, detector confidence $\geq 0.35$ & 0.4290 \\
    + learned selection value & 0.4547 \\
    + second refiner averaged & 0.4583 \\
    + candidates of the 1536~px detector & 0.4596 \\
@@ABLATION@@    \bottomrule
  \end{tabular}
\end{table}

\textbf{One-to-many and many-to-one relations.} A prediction lies inside a label when more than half of its pixels do, and the reverse. Of the 8,199 labels, @@O2M@@ (@@O2MP@@\%) hold two or more predictions (fragmentation), and @@M2O@@ of the @@NPRED@@ prediction and view pairs (@@M2OP@@\%) hold two or more labels (merging). Splitting and merging are therefore rare. The dominant errors are labelled filaments without a prediction and predictions that no annotator of that view drew.

\begin{table}[t]
  \caption{Causes of the errors of stage~1 (out of fold, 2,213 false positives and 2,979 false negatives).}
  \label{tab:errors}
  \centering\small
  \begin{tabular}{lr}
    \toprule
    False positives & Share \\
    \midrule
    Fragment of a larger labelled filament & 32\% \\
    No label nearby in this view & 31\% \\
    Matches another annotator's label of the image & 18\% \\
    Covers a label but is much larger & 14\% \\
    IoU just under 0.5 & 5\% \\
    \midrule
    False negatives & Share \\
    \midrule
    Good candidate existed, dropped by the keep rule & 25\% \\
    Only a poor candidate existed & 21\% \\
    Partly covered by a kept prediction & 20\% \\
    Inside a larger kept prediction & 16\% \\
    No candidate & 13\% \\
    IoU just under 0.5, or split & 5\% \\
    \bottomrule
  \end{tabular}
\end{table}

\textbf{Public leaderboard.} The submitted file scores 0.41 on the public half of the test set, against @@PQ2@@ out of fold. We see the same gap for every version of the pipeline. Random 90-image subsets of the out-of-fold results have a standard deviation of 0.011 and never fall below 0.415 in 20,000 draws, and year, site and number of annotators are distributed alike in training and test data, so we cannot attribute the gap to sampling. We report it as an open observation.

\textbf{Efficiency.} @@SPEED@@

\textbf{Strengths.} (i) The keep rule optimises the competition metric directly and is trained on the disagreement between annotators instead of ignoring it. (ii) Instances are disjoint by construction, and splitting and merging are rare. (iii) Every component was accepted only after an out-of-fold gain on all 707 images, and the whole pipeline is one class (\texttt{FilamentPipeline}) that reproduces the submitted file from the weights in the repository.

\textbf{Weaknesses and possible improvements.} (i) Recognition, not outline quality, limits the score. Table~\ref{tab:errors} classifies the errors: for a quarter of the false negatives a good candidate existed and was dropped, and 18\% of the false positives match the label of \emph{another} annotator of the same image, so part of the error is annotator disagreement that no model can remove. A perfect keep rule on our candidates would reach about 0.53; closing that gap needs information about annotator behaviour that we could not extract from the image. (ii) Thresholds (selection 0.24, paste 0.65, stage~2 0.40) were chosen on the same out-of-fold predictions that are reported, each from a handful of values, so the reported PQ is slightly optimistic. (iii) The pipeline has several stages and @@NFILES@@ weight files; distilling it into one network would simplify use. (iv) Thin barbs are often cut, because the mean-annotator target rewards the filament body. A target that weights barbs, or labels with annotator identity, would address this. @@WEAK_EXTRA@@

% ==================================== %
%            ACKNOWLEDGMENT            %
% ==================================== %
\section{Acknowledgment}\label{sec:acknowledgment}
% Do not change
The organization of the Filament Segmentation Challenge 2026 is partially supported by the U.S. National Science Foundation (NSF) under Grant No. 2209912 and 2433781, directorate for Computer and Information Science and Engineering (CSE), and Office of Advanced Cyberinfrastructure (OAC), and Grant No. 2511630, AST Division Of Astronomical Sciences and MPS Directorate for Mathematical and Physical Sciences.

The Filament Segmentation Challenge 2026 is sponsored by the U.S. National Science Foundation (NSF) National Solar Observatory (NSO).

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a cooperative agreement with the National Science Foundation and with contribution from the National Oceanic and Atmospheric Administration. The GONG network of instruments is hosted by the Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrof\'isica de Canarias, and Cerro Tololo Interamerican Observatory.

\bibliographystyle{ACM-Reference-Format}
\bibliography{main,refs}

\end{document}
"""

V = {
    "v7": {
        "TITLE": "Propose, Refine, Select: Annotator-Aware Instance Segmentation of Solar Filaments",
        "ABSTRACT": r"Annotators of the MAGFiLO dataset agree only moderately on which structures are filaments (one annotator scored against another reaches a Panoptic Quality of 0.34), so we split the task into recall-oriented proposal, per-instance mask refinement, and a learned decision that keeps an instance when its expected matched IoU over annotators is high enough to raise Panoptic Quality. Two YOLO11s-seg detectors propose boxes, box-seeded U-Nets segment each box, a gradient-boosted model selects instances, and a second U-Net re-segments each kept instance in a rotation-normalised crop. On all 707 training images, predicted out of fold, the pipeline reaches a Panoptic Quality of @@PQ@@ (95\% interval @@PQCI@@) and a mean Dice of @@DICE@@; it scores 0.41 on the public leaderboard and runs in about @@SPI@@~s per image on one GPU.",
        "INTRO_END": r"The resulting pipeline reaches an out-of-fold PQ of @@PQ@@ and 0.41 on the public leaderboard.",
        "SELECT": r"Instances with a value at or below $\tau=0.24$ are dropped.",
        "STAGE2": r"\textbf{Stage 2: re-segmentation in context.} Each kept instance is cut out along its principal axis into a 384$\times$128 crop, so that every filament is horizontal and fills the crop. A U-Net with a ResNet34 encoder receives five channels: gray, CLAHE, the probability map of a dense full-image U-Net (ResNet34, tiled inference), the instance's own stage-1 mask, and the masks of the other kept instances. It is trained with binary cross-entropy plus Dice loss on a soft target: the mean, over the annotator views that contain this filament (IoU above 0.2 with the instance), of the best-overlapping label. The target is thus the share of annotators who marked each pixel. Five fold models and four flips are averaged and thresholded at 0.40.",
        "PRETRAINED": "",
        "ABLATION": "    + stage 2 (this submission) & 0.4626 \\\\\n",
        "WEAK_EXTRA": r"(v) A heavier stage~2 helps: replacing the U-Net by fine-tuned SAM~2.1 models and adding their mask-quality score to the keep rule raises the out-of-fold PQ by @@DIFF@@ (95\% interval @@DIFFCI@@, paired bootstrap) at about nine times the run time. That variant is documented in the repository notes; this submission is the faster configuration.",
        "NFILES": "twelve",
    },
    "v11": {
        "TITLE": "Propose, Refine, Select: Annotator-Aware Instance Segmentation of Solar Filaments with Fine-Tuned SAM 2.1",
        "ABSTRACT": r"Annotators of the MAGFiLO dataset agree only moderately on which structures are filaments (one annotator scored against another reaches a Panoptic Quality of 0.34), so we split the task into recall-oriented proposal, per-instance mask refinement, and a learned decision that keeps an instance when its expected matched IoU over annotators is high enough to raise Panoptic Quality. Two YOLO11s-seg detectors propose boxes, box-seeded U-Nets segment each box, fully fine-tuned SAM~2.1 models re-segment each instance from a box prompt, and instances are selected by the mean of a gradient-boosted value and the fine-tuned mask-quality score of SAM~2.1. On all 707 training images, predicted out of fold, the pipeline reaches a Panoptic Quality of @@PQ@@ (95\% interval @@PQCI@@) and a mean Dice of @@DICE@@; it scores 0.41 on the public leaderboard.",
        "INTRO_END": r"The resulting pipeline reaches an out-of-fold PQ of @@PQ@@ and 0.41 on the public leaderboard.",
        "SELECT": r"Instances with a value $v$ at or below 0.12 are dropped at this point; the final decision follows stage~2.",
        "STAGE2": r"\textbf{Stage 2: re-segmentation with SAM 2.1.} Each remaining instance is cut out along its principal axis into a 384$\times$128 crop, so that every filament is horizontal and fills the crop. The crop has five channels: gray, CLAHE, the probability map of a dense full-image U-Net (ResNet34, tiled inference), the instance's own stage-1 mask, and the masks of the other instances. SAM~2.1 \cite{sam2} is used as a box-prompted refiner: the first three channels, resized to 1024$\times$1024, form its input image and the bounding box of the stage-1 mask is the prompt. All weights (image encoder, prompt encoder, mask decoder) are fine-tuned for eight epochs with binary cross-entropy plus $0.5\,\times$ Dice loss on a soft target: the mean, over the annotator views that contain this filament (IoU above 0.2 with the instance), of the best-overlapping label, that is, the share of annotators who marked each pixel. Base-plus and large models, five folds each, are averaged over four flips and thresholded at 0.40. Out of fold this stage adds 0.011 PQ over stage~1, against 0.003 for a ResNet34 U-Net trained on the same crops and targets." "\n\n" r"\textbf{Final selection.} In five further SAM~2.1 base-plus models the IoU-prediction head is trained, together with the mask loss, to regress the matched IoU of every candidate (mean over views, zero if unmatched), in the spirit of Mask Scoring R-CNN \cite{msrcnn}. This score $s$ sees the image at higher resolution than the tabular features of the selection value $v$. An instance is kept when $(v+s)/2>0.24$.",
        "PRETRAINED": ", the public SAM~2.1 checkpoints",
        "ABLATION": "    + stage 2 with a ResNet34 U-Net & 0.4626 \\\\\n    stage 2 with SAM 2.1 base-plus and large instead & 0.4704 \\\\\n    + SAM 2.1 quality score in the keep rule (this submission) & 0.4725 \\\\\n",
        "WEAK_EXTRA": r"(v) Cost: fifteen SAM~2.1 models make inference about nine times slower than the same pipeline with a ResNet34 U-Net in stage~2, for a gain of @@DIFF@@ PQ (95\% interval @@DIFFCI@@, paired bootstrap). Distilling the SAM ensemble into one model is the obvious next step.",
        "NFILES": "22",
    },
}


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(c) -> str:
    return f"{c[0]:.3f} to {c[1]:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v7", "v11"])
    ap.add_argument("--author", default="Omkar Kabde")
    ap.add_argument("--country", default="India")
    ap.add_argument("--email", default="omkarkabde@gmail.com")
    ap.add_argument("--no_compile", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    v = a.version
    n = json.loads((ROOT / "results" / "report_figures" / v / "numbers.json").read_text())
    p, c, s = n["point"], n["ci95"], n["selected"]
    ver = ROOT / "results" / "verify_release" / v / "predict" / "summary.json"
    rep = json.loads((ver if ver.is_file() else ROOT / "results" / "predict" / f"{v}_repro" / "summary.json").read_text())
    vs = ROOT / "results" / "verify_release" / v / "summary.json"
    gpu = json.loads(vs.read_text()).get("gpu", "") if vs.is_file() else ""
    cmp_ = rep["compare"]
    spi = rep["seconds_per_image"]
    speed = (f"On one GPU{' (' + gpu + ')' if gpu else ''} the full pipeline takes {spi:.1f}~s per 2048$\\times$2048 image "
             f"({rep['seconds'] / 60:.0f} minutes for the 180 test images, model loading included). "
             f"Run from the repository weights in a fresh environment, it reproduces the submitted file: {cmp_['matched_iou_gt_0.5']:,} of "
             f"{cmp_['reference_rows']:,} submitted instances are matched, with a mean IoU of {cmp_['mean_iou_of_matched']:.4f}"
             + (f" ({cmp_['identical_masks']:,} masks are identical; the rest differ by a few boundary pixels because of GPU arithmetic)." if cmp_["identical_masks"] else "."))
    extra = rep["rows"] - cmp_["matched_iou_gt_0.5"]
    if extra > 0:
        speed += f" The rerun holds {extra} additional instance{'s' if extra > 1 else ''} whose keep value lies at the threshold."
    if v == "v7":
        speed += " On a CPU it takes about 50~s per image."
    o2m = sum(x for k, x in n["gt_predictions_inside_counts"].items() if int(k) >= 2)
    m2o = sum(x for k, x in n["pred_labels_inside_counts"].items() if int(k) >= 2)
    npred = sum(n["pred_labels_inside_counts"].values())
    rub = json.loads((ROOT / "results" / "rubric_metrics" / f"dump_kf5b_fold0_x5_avg384_pool1536_{v}" / "summary.json").read_text())["selected"]
    d = n["v11_minus_v7"]
    rep_map = {
        "AUTHOR": a.author, "COUNTRY": a.country, "EMAIL": a.email, "EXAMPLE": n["example_image"],
        "PQ": f"{p['pq']:.4f}", "PQ2": f"{p['pq']:.3f}", "PQCI": ci(c["pq"]), "SQ": f3(p["sq"]), "SQCI": ci(c["sq"]),
        "RQ": f3(p["rq"]), "RQCI": ci(c["rq"]), "DICE": f3(p["mean_dice"]), "DICECI": ci(c["mean_dice"]),
        "IOU": f3(p["mean_iou"]), "IOUCI": ci(c["mean_iou"]), "MEDDICE": f3(s["median_dice_per_view"]),
        "GDICE": f3(s["global_dice"]), "PREC": f3(s["pixel_precision"]), "REC": f3(s["pixel_recall"]),
        "TP": f"{s['tp']:,}", "FP": f"{s['fp']:,}", "FN": f"{s['fn']:,}",
        "Q25": f3(n["dice_quantiles"]["0.25"]), "Q75": f3(n["dice_quantiles"]["0.75"]),
        "LOWDICE": str(rub["dice_per_view_hist"]["0.00-0.10"]),
        "PAIRIOU": f3(s["matched_pair_iou_mean"]), "PAIRDICE": f3(s["matched_pair_dice_mean"]),
        "O2M": f"{o2m:,}", "O2MP": f"{100 * o2m / 8199:.1f}", "M2O": f"{m2o:,}", "NPRED": f"{npred:,}",
        "M2OP": f"{100 * m2o / npred:.1f}", "SPEED": speed, "SPI": f"{spi:.1f}",
        "DIFF": f"{d['pq']:.3f}", "DIFFCI": f"{d['pq_ci95'][0]:.3f} to {d['pq_ci95'][1]:.3f}",
    }
    tex = HEAD
    for _ in range(2):  # version blocks contain number placeholders
        for k, x in {**V[v], **rep_map}.items():
            tex = tex.replace(f"@@{k}@@", x)
    left = re.findall(r"@@\w+@@", tex)
    assert not left, left
    out = ROOT / "report" / v
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.tex").write_text(tex)
    (out / "refs.bib").write_text(REFS)
    (out / "preamble.tex").write_text(PREAMBLE)
    (out / "main.bib").write_text(MAINBIB)
    summary = {"version": v, "compiled": False, "numbers": rep_map | {"SPEED": speed}}
    if not a.no_compile:
        r = subprocess.run(["tectonic", "--keep-logs", "main.tex"], cwd=out, capture_output=True, text=True)
        (ROOT / "results" / "build_report" / v).mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "build_report" / v / "tectonic.log").write_text(r.stdout + r.stderr)
        summary["compiled"] = r.returncode == 0
        if r.returncode == 0:
            shutil.move(out / "main.pdf", out / "report.pdf")
            info = subprocess.run(["pdfinfo", str(out / "report.pdf")], capture_output=True, text=True).stdout
            summary["pages"] = int(re.search(r"Pages:\s+(\d+)", info).group(1))
            txt = subprocess.run(["pdftotext", "-layout", str(out / "report.pdf"), "-"], capture_output=True, text=True).stdout
            pages = txt.split("\f")
            summary["acknowledgment_page"] = next((i + 1 for i, t in enumerate(pages) if "ACKNOWLEDGMENT" in t.upper()), None)
            summary["warnings"] = [l for l in (r.stdout + r.stderr).splitlines() if "Overfull" in l or "undefined" in l.lower()][:20]
        else:
            summary["error_tail"] = (r.stdout + r.stderr)[-2500:]
        for junk in ["main.log", "main.blg", "main.aux", "main.bbl", "main.out"]:
            (out / junk).unlink(missing_ok=True)
    res = ROOT / "results" / "build_report" / v
    res.mkdir(parents=True, exist_ok=True)
    summary["seconds"] = round(time.time() - t0, 1)
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: x for k, x in summary.items() if k != "numbers"}, indent=2))


if __name__ == "__main__":
    main()
