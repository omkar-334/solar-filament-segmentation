"""Figures and numbers of the technical report for one pipeline version.

  fig_pipeline.pdf   diagram of the data pipeline
  fig_examples.jpg   predicted instances on one test image (from the submitted CSV) with zooms
  fig_metrics.pdf    distributions of Dice and IoU per annotator view and of matched-pair IoU
  numbers.json       out-of-fold metrics with 95% bootstrap intervals (images resampled),
                     and the paired difference between v11 and v7

Usage: python scripts/report_figures.py --version=v7|v11 [--image=<test image stem>]
Writes report/<version>/figures/* and results/report_figures/<version>/{numbers.json,bootstrap.csv}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solar_filament.rle import decode_mask  # noqa: E402

RUBRIC = {v: ROOT / "results" / "rubric_metrics" / f"dump_kf5b_fold0_x5_avg384_pool1536_{v}" for v in ("v7", "v11")}
SUBMISSION = {"v7": "submission_rescored_v7_stage2.csv", "v11": "submission_rescored_v11_sam2.csv"}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42})


def per_image(v: str) -> dict:
    """Per image: pooled counts over its annotator views and the per-view Dice and IoU."""
    d: dict = {}
    with (RUBRIC[v] / "per_view.csv").open() as fh:
        for r in csv.DictReader(fh):
            e = d.setdefault(r["file"], {"tp": 0, "fp": 0, "fn": 0, "s": 0.0, "dice": [], "iou": []})
            e["tp"] += int(r["tp"]); e["fp"] += int(r["fp"]); e["fn"] += int(r["fn"]); e["s"] += float(r["sum_iou"])
            e["dice"].append(float(r["dice"])); e["iou"].append(float(r["iou"]))
    return d


def stats(tp, fp, fn, s, dice_sum, iou_sum, views):
    return {"pq": s / (tp + 0.5 * fp + 0.5 * fn), "sq": s / tp, "rq": tp / (tp + 0.5 * fp + 0.5 * fn),
            "mean_dice": dice_sum / views, "mean_iou": iou_sum / views}


def bootstrap(v: str, n: int = 4000, seed: int = 0):
    data = {u: per_image(u) for u in ("v7", "v11")}
    files = sorted(data["v7"])
    assert files == sorted(data["v11"])
    A = {u: np.array([[data[u][f]["tp"], data[u][f]["fp"], data[u][f]["fn"], data[u][f]["s"], sum(data[u][f]["dice"]),
                       sum(data[u][f]["iou"]), len(data[u][f]["dice"])] for f in files], float) for u in data}
    point = {u: stats(*A[u].sum(axis=0)) for u in A}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n):
        idx = rng.integers(0, len(files), len(files))
        s7, s11 = stats(*A["v7"][idx].sum(axis=0)), stats(*A["v11"][idx].sum(axis=0))
        rows.append({"b": b, **{f"v7_{k}": x for k, x in s7.items()}, **{f"v11_{k}": x for k, x in s11.items()},
                     "pq_diff": s11["pq"] - s7["pq"], "dice_diff": s11["mean_dice"] - s7["mean_dice"]})
    ci = {k: [float(np.percentile([r[k] for r in rows], q)) for q in (2.5, 97.5)] for k in rows[0] if k != "b"}
    return point, ci, rows, len(files)


def fig_pipeline(v: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 25); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#333333", lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.3, linespacing=1.15)

    def arrow(x0, y0, x1, y1, **kw):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="#333333", **kw))

    blue, green, orange, grey = "#dbe7f5", "#dff0df", "#fbe6cf", "#eeeeee"
    box(0.5, 9, 8.5, 8, "H-alpha\nimage\n2048 x 2048", grey)
    box(12, 14.5, 12.5, 8, "YOLO11s-seg\n1024 px\nconf >= 0.05", blue)
    box(12, 3.5, 12.5, 8, "YOLO11s-seg\n1536 px\nconf >= 0.05", blue)
    box(27.5, 14.5, 13.5, 8, "seeded U-Nets\nR18/256, R34/384\n4 flips", blue)
    box(27.5, 3.5, 13.5, 8, "seeded U-Net\nR18/256\n4 flips", blue)
    box(44, 9, 11, 8, "paste, pixel\nclaiming\np >= 0.65", green)
    arrow(9.3, 14, 11.7, 18); arrow(9.3, 12, 11.7, 8)
    arrow(24.8, 18.5, 27.2, 18.5); arrow(24.8, 7.5, 27.2, 7.5)
    arrow(41.3, 18, 43.7, 14.5); arrow(41.3, 8, 43.7, 11.5)
    arrow(55.3, 13, 57.7, 13)
    if v == "v7":
        box(58, 9, 12, 8, "LightGBM:\nexpected IoU\n> 0.24", orange)
        box(73, 9, 14, 8, "stage 2:\nR34 U-Net x 5\nrotated crop", blue)
        box(90, 9, 9.5, 8, "instances\n(RLE rows)", grey)
        arrow(70.3, 13, 72.7, 13); arrow(87.3, 13, 89.7, 13)
    else:
        box(58, 9, 9.5, 8, "LightGBM\nvalue v\n> 0.12", orange)
        box(70.5, 9, 11, 8, "stage 2:\nSAM 2.1 B+, L\n5 folds each", blue)
        box(84.5, 9, 9.5, 8, "SAM 2.1 B+\nscore s; keep\n(v+s)/2 > 0.24", orange)
        box(96.3, 9, 3.4, 8, "RLE", grey)
        arrow(67.8, 13, 70.2, 13); arrow(81.8, 13, 84.2, 13); arrow(94.3, 13, 96, 13)
    ax.text(50, 23.6, "dense U-Net probability map (R34, tiled) is a context channel of every stage-2 crop",
            ha="center", va="center", fontsize=5.8, style="italic", color="#444444")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_examples(v: str, out: Path, stem: str) -> str:
    csv.field_size_limit(10 ** 9)
    masks: dict = {}
    with (ROOT / SUBMISSION[v]).open() as fh:
        for r in csv.DictReader(fh):
            masks.setdefault(r["filament_id"].rsplit("_", 1)[0], []).append(r["segmentation_rle"])
    if not stem:
        stem = max(masks, key=lambda s: len(masks[s]))
    path = next((ROOT / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images").glob(stem + ".*"))
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    ms = [decode_mask(r, 2048, 2048) for r in masks[stem]]
    rng = np.random.default_rng(5)
    cols = rng.uniform(0.25, 1.0, size=(len(ms), 3))
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255
    for m, c in zip(ms, cols):
        rgb[m] = 0.4 * rgb[m] + 0.6 * c
    order = np.argsort([-m.sum() for m in ms])
    picks, centres = [], []
    for i in order:  # three zoom windows around large filaments that are far apart
        ys, xs = np.nonzero(ms[i]); c = (int(ys.mean()), int(xs.mean()))
        if all(abs(c[0] - o[0]) + abs(c[1] - o[1]) > 420 for o in centres):
            picks.append(i); centres.append(c)
        if len(picks) == 3:
            break
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.85), gridspec_kw={"wspace": 0.03})
    ax[0].imshow(rgb); ax[0].set_title(f"{stem}: {len(ms)} instances", fontsize=6.5, pad=2)
    half = 190
    for k, (cy, cx) in enumerate(centres):
        y0, x0 = int(np.clip(cy - half, 0, 2048 - 2 * half)), int(np.clip(cx - half, 0, 2048 - 2 * half))
        ax[0].add_patch(plt.Rectangle((x0, y0), 2 * half, 2 * half, fill=False, ec="white", lw=0.5))
        ax[0].text(x0 + 2 * half + 12, y0 + 60, "abc"[k], color="white", fontsize=6)
        ax[k + 1].imshow(gray[y0:y0 + 2 * half, x0:x0 + 2 * half], cmap="gray", vmin=0, vmax=255)
        for m, c in zip(ms, cols):
            w = m[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if w.any():
                ax[k + 1].contour(w, levels=[0.5], colors=[tuple(c)], linewidths=0.7)
        ax[k + 1].set_title(f"({'abc'[k]}) zoom, outlines", fontsize=6.5, pad=2)
    for a in ax:
        a.axis("off")
    fig.savefig(out, dpi=330, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return stem


def fig_metrics(v: str, out: Path) -> None:
    sel = json.loads((RUBRIC[v] / "summary.json").read_text())["selected"]
    fig, ax = plt.subplots(1, 3, figsize=(3.35, 1.45), gridspec_kw={"wspace": 0.42})
    for a, key, title, col in [(ax[0], "dice_per_view_hist", "Dice per view", "#4c72b0"),
                               (ax[1], "iou_per_view_hist", "IoU per view", "#55a868"),
                               (ax[2], "matched_pair_iou_hist", "matched-pair IoU", "#c44e52")]:
        h = sel[key]
        a.bar(np.arange(len(h)) / 10 + 0.05, list(h.values()), width=0.09, color=col)
        a.set_title(title, fontsize=6.5, pad=2); a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1])
        a.tick_params(labelsize=5.5, length=2, pad=1)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("count", fontsize=6, labelpad=1)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v7", "v11"])
    ap.add_argument("--image", default="")
    a = ap.parse_args()
    t0 = time.time()
    v = a.version
    figs = ROOT / "report" / v / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results" / "report_figures" / v
    res.mkdir(parents=True, exist_ok=True)
    point, ci, rows, n_img = bootstrap(v)
    with (res / "bootstrap.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sel = json.loads((RUBRIC[v] / "summary.json").read_text())["selected"]
    fig_pipeline(v, figs / "fig_pipeline.pdf")
    stem = fig_examples(v, figs / "fig_examples.jpg", a.image)
    fig_metrics(v, figs / "fig_metrics.pdf")
    numbers = {"version": v, "images": n_img, "views": sel["views"], "bootstrap_resamples": len(rows),
               "point": point[v], "ci95": {k[len(v) + 1:]: x for k, x in ci.items() if k.startswith(v + "_")},
               "v11_minus_v7": {"pq": point["v11"]["pq"] - point["v7"]["pq"], "pq_ci95": ci["pq_diff"],
                                "mean_dice": point["v11"]["mean_dice"] - point["v7"]["mean_dice"], "mean_dice_ci95": ci["dice_diff"],
                                "share_of_resamples_with_v11_ahead": float(np.mean([r["pq_diff"] > 0 for r in rows]))},
               "selected": {k: x for k, x in sel.items() if not isinstance(x, (dict, list))},
               "dice_quantiles": sel["dice_per_view_quantiles"],
               "gt_predictions_inside_counts": sel["gt_predictions_inside_counts"],
               "pred_labels_inside_counts": sel["pred_labels_inside_counts"],
               "example_image": stem, "seconds": round(time.time() - t0, 1)}
    (res / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
