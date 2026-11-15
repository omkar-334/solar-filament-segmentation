"""Run the final pipeline on a directory of images and write a submission CSV.

Usage: python scripts/predict.py --images=<dir> --weights=<dir> --out=submission.csv
       [--compare=<reference submission.csv>] [--tag=v7_repro] [--limit=N]
Writes the CSV and results/predict/<tag>/{summary.json,per_image.csv}. With --compare,
each image's instances are matched to the reference (IoU > 0.5) and the agreement is logged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_filament.pipeline import FilamentPipeline  # noqa: E402
from solar_filament.rle import audit_submission, decode_mask, encode_mask  # noqa: E402

RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))


def agreement(labels: np.ndarray, ref_rles: list[str]) -> dict:
    n = int(labels.max())
    area = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    ious = []
    for rle in ref_rles:
        g = decode_mask(rle, *labels.shape)
        inter = np.bincount(labels[g], minlength=n + 1).astype(np.float64)
        iou = inter / (area + g.sum() - inter + 1e-9)
        iou[0] = 0
        ious.append(float(iou.max()) if n else 0.0)
    ious = np.array(ious)
    return {"n_ref": len(ref_rles), "matched": int((ious > 0.5).sum()),
            "mean_iou_matched": float(ious[ious > 0.5].mean()) if (ious > 0.5).any() else 0.0,
            "identical": int((ious > 0.999).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.environ.get("SOLAR_TEST", "MAGFiLO_1.0_Kaggle_2026/test/test_images"))
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--compare", default="")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--version", default="auto", choices=["auto", "v7", "v11"],
                    help="auto: v11 when the SAM weight files are in --weights, otherwise v7")
    a = ap.parse_args()
    t0 = time.time()
    paths = sorted(Path(a.images).glob("*.jp*g"))
    if a.limit:
        paths = paths[: a.limit]
    ref = defaultdict(list)
    if a.compare:
        csv.field_size_limit(10 ** 9)
        with open(a.compare) as fh:
            for r in csv.DictReader(fh):
                ref[r["filament_id"].rsplit("_", 1)[0]].append(r["segmentation_rle"])
    pipe = FilamentPipeline(a.weights, version=a.version)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_image = []
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filament_id", "segmentation_rle"])
        for i, p in enumerate(paths):
            t1 = time.time()
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            labels, st = pipe.predict(gray, return_stages=True)
            n = int(labels.max())
            for k in range(1, n + 1):
                w.writerow([f"{p.stem}_{k}", encode_mask(labels == k)])
            row = {"image": p.stem, "instances": n, "stage1_instances": int(st["stage1"].max()),
                   "candidates_1024": st["candidates"][0], "candidates_1536": st["candidates"][1],
                   "seconds": round(time.time() - t1, 2)}
            if a.compare:
                row.update(agreement(labels, ref[p.stem]))
            per_image.append(row)
            if i % 20 == 0:
                print(i, row, flush=True)
    shapes = {p.stem: (2048, 2048) for p in paths}
    rows = audit_submission(out, shapes, expected_image_ids=shapes.keys())
    res = RESULTS / "predict" / a.tag
    res.mkdir(parents=True, exist_ok=True)
    with (res / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0])); w.writeheader(); w.writerows(per_image)
    summary = {"args": vars(a), "version": pipe.version, "device": str(pipe.device), "images": len(paths), "rows": int(rows),
               "seconds": round(time.time() - t0, 1),
               "seconds_per_image": round(float(np.mean([r["seconds"] for r in per_image])), 2)}
    if a.compare:
        n_ref = sum(r["n_ref"] for r in per_image); m = sum(r["matched"] for r in per_image)
        summary["compare"] = {"reference_rows": n_ref, "matched_iou_gt_0.5": m,
                              "identical_masks": sum(r["identical"] for r in per_image),
                              "mean_iou_of_matched": float(np.sum([r["mean_iou_matched"] * r["matched"] for r in per_image]) / max(1, m)),
                              "images_with_same_count": sum(r["instances"] == r["n_ref"] for r in per_image)}
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
