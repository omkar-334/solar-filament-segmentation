"""Candidate boxes from dense U-Net probability maps (a second, different candidate family).

For every image the probability map is thresholded at several levels; each connected
component (area >= 96 px at 2048 px) becomes one proposal: its box, the level it came
from, and a score (mean probability inside the component times --score_scale, so that
confident detector candidates still claim pixels first when the pools are merged).
Components whose box overlaps an already kept box by IoU > 0.8 are dropped.

Usage: python dense_proposals.py --maps=/data/runs/unet_prob_oof --out=/data/runs/dense_proposals_oof.json
Writes the proposals JSON and results/dense_proposals/<maps name>/{summary.json,per_image.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

RESULTS = Path(os.environ.get("SOLAR_RESULTS", Path(__file__).resolve().parents[1] / "results"))
LEVELS = [0.15, 0.30, 0.45, 0.60, 0.75]
A: dict = {}


def box_iou(b, B):
    x0 = np.maximum(b[0], B[:, 0]); y0 = np.maximum(b[1], B[:, 1])
    x1 = np.minimum(b[2], B[:, 2]); y1 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def one(path: str):
    p = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if p.shape != (2048, 2048):
        p = cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    rows = []
    for level in LEVELS:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((p >= level * 255).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 96:
                continue
            box = [float(x), float(y), float(x + w), float(y + h)]
            if rows and box_iou(box, np.array([r["bbox"] for r in rows])).max() > 0.8:
                continue
            mean = float(p[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].mean()) / 255
            rows.append({"bbox": box, "score": round(mean * A["scale"], 5), "level": level, "area": int(area)})
    rows.sort(key=lambda r: -r["score"])
    return Path(path).stem, rows[: A["cap"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=".jpeg", help="image extension used as key in proposal files")
    ap.add_argument("--score_scale", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    A.update(scale=a.score_scale, cap=a.cap)
    t0 = time.time()
    paths = sorted(str(p) for p in Path(a.maps).glob("*.png"))
    with Pool(a.workers) as pool:
        res = pool.map(one, paths)
    preds = {stem + a.ext: rows for stem, rows in res}
    Path(a.out).write_text(json.dumps({"maps": a.maps, "levels": LEVELS, "score_scale": a.score_scale,
                                       "predictions": preds}))
    out = RESULTS / "dense_proposals" / Path(a.maps).name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "n"] + [f"n_level_{l}" for l in LEVELS])
        for f, rows in preds.items():
            w.writerow([f, len(rows)] + [sum(r["level"] == l for r in rows) for l in LEVELS])
    n = [len(r) for r in preds.values()]
    summary = {"maps": a.maps, "out": a.out, "images": len(preds), "proposals": int(sum(n)),
               "per_image_mean": float(np.mean(n)), "per_image_max": int(max(n)), "levels": LEVELS,
               "score_scale": a.score_scale, "cap": a.cap, "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
