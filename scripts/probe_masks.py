"""Build diagnostic variants of a submission: every mask grown or shrunk by a few pixels.

Used to probe whether test labels are drawn thicker or thinner than training labels.
Growing claims only pixels that no other instance of the image holds, so masks never overlap;
a shrunk mask is kept unchanged if fewer than 96 px would remain.

Usage: python probe_masks.py --csv=<submission.csv> --px=1 --out=<new.csv>   (px < 0 shrinks)
Writes the CSV and results/probe_masks/<out stem>.json.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_filament.rle import audit_submission, decode_mask, encode_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--px", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--images", default="MAGFiLO_1.0_Kaggle_2026/test/test_images")
a = ap.parse_args()
csv.field_size_limit(10 ** 9)
rows = defaultdict(list)
for r in csv.DictReader(open(a.csv)):
    rows[r["filament_id"].rsplit("_", 1)[0]].append((r["filament_id"], r["segmentation_rle"]))
k = np.ones((3, 3), np.uint8)
ratios, unchanged = [], 0
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["filament_id", "segmentation_rle"])
    for stem, items in rows.items():
        lab = np.zeros((2048, 2048), np.uint16)
        for i, (_, rle) in enumerate(items):
            lab[decode_mask(rle, 2048, 2048)] = i + 1
        new = lab.copy() if a.px > 0 else np.zeros_like(lab)
        for i, (fid, _) in enumerate(items):
            m = (lab == i + 1).astype(np.uint8)
            if a.px > 0:
                g = cv2.dilate(m, k, iterations=a.px).astype(bool) & (new == 0)
                new[g] = i + 1
            else:
                e = cv2.erode(m, k, iterations=-a.px).astype(bool)
                if e.sum() < 96:
                    e = m.astype(bool); unchanged += 1
                new[e] = i + 1
            ratios.append(float((new == i + 1).sum()) / max(1, int(m.sum())))
            w.writerow([fid, encode_mask(new == i + 1)])
shapes = {p.stem: (2048, 2048) for p in sorted(Path(a.images).glob("*.jp*g"))}
n = audit_submission(Path(a.out), shapes, expected_image_ids=shapes.keys())
s = {"args": vars(a), "rows": int(n), "median_area_ratio": float(np.median(ratios)), "kept_unchanged": unchanged}
Path("results/probe_masks").mkdir(parents=True, exist_ok=True)
(Path("results/probe_masks") / (Path(a.out).stem + ".json")).write_text(json.dumps(s, indent=2))
print(s)
