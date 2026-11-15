"""Check the solar rotation mapping on training pairs taken 0.5 to 3 days apart.

For each pair (A, B) the labels of A are moved to the time of B with rotation +1, 0
(no motion) and -1 (wrong way). The share of B's labelled pixels covered by the moved
labels of A (dilated 5 px) tells which mapping is right.

Usage: python scripts/temporal_check.py [--pairs=60]
Writes results/temporal_check/{summary.json,pairs.csv}.
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solar_filament.temporal import obs_days, rotated_polygon_mask  # noqa: E402

DATA = Path(os.environ.get("SOLAR_DATA", ROOT / "MAGFiLO_1.0_Kaggle_2026/train"))
RESULTS = Path(os.environ.get("SOLAR_RESULTS", ROOT / "results"))
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=int, default=60); a = ap.parse_args()
t0 = time.time()
coco = json.loads((DATA / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
anns = defaultdict(list)
for x in coco["annotations"]:
    anns[x["image_id"]].append(x)
first_view = {}
for im in coco["images"]:
    first_view.setdefault(im["file_name"], im["id"])
files = sorted(first_view, key=obs_days)
days = np.array([obs_days(f) for f in files])
rng = np.random.default_rng(0)
K = np.ones((11, 11), np.uint8)
rows = []
for i in rng.permutation(len(files) - 1):
    j = i + 1
    dt = days[j] - days[i]
    if not 0.5 <= dt <= 3 or not anns[first_view[files[i]]] or not anns[first_view[files[j]]]:
        continue
    gb = np.zeros((2048, 2048), bool)
    for x in anns[first_view[files[j]]]:
        gb |= rotated_polygon_mask(x["segmentation"], 0.0)
    row = {"a": files[i], "b": files[j], "dt_days": round(float(dt), 3), "b_pixels": int(gb.sum())}
    for name, direction in (("forward", 1.0), ("static", 0.0), ("backward", -1.0)):
        ga = np.zeros((2048, 2048), bool)
        for x in anns[first_view[files[i]]]:
            ga |= rotated_polygon_mask(x["segmentation"], dt, direction=direction)
        ga = cv2.dilate(ga.astype(np.uint8), K).astype(bool)
        row[name] = round(float((ga & gb).sum() / max(1, gb.sum())), 4)
    rows.append(row)
    if len(rows) >= a.pairs:
        break
out = RESULTS / "temporal_check"; out.mkdir(parents=True, exist_ok=True)
with (out / "pairs.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary = {"pairs": len(rows), "mean_dt_days": float(np.mean([r["dt_days"] for r in rows])),
           **{f"coverage_{k}": float(np.mean([r[k] for r in rows])) for k in ("forward", "static", "backward")},
           "seconds": round(time.time() - t0, 1)}
(out / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
