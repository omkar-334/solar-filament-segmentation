"""Measure annotator-vs-annotator PQ on images with 2-3 independent annotator views.

For each physical image with k>=2 views, every ordered pair (A as prediction, B as
reference) is scored with the same PQ code used for model validation. Also scores
a "medoid" strategy: the view that agrees most with the others, scored against the
others. Writes per-pair JSONL and summary JSON to results/inter_annotator_ceiling/.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solar_filament.data import polygon_mask  # noqa: E402
from solar_filament.metrics import aggregate_pq, pq_counts  # noqa: E402

ANN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = ROOT / "results/inter_annotator_ceiling"


def as_labels(masks: list[np.ndarray], shape) -> np.ndarray:
    """Paint instances in order; later instances do not overwrite earlier ones."""
    labels = np.zeros(shape, dtype=np.uint16)
    for i, m in enumerate(masks, start=1):
        labels[m & (labels == 0)] = i
    return labels


def score_file(item):
    filename, views = item
    h, w = views[0]["height"], views[0]["width"]
    masks = [
        [polygon_mask(a.get("segmentation"), h, w).astype(bool) for a in v["anns"]]
        for v in views
    ]
    rows = []
    for i in range(len(views)):
        pred = as_labels(masks[i], (h, w))
        for j in range(len(views)):
            if i == j:
                continue
            c = pq_counts(pred, masks[j])
            rows.append({"file": filename, "pred_view": i, "ref_view": j,
                         "n_pred": len(masks[i]), "n_ref": len(masks[j]), **c})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    coco = json.loads(ANN.read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    by_file = defaultdict(list)
    for im in coco["images"]:
        by_file[im["file_name"]].append({**im, "anns": anns[im["id"]]})
    multi = sorted((f, v) for f, v in by_file.items() if len(v) >= 2)
    with Pool(8) as pool:
        all_rows = [r for rows in pool.imap_unordered(score_file, multi) for r in rows]
    with (OUT / "pairs.jsonl").open("w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")

    pairwise = aggregate_pq(all_rows)
    # Medoid strategy: per image pick the view whose mean PQ vs others is highest
    # (an oracle choice, so an optimistic bound for "predict one annotator's style").
    by_img = defaultdict(list)
    for r in all_rows:
        by_img[r["file"]].append(r)
    medoid_rows, worst_rows = [], []
    for f, rows in by_img.items():
        per_view = defaultdict(list)
        for r in rows:
            per_view[r["pred_view"]].append(r)
        means = {v: np.mean([x["pq"] for x in rs]) for v, rs in per_view.items()}
        medoid_rows += per_view[max(means, key=means.get)]
        worst_rows += per_view[min(means, key=means.get)]
    counts = defaultdict(int)
    for f, v in multi:
        counts[len(v)] += 1
    summary = {
        "images_with_multiple_views": len(multi),
        "views_per_image_histogram": dict(counts),
        "ordered_pairs": len(all_rows),
        "pairwise_pq": pairwise,
        "sq_pairwise": pairwise["sum_iou"] / max(1, pairwise["tp"]),
        "oracle_best_view_pq": aggregate_pq(medoid_rows),
        "oracle_worst_view_pq": aggregate_pq(worst_rows),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
