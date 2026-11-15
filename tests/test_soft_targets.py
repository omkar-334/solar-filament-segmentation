"""Soft-consensus refiner targets: values in [0,1], fractional where annotators disagree."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solar_filament.crop_refiner import SeedCropDataset, build_oof_proposal_crop_samples  # noqa: E402

DATA = ROOT / "MAGFiLO_1.0_Kaggle_2026/train"


def test_soft_targets_multi_view_image():
    coco = json.loads((DATA / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    counts = defaultdict(int)
    for im in coco["images"]:
        counts[im["file_name"]] += 1
    fname = sorted(f for f, c in counts.items() if c == 3)[0]
    records = [{"file_name": fname}]
    samples = build_oof_proposal_crop_samples(coco, DATA / "train_images", records, {fname: []},
                                              soft_consensus=True)
    pos = [s for s in samples if s.get("soft_polygon_sets")]
    assert pos and all(s["n_views"] == 3 for s in pos)
    ds = SeedCropDataset(pos)
    fractional = 0
    for i in range(len(pos)):
        _, t = ds[i]
        t = t.numpy()
        assert t.min() >= 0 and t.max() <= 1.0001
        fractional += int(((t > 0.01) & (t < 0.99)).sum() > 0)
    assert fractional > 0
