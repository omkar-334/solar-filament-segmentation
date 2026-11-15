"""The Dice in scripts/rubric_metrics.py must equal the organisers' torchmetrics DiceScore.

The rubric names torchmetrics.segmentation.DiceScore. For a binary foreground mask scored
per sample (one image and annotator view), the micro-averaged foreground Dice is
2 |P and G| / (|P| + |G|), which is what ``semantic_dice`` computes from pixel counts.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torchmetrics.segmentation import DiceScore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rubric_metrics import semantic_dice  # noqa: E402


def test_semantic_dice_matches_torchmetrics():
    rng = np.random.default_rng(0)
    pred = rng.random((6, 64, 64)) > 0.6
    gt = rng.random((6, 64, 64)) > 0.7
    ours = [semantic_dice(float(p.sum()), float(g.sum()), float((p & g).sum())) for p, g in zip(pred, gt)]
    metric = DiceScore(num_classes=2, include_background=False, average="micro", input_format="index")
    theirs = metric(torch.from_numpy(pred).long(), torch.from_numpy(gt).long()).item()  # mean over samples
    assert abs(float(np.mean(ours)) - theirs) < 1e-6
    for p, g, o in zip(pred, gt, ours):
        m = DiceScore(num_classes=2, include_background=False, average="micro", input_format="index")
        assert abs(m(torch.from_numpy(p[None]).long(), torch.from_numpy(g[None]).long()).item() - o) < 1e-6
