"""Image-based candidate scorer: predicts a candidate instance's expected matched IoU.

Input per candidate (4 x 128 x 128, uint8): grayscale crop, CLAHE crop, the
refiner probability, and the instance's final (claimed) mask, all in the
proposal's square crop frame. Its out-of-fold prediction is fed to the LightGBM
selector as an extra feature.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from solar_filament.postproc import paste_masks

SIZE = 128
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def candidate_tensors(
    rows: list[dict], probs: np.ndarray, gray: np.ndarray, threshold: float = 0.65, min_area: int = 96,
) -> tuple[np.ndarray, list[int]]:
    """Paste candidates exactly like the selector, then cut a 4-channel crop per instance."""
    labels, owner = paste_masks(rows, probs, gray.shape, threshold, min_area)
    h, w = gray.shape
    out = np.zeros((len(owner), 4, SIZE, SIZE), dtype=np.uint8)
    for k, idx in enumerate(owner):
        x0, y0, side = rows[idx]["crop"]
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
        g = np.zeros((side, side), np.uint8)
        m = np.zeros((side, side), np.uint8)
        g[ya - y0 : yb - y0, xa - x0 : xb - x0] = gray[ya:yb, xa:xb]
        m[ya - y0 : yb - y0, xa - x0 : xb - x0] = (labels[ya:yb, xa:xb] == k + 1) * 255
        out[k, 0] = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out[k, 1] = cv2.resize(_CLAHE.apply(g), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out[k, 2] = cv2.resize(probs[idx], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out[k, 3] = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out, owner


class CandidateScorer(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = net.conv1.weight.data
        net.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        net.conv1.weight.data = torch.cat([w, w[:, :1]], dim=1) * 0.75
        net.fc = nn.Linear(512, 1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x.float() / 255.0 - 0.45) / 0.25)[:, 0]


def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x
