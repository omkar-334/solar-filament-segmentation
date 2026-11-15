"""Stage 2: rotation-normalised, context-conditioned re-segmentation of selected instances.

Each selected instance is cut out along its own long axis into a 384 x 128 crop with five
channels (gray, CLAHE, dense U-Net probability, its own stage-1 mask, the other selected
instances). A ResNet34 U-Net predicts the probability that an annotator who drew this
filament marked each pixel. The thresholded map replaces the stage-1 mask.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

W, H = 384, 128


def rect_affine(mask_win: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """2x3 affine from image coordinates to the 384 x 128 crop, long axis horizontal."""
    ys, xs = np.nonzero(mask_win)
    pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T) + np.eye(2) * 1e-6
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    if abs(v[0]) < abs(v[1]) and v[1] < 0 or abs(v[0]) >= abs(v[1]) and v[0] < 0:
        v = -v
    u = np.array([-v[1], v[0]])
    with np.errstate(all="ignore"):  # some BLAS builds raise spurious matmul warnings here
        a, b = (pts - c) @ v, (pts - c) @ u
    c = c + v * (a.max() + a.min()) / 2 + u * (b.max() + b.min()) / 2
    ext_long = max((a.max() - a.min() + 1) * 1.5, 64.0) + 16
    ext_short = max((b.max() - b.min() + 1) * 2.0 + 24, ext_long / 3)
    ext_long = max(ext_long, ext_short)
    sx, sy = W / ext_long, H / ext_short
    return np.array([[v[0] * sx, v[1] * sx, W / 2 - sx * (c @ v)],
                     [u[0] * sy, u[1] * sy, H / 2 - sy * (c @ u)]], dtype=np.float64)


def warp(img: np.ndarray, M: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def window_prob(prob: np.ndarray, M: np.ndarray, shape) -> tuple[int, int, int, int, np.ndarray]:
    """Warp a crop probability map back to image space; returns the window and its map."""
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
    corners = (inv @ np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], dtype=np.float64).T).T[:, :2]
    x0 = int(max(0, np.floor(corners[:, 0].min()))); y0 = int(max(0, np.floor(corners[:, 1].min())))
    x1 = int(min(shape[1], np.ceil(corners[:, 0].max()) + 1)); y1 = int(min(shape[0], np.ceil(corners[:, 1].max()) + 1))
    Mw = M.copy()
    Mw[:, 2] += M[:, :2] @ np.array([x0, y0], dtype=np.float64)
    p = cv2.warpAffine(prob.astype(np.float32), Mw, (x1 - x0, y1 - y0),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    return x0, y0, x1, y1, p


def build_model(pretrained: bool):
    """ResNet34 U-Net with a five-channel stem (gray, CLAHE, dense map, seed, other instances)."""
    import os
    import torch
    from .crop_refiner import CropUNet

    enc = os.environ.get("STAGE2_ENCODER", "")
    if enc:  # experiment only: any segmentation_models_pytorch encoder
        return _SmpNet(enc, pretrained)
    model = CropUNet(pretrained=pretrained, encoder_name="resnet34")
    w = model.stem[0].weight.data
    conv = torch.nn.Conv2d(5, 64, 7, 2, 3, bias=False)
    conv.weight.data = w.mean(dim=1, keepdim=True).repeat(1, 5, 1, 1) * 3 / 5
    model.stem[0] = conv
    model.mean = torch.tensor([0.45, 0.45, 0.2, 0.1, 0.1]).view(1, 5, 1, 1)
    model.std = torch.tensor([0.25, 0.25, 0.3, 0.3, 0.3]).view(1, 5, 1, 1)
    return model


class _SmpNet:
    """Factory: U-Net with a segmentation_models_pytorch encoder and the same five-channel input."""

    def __new__(cls, encoder: str, pretrained: bool):
        import segmentation_models_pytorch as smp
        import torch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=5, classes=1)
                self.register_buffer("mean", torch.tensor([0.45, 0.45, 0.2, 0.1, 0.1]).view(1, 5, 1, 1))
                self.register_buffer("std", torch.tensor([0.25, 0.25, 0.3, 0.3, 0.3]).view(1, 5, 1, 1))

            def forward(self, x):
                return self.net((x - self.mean) / self.std)

        return Net()


def instance_crops(labels: np.ndarray, gray: np.ndarray, unet: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(label index, affine, uint8 crop [5, H, W]) for every instance of a label map."""
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    if unet.shape != gray.shape:
        unet = cv2.resize(unet, gray.shape[::-1], interpolation=cv2.INTER_LINEAR)
    sel_map = (labels > 0).astype(np.uint8) * 255
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels)):
        if sl is None:
            continue
        m_win = labels[sl] == i + 1
        M = rect_affine(m_win, sl[1].start, sl[0].start)
        seed = np.zeros(gray.shape, np.uint8)
        seed[sl][m_win] = 255
        others = sel_map.copy()
        others[sl][m_win] = 0
        out.append((i, M, np.stack([warp(c, M) for c in (gray, clahe, unet, seed, others)])))
    return out


def predict_crops(models: list, crops: np.ndarray, device, batch: int = 96) -> np.ndarray:
    """Mean probability of the models over four flips; crops are uint8 [N, 5, H, W]."""
    import torch

    probs = np.zeros((len(crops), H, W), np.float32)
    amp = device.type == "cuda"
    for model in models:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            for b in range(0, len(crops), batch):
                x = torch.from_numpy(crops[b:b + batch]).to(device).float() / 255
                p = torch.sigmoid(model(x).float())
                p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
                p = p + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
                p = p + torch.sigmoid(model(x.flip(-1, -2)).float()).flip(-1, -2)
                probs[b:b + batch] += (p / 4)[:, 0].cpu().numpy() / len(models)
    return probs


def paste_refined(labels: np.ndarray, order_value: np.ndarray, items, threshold: float = 0.40,
                  min_area: int = 96) -> tuple[np.ndarray, list[str]]:
    """Replace stage-1 masks by thresholded stage-2 maps, claiming pixels in ``order_value``
    order (highest first). A stage-1 mask is kept when fewer than ``min_area`` unclaimed
    pixels remain. ``items`` are (label index, affine, probability crop). Label ids are kept."""
    slices = ndimage.find_objects(labels)
    wins = {idx: window_prob(prob, M, labels.shape) for idx, M, prob in items}
    new = np.zeros(labels.shape, np.uint16)
    used = [""] * len(order_value)
    for i in np.argsort(-np.asarray(order_value), kind="stable"):
        sl = slices[i] if i < len(slices) else None
        if sl is None:
            continue
        used[i] = "stage1"
        if i in wins:
            x0, y0, x1, y1, p = wins[i]
            m = (p >= threshold) & (new[y0:y1, x0:x1] == 0)
            if m.sum() >= min_area:
                new[y0:y1, x0:x1][m] = i + 1
                used[i] = "stage2"
        if used[i] == "stage1":
            m = (labels[sl] == i + 1) & (new[sl] == 0)
            new[sl][m] = i + 1
    return new, used
