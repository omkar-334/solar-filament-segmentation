"""SAM 2.1 as stage-2 refiner and candidate scorer (used by the v11 pipeline).

The first three channels of a stage-2 crop (gray, CLAHE, dense map) are stretched to
1024 x 1024 as the image and the bounding box of the seed mask (channel 3) is the prompt.
The mask logits are returned in the crop frame; the IoU head, fine-tuned on the matched
IoU of the candidates, is the quality score.
"""

from __future__ import annotations

import numpy as np

H, W, S = 128, 384, 1024


def boxes_of(seed, jitter: float = 0.0):
    """Bounding boxes (x0, y0, x1, y1) in the 1024 frame of a batch of seed masks (B, H, W)."""
    import torch

    B = seed.shape[0]
    cols, rows = seed.any(1), seed.any(2)
    out = torch.zeros(B, 4, device=seed.device)
    for b in range(B):
        if not cols[b].any():
            out[b] = torch.tensor([0, 0, W, H], device=seed.device)
            continue
        cx, ry = torch.nonzero(cols[b])[:, 0], torch.nonzero(rows[b])[:, 0]
        out[b] = torch.stack([cx.min(), ry.min(), cx.max() + 1, ry.max() + 1]).float()
    if jitter:
        w, h = out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]
        out = out + (torch.rand_like(out) - 0.5) * 2 * jitter * torch.stack([w, h, w, h], 1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, W) * S / W
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, H) * S / H
    return out[:, None]


def forward(model, x5, jitter, mean, std, with_score=False):
    """Mask logits [B, 1, H, W] (and the IoU-head score [B]) for float crops x5 [B, 5, H, W] in 0..1."""
    import torch.nn.functional as F

    img = F.interpolate(x5[:, :3], size=(S, S), mode="bilinear", align_corners=False)
    img = (img - mean) / std
    out = model(pixel_values=img, input_boxes=boxes_of(x5[:, 3] > 0.5, jitter), multimask_output=False)
    m = out.pred_masks
    m = m.reshape(m.shape[0], -1, *m.shape[-2:])[:, :1].float()
    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    if with_score:
        return m, out.iou_scores.reshape(m.shape[0], -1)[:, 0].float()
    return m


def norm(device):
    import torch

    return (torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))


def load(path, device):
    """Model of a sam2_refiner.py checkpoint ({"model": half-precision state dict, "name": hub id}).
    Only the configuration is fetched from the hub; the weights come from the checkpoint."""
    import torch
    from transformers import Sam2Config, Sam2Model

    ck = torch.load(path, map_location="cpu")
    net = Sam2Model(Sam2Config.from_pretrained(ck["name"]))
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return net.to(device).eval()


def predict_masks(models, crops: np.ndarray, device, batch: int = 16) -> np.ndarray:
    """Mean probability over the models and four flips; crops are uint8 [N, 5, H, W]."""
    import torch

    probs = np.zeros((len(crops), H, W), np.float32)
    mean, std = norm(device)
    for net in models:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for b in range(0, len(crops), batch):
                x = torch.from_numpy(crops[b:b + batch]).to(device).float() / 255
                p = torch.sigmoid(forward(net, x, 0.0, mean, std))
                p = p + torch.sigmoid(forward(net, x.flip(-1), 0.0, mean, std)).flip(-1)
                p = p + torch.sigmoid(forward(net, x.flip(-2), 0.0, mean, std)).flip(-2)
                p = p + torch.sigmoid(forward(net, x.flip(-1, -2), 0.0, mean, std)).flip(-1, -2)
                probs[b:b + batch] += (p / 4)[:, 0].float().cpu().numpy() / len(models)
    return probs


def predict_scores(models, crops: np.ndarray, device, batch: int = 16) -> np.ndarray:
    """Mean IoU-head score over the models (no flips); crops are uint8 [N, 5, H, W]."""
    import torch

    sc = np.zeros(len(crops), np.float32)
    mean, std = norm(device)
    for net in models:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for b in range(0, len(crops), batch):
                x = torch.from_numpy(crops[b:b + batch]).to(device).float() / 255
                sc[b:b + batch] += forward(net, x, 0.0, mean, std, with_score=True)[1].cpu().numpy() / len(models)
    return sc
