"""CPU post-processing on dumped detector proposals and refiner probability crops.

Shared by local validation experiments and final test inference so that the
selected settings are applied identically.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

CROP = 256
EIGHT = ndimage.generate_binary_structure(2, 2)


def paste_masks(
    rows: list[dict],
    probs: np.ndarray,
    shape: tuple[int, int],
    threshold: float = 0.65,
    min_area: int = 96,
    order: np.ndarray | None = None,
    keep: np.ndarray | None = None,
    largest_cc: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Claim pixels in ``order``; returns a label map and the proposal index per label."""
    h, w = shape
    labels = np.zeros(shape, dtype=np.uint16)
    owner: list[int] = []
    if order is None:
        order = np.argsort([-r["score"] for r in rows], kind="stable")
    for idx in order:
        if keep is not None and not keep[idx]:
            continue
        x0, y0, side = rows[idx]["crop"]
        p = probs[idx].astype(np.float32) / 255.0
        m = cv2.resize(p, (side, side), interpolation=cv2.INTER_LINEAR) >= threshold
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
        if xb <= xa or yb <= ya:
            continue
        cand = m[ya - y0 : yb - y0, xa - x0 : xb - x0] & (labels[ya:yb, xa:xb] == 0)
        if largest_cc and cand.any():
            lab, n = ndimage.label(cand, structure=EIGHT)
            if n > 1:
                cand = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        if int(cand.sum()) < min_area:
            continue
        labels[ya:yb, xa:xb][cand] = len(owner) + 1
        owner.append(int(idx))
    return labels, owner


def instance_features(
    labels: np.ndarray, owner: list[int], rows: list[dict], probs: np.ndarray,
    yolo: np.ndarray, gray: np.ndarray, aux: dict[str, list[dict]] | None = None,
    unet: np.ndarray | None = None,
) -> np.ndarray:
    """Per-instance features for the expected-matched-IoU model.

    ``aux`` maps auxiliary detector name -> its proposals for this image; for
    each (sorted by name) we add best box IoU, that box's score and the max
    IoU*score, giving 3 agreement features per auxiliary detector.
    """
    h, w = labels.shape
    n = len(owner)
    aux = aux or {}
    aux_boxes = {k: (np.array([r["bbox"] for r in v], dtype=np.float32).reshape(-1, 4),
                     np.array([r["score"] for r in v], dtype=np.float32)) for k, v in sorted(aux.items())}
    n_unet = len(UNET_FEATURES) if unet is not None else 0
    if n == 0:
        return np.zeros((0, len(FEATURES) + 3 * len(aux_boxes) + n_unet), dtype=np.float32)
    objs = ndimage.find_objects(labels)
    scores = np.array([rows[i]["score"] for i in owner])
    feats = []
    cy0, cx0, rad = h / 2, w / 2, 0.46 * min(h, w)
    for k, idx in enumerate(owner):
        sl = objs[k]
        m = labels[sl] == k + 1
        area = float(m.sum())
        x0, y0, side = rows[idx]["crop"]
        p = probs[idx].astype(np.float32) / 255.0
        yo = yolo[idx].astype(np.float32) / 255.0
        pin = p >= 0.5
        yin = yo >= 0.5
        inter = float((pin & yin).sum())
        uni = float((pin | yin).sum()) or 1.0
        ys, xs = np.nonzero(m)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        r = np.hypot(ys.mean() - cy0, xs.mean() - cx0) / rad
        bh, bw = m.shape
        # Elongation from second moments.
        cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))
        elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
        inside = gray[sl][m].astype(np.float32)
        ring_sl = tuple(slice(max(0, s.start - 8), s.stop + 8) for s in sl)
        ring = gray[ring_sl].astype(np.float32)
        bx0, by0, bx1, by1 = rows[idx]["bbox"]
        n_cc = ndimage.label(m, structure=EIGHT)[1]
        feats.append([
            rows[idx]["score"],
            float(np.sum(scores > rows[idx]["score"])),
            np.log1p(area),
            float(p[pin].mean()) if pin.any() else 0.0,
            float((p >= 0.5).mean()),
            float((p >= 0.8).sum() / max(1, pin.sum())),
            inter / uni,
            float(yin.mean()),
            r,
            elong,
            np.log1p(side),
            area / max(1.0, (bx1 - bx0) * (by1 - by0)),
            float(np.median(ring) - inside.mean()) / (float(ring.std()) + 1.0),
            float(inside.mean()) / (float(np.median(ring)) + 1.0),
            float(n_cc),
            float(n),
            area / max(1.0, float(pin.sum()) * (side / p.shape[0]) ** 2),
            *_aux_agreement(rows[idx]["bbox"], aux_boxes),
            *(_unet_agreement(labels, k + 1, sl, unet) if unet is not None else []),
        ])
    return np.asarray(feats, dtype=np.float32)


UNET_FEATURES = ["unet_mean", "unet_frac50", "unet_cc_iou", "unet_ring_mean"]

FEATURES = [
    "det_score", "det_rank", "log_area", "ref_mean_p", "ref_frac_fg", "ref_frac_conf",
    "yolo_ref_iou", "yolo_frac_fg", "disk_radius", "elongation", "log_side",
    "box_fill", "contrast", "rel_intensity", "n_components", "n_instances_img",
    "claimed_fraction",
]


def matched_iou_per_instance(labels: np.ndarray, n: int, gt_masks: list[np.ndarray]) -> np.ndarray:
    """IoU of each predicted instance with its matched GT (0 if IoU<=0.5).

    IoU>0.5 matches are unique, so this equals the Hungarian PQ contribution.
    """
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    pred_area = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    for g in gt_masks:
        ga = float(g.sum())
        if ga == 0:
            continue
        inter = np.bincount(labels[g], minlength=n + 1).astype(np.float64)
        iou = inter / (pred_area + ga - inter + 1e-9)
        iou[0] = 0
        j = int(np.argmax(iou))
        if iou[j] > 0.5:
            out[j - 1] = max(out[j - 1], iou[j])
    return out


def _aux_agreement(box, aux_boxes) -> list[float]:
    out: list[float] = []
    x0, y0, x1, y1 = box
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    for boxes, scores in aux_boxes.values():
        if len(boxes) == 0:
            out += [0.0, 0.0, 0.0]
            continue
        iw = np.clip(np.minimum(x1, boxes[:, 2]) - np.maximum(x0, boxes[:, 0]), 0, None)
        ih = np.clip(np.minimum(y1, boxes[:, 3]) - np.maximum(y0, boxes[:, 1]), 0, None)
        inter = iw * ih
        iou = inter / (area + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter + 1e-6)
        j = int(np.argmax(iou))
        out += [float(iou[j]), float(scores[j]), float(np.max(iou * scores))]
    return out


def rescored_labels(
    rows: list[dict], probs: np.ndarray, yolo: np.ndarray, gray: np.ndarray, model,
    tau: float, threshold: float = 0.65, min_area: int = 96, unet: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Final selection rule: paste all candidates in detector-score order, predict
    expected matched IoU per instance, drop instances at or below ``tau``.

    Dropped pixels are not re-claimed, which is exactly what validation measured.
    Returns compacted labels and the kept instances' predicted values.
    """
    labels, owner = paste_masks(rows, probs, gray.shape, threshold, min_area)
    if not owner:
        return labels, np.zeros(0)
    pred = model.predict(instance_features(labels, owner, rows, probs, yolo, gray, unet=unet))
    keep = pred > tau
    remap = np.zeros(len(owner) + 1, dtype=np.uint16)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint16)
    return remap[labels], pred[keep]


def largest_component_only(labels: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of every instance."""
    out = np.zeros_like(labels)
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        m = labels[sl] == i
        lab, n = ndimage.label(m, structure=EIGHT)
        if n > 1:
            m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        out[sl][m] = i
    return out


def separate_instances(labels: np.ndarray) -> np.ndarray:
    """Drop pixels that 8-touch a different instance, so no two instances touch."""
    lab = labels.astype(np.int64)
    mx = ndimage.maximum_filter(lab, size=3)
    lo = lab.copy()
    lo[lo == 0] = np.iinfo(np.int64).max
    mn = ndimage.minimum_filter(lo, size=3)
    out = labels.copy()
    out[(lab > 0) & ((mx != lab) | (mn != lab))] = 0
    return out


def scorer_robust(labels: np.ndarray, min_area: int = 96) -> np.ndarray:
    """One connected piece per instance and no touching instances, so a scorer that
    re-instances by connected components sees exactly our instances."""
    out = largest_component_only(separate_instances(largest_component_only(labels)))
    areas = np.bincount(out.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    remap = np.zeros(len(areas), dtype=np.uint16)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return remap[out]


def _unet_agreement(labels: np.ndarray, label: int, sl, unet: np.ndarray, pad: int = 24) -> list[float]:
    """Agreement of an instance with an independent U-Net probability map (uint8)."""
    h, w = labels.shape
    ys = slice(max(0, sl[0].start - pad), min(h, sl[0].stop + pad))
    xs = slice(max(0, sl[1].start - pad), min(w, sl[1].stop + pad))
    m = labels[ys, xs] == label
    p = unet[ys, xs].astype(np.float32) / 255.0
    inside = p[m]
    fg = p >= 0.5
    cc, _ = ndimage.label(fg, structure=EIGHT)
    touched = np.unique(cc[m & fg])
    touched = touched[touched != 0]
    comp = np.isin(cc, touched) if len(touched) else np.zeros_like(fg)
    inter = float((comp & m).sum())
    union = float((comp | m).sum()) or 1.0
    ring = ndimage.binary_dilation(m, iterations=6) & ~m
    return [float(inside.mean()), float((inside >= 0.5).mean()), inter / union,
            float(p[ring].mean()) if ring.any() else 0.0]
