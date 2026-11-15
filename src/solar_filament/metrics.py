from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def connected_components(
    probability: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 64,
    merge_distance: int = 0,
) -> np.ndarray:
    """Threshold probabilities into disjoint instances.

    ``merge_distance`` groups foreground components whose Chebyshev gap is at
    most approximately this many pixels. The returned labels still cover only
    thresholded foreground pixels; dilation is used only to decide grouping.
    """
    foreground = np.asarray(probability >= threshold, dtype=bool)
    structure = ndimage.generate_binary_structure(
        2, 2
    )  # 8-connectivity preserves diagonal threads.
    if merge_distance > 0 and foreground.any():
        # Two fragments separated by <= merge_distance are joined by expanding
        # each side by half the gap. Keep the original mask geometry afterward.
        radius = max(1, int(np.ceil(merge_distance / 2)))
        expanded = ndimage.binary_dilation(
            foreground, structure=structure, iterations=radius
        )
        groups, _ = ndimage.label(expanded, structure=structure)
        labels = groups * foreground
        _, inverse = np.unique(labels[foreground], return_inverse=True)
        labels[foreground] = inverse + 1
        count = int(labels.max())
    else:
        labels, count = ndimage.label(foreground, structure=structure)
    if count == 0:
        return np.zeros(foreground.shape, dtype=np.uint16)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    remap = np.zeros(count + 1, dtype=np.uint16)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return remap[labels]


def filter_components(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Drop instances below ``min_area`` and compact the remaining IDs."""
    labels = np.asarray(labels)
    if min_area <= 1 or not np.any(labels):
        return labels
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    remap = np.zeros(len(areas), dtype=np.uint16)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return remap[labels]


def pq_counts(
    pred_labels: np.ndarray, gt_masks: list[np.ndarray], iou_threshold: float = 0.5
) -> dict[str, float | int]:
    pred = np.asarray(pred_labels, dtype=np.int32)
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]
    n_pred, n_gt = len(pred_ids), len(gt_masks)
    pred_areas_all = np.bincount(pred.ravel())
    pred_areas = np.array([pred_areas_all[i] for i in pred_ids], dtype=np.float64)
    ious = np.zeros((n_pred, n_gt), dtype=np.float64)
    pred_id_to_col = {int(pid): i for i, pid in enumerate(pred_ids)}

    for j, gt in enumerate(gt_masks):
        gt = np.asarray(gt, dtype=bool)
        area_gt = int(gt.sum())
        if area_gt == 0 or n_pred == 0:
            continue
        intersections_by_label = np.bincount(pred[gt], minlength=int(pred.max()) + 1)
        for pid, i in pred_id_to_col.items():
            inter = int(intersections_by_label[pid])
            if inter:
                union = pred_areas[i] + area_gt - inter
                ious[i, j] = inter / union

    if n_pred and n_gt:
        rows, cols = linear_sum_assignment(ious, maximize=True)
        matched = [(r, c) for r, c in zip(rows, cols) if ious[r, c] > iou_threshold]
    else:
        matched = []
    tp = len(matched)
    fp, fn = n_pred - tp, n_gt - tp
    sum_iou = float(sum(ious[r, c] for r, c in matched))
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
        "pq": sum_iou / denom if denom else 0.0,
    }


def aggregate_pq(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    sum_iou = sum(float(r["sum_iou"]) for r in rows)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
        "pq_ratio_of_sums": sum_iou / denom if denom else 0.0,
        "pq_mean_per_view": float(np.mean([float(r["pq"]) for r in rows]))
        if rows
        else 0.0,
    }
