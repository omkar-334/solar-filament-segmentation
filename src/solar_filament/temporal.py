"""Temporal neighbours: map labels of a training image taken at another time onto the
current image by solar differential rotation.

Only training-split labels are used. The disk geometry is fixed in this dataset
(centre about (1023, 1024), radius about 900 px, north up, rotation left to right).
The tilt of the solar axis towards the observer (B0, at most 7.25 degrees) is ignored.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

CX, CY, R = 1023.1, 1024.1, 899.7


def obs_days(file_name: str) -> float:
    """Observation time in days from the 14-digit timestamp at the start of the file name."""
    return datetime.strptime(file_name[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp() / 86400.0


def rotate_points(xy: np.ndarray, dt_days: float, direction: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Move image points by ``dt_days`` of solar rotation. Returns new points and a mask of
    points that are on the visible disk before and after."""
    x = (xy[:, 0] - CX) / R
    y = (CY - xy[:, 1]) / R
    r2 = x * x + y * y
    ok = r2 < 0.995
    z = np.sqrt(np.clip(1 - r2, 0, None))
    lat = np.arcsin(np.clip(y, -1, 1))
    lon = np.arctan2(x, z)
    s2 = np.sin(lat) ** 2
    omega = 14.713 - 2.396 * s2 - 1.787 * s2 * s2 - 0.9856  # synodic, degrees per day
    lon2 = lon + direction * np.deg2rad(omega * dt_days)
    ok &= np.abs(lon2) < np.deg2rad(88)
    x2 = np.cos(lat) * np.sin(lon2)
    return np.stack([x2 * R + CX, xy[:, 1]], axis=1), ok


def rotated_polygon_mask(segmentation, dt_days: float, shape=(2048, 2048), direction: float = 1.0) -> np.ndarray:
    """Rasterise a COCO polygon after rotating its vertices; empty if it leaves the disk."""
    out = np.zeros(shape, np.uint8)
    for poly in segmentation if isinstance(segmentation, list) else []:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        new, ok = rotate_points(pts, dt_days, direction)
        if ok.mean() < 0.8:
            continue
        cv2.fillPoly(out, [np.round(new[ok]).astype(np.int32)], 1)
    return out.astype(bool)
