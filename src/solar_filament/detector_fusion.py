from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage


def detector_gated_instances(
    foreground_probability: np.ndarray,
    detections: Iterable[tuple[np.ndarray, float]],
    foreground_threshold: float = 0.5,
    detection_threshold: float = 0.3,
    grow_radius: int = 4,
    min_area: int = 200,
    close_radius: int = 1,
    fill_holes: bool = True,
    valid_region: np.ndarray | None = None,
) -> np.ndarray:
    """Use detector masks to assign U-Net foreground pixels to instances.

    Detection masks propose instance extent and rank contested pixels. The
    foreground network supplies the pixel detail. Proposals claim pixels in
    descending confidence order, so output instances are disjoint.
    """
    probability = np.asarray(foreground_probability, dtype=np.float32)
    foreground = probability >= foreground_threshold
    if close_radius > 0:
        structure = _disk(close_radius)
        foreground = ndimage.binary_closing(foreground, structure=structure)
    if fill_holes:
        foreground = ndimage.binary_fill_holes(foreground)
    if valid_region is not None:
        region = np.asarray(valid_region, dtype=bool)
        if region.shape != foreground.shape:
            raise ValueError(
                f"Valid region shape {region.shape} != foreground shape {foreground.shape}"
            )
        foreground &= region
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.uint16)

    proposals = []
    grow = _disk(grow_radius) if grow_radius > 0 else None
    for mask, score in detections:
        if float(score) < detection_threshold:
            continue
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape != foreground.shape:
            raise ValueError(
                f"Detection mask shape {candidate.shape} != foreground shape {foreground.shape}"
            )
        if grow is not None:
            candidate = ndimage.binary_dilation(candidate, structure=grow)
        candidate &= foreground
        if candidate.any():
            proposals.append((float(score), candidate))

    proposals.sort(key=lambda row: row[0], reverse=True)
    claimed = np.zeros(foreground.shape, dtype=bool)
    instances: list[np.ndarray] = []
    for _, candidate in proposals:
        instance = candidate & ~claimed
        area = int(instance.sum())
        if area < min_area:
            continue
        instances.append(instance)
        claimed |= instance

    labels = np.zeros(foreground.shape, dtype=np.uint16)
    for instance_id, instance in enumerate(instances, start=1):
        labels[instance] = instance_id
    return labels


def solar_disk_mask(gray_image: np.ndarray, erosion_radius: int = 0) -> np.ndarray:
    """Estimate the solar disk as the largest bright connected region.

    This follows the solar-disk constraint used in the public two-stage
    reference pipeline. Erosion can keep uncertain limb pixels out of
    predictions; use radius 0 to retain the detected disk boundary.
    """
    return solar_disk_masks(gray_image, (erosion_radius,))[erosion_radius]


def solar_disk_masks(
    gray_image: np.ndarray,
    erosion_radii: Iterable[int],
) -> dict[int, np.ndarray]:
    """Compute several inward disk margins from one disk estimate."""
    image = np.asarray(gray_image)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}")
    radii = {max(0, int(radius)) for radius in erosion_radii}
    if not radii:
        return {}
    bright = image > np.percentile(image, 60)
    labels, count = ndimage.label(bright)
    if count == 0:
        full_disk = np.ones(image.shape, dtype=bool)
        return {radius: full_disk for radius in radii}
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    disk = labels == int(np.argmax(areas))
    disk = ndimage.binary_closing(disk, structure=np.ones((25, 25), dtype=bool))
    disk = ndimage.binary_fill_holes(disk)
    distances = ndimage.distance_transform_edt(disk)
    return {
        radius: (disk if radius == 0 else distances > radius).astype(bool, copy=False)
        for radius in radii
    }


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def export_yolo_segmentation_dataset(
    annotation_json: str | Path,
    image_root: str | Path,
    records: list[dict],
    train_records: list[dict],
    val_records: list[dict],
    output_root: str | Path,
) -> Path:
    """Export each annotator view as one YOLO segmentation training example."""
    import json
    import os
    import shutil

    root = Path(output_root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(annotation_json).read_text())
    anns_by_image: dict[str, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    records_by_id = {record["image_id"]: record for record in records}

    for split, split_records in (("train", train_records), ("val", val_records)):
        for record in split_records:
            image_id = record["image_id"]
            source = Path(image_root) / record["file_name"]
            image_name = f"{image_id}.jpg"
            image_path = root / "images" / split / image_name
            label_path = root / "labels" / split / f"{image_id}.txt"
            if not image_path.exists():
                try:
                    os.symlink(source, image_path)
                except OSError:
                    shutil.copy2(source, image_path)
            record_meta = records_by_id[image_id]
            width, height = int(record_meta["width"]), int(record_meta["height"])
            lines = []
            for ann in anns_by_image.get(image_id, []):
                segmentation = ann.get("segmentation")
                if not isinstance(segmentation, list) or not segmentation:
                    continue
                polygons = (
                    [segmentation]
                    if isinstance(segmentation[0], (int, float))
                    else segmentation
                )
                # MAGFiLO currently has one polygon per filament. If a future
                # record has multiple polygons, retain the largest contour.
                polygon = max(polygons, key=lambda points: len(points), default=[])
                if len(polygon) < 6:
                    continue
                normalized = []
                for index in range(0, len(polygon) - 1, 2):
                    x = min(max(float(polygon[index]) / width, 0.0), 1.0)
                    y = min(max(float(polygon[index + 1]) / height, 0.0), 1.0)
                    normalized.extend((x, y))
                if len(normalized) >= 6:
                    lines.append("0 " + " ".join(f"{value:.6f}" for value in normalized))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: filament\n"
    )
    return yaml_path

