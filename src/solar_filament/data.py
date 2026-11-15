from __future__ import annotations

import os

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset, WeightedRandomSampler


def _polygons(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if isinstance(segmentation[0], (int, float)):
        return [segmentation]
    return [p for p in segmentation if isinstance(p, list) and len(p) >= 6]


def polygon_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Rasterize one COCO polygon annotation into a uint8 binary mask."""
    polys = _polygons(segmentation)
    if not polys:
        return np.zeros((height, width), dtype=np.uint8)
    raster = os.environ.get("SOLAR_RASTER", "coco")
    if raster != "coco":  # diagnostic only: other common polygon rasterisers
        pts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys]
        if raster == "cv2":
            import cv2
            out = np.zeros((height, width), np.uint8)
            cv2.fillPoly(out, [np.rint(q).astype(np.int32) for q in pts], 1)
            return out
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        for q in pts:
            draw.polygon([tuple(x) for x in q], outline=1 if raster == "pil_outline" else 0, fill=1)
        return np.array(img, dtype=np.uint8)
    rles = mask_utils.frPyObjects(polys, height, width)
    rle = mask_utils.merge(rles)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8, copy=False)


def polygon_centers(segmentation: Any) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for poly in _polygons(segmentation):
        xs = poly[0::2]
        ys = poly[1::2]
        if xs and ys:
            centers.append((
                round((min(xs) + max(xs)) / 2),
                round((min(ys) + max(ys)) / 2),
            ))
    return centers


def prepare_records(
    data_root: str | Path, cache_dir: str | Path, preserve_instances: bool = False
) -> list[dict[str, Any]]:
    """Cache union masks and, when requested, per-instance label maps."""
    data_root, cache_dir = Path(data_root), Path(cache_dir)
    annotation_path = (
        data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    image_root = data_root / "train" / "train_images"
    mask_root = cache_dir / "union_masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    instance_root = cache_dir / "instance_maps"
    if preserve_instances:
        instance_root.mkdir(parents=True, exist_ok=True)
    coco = json.loads(annotation_path.read_text())
    anns_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    records: list[dict[str, Any]] = []
    for image in coco["images"]:
        image_id = image["id"]
        filename = image["file_name"]
        image_path = image_root / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing training image: {image_path}")
        key = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:20]
        target_path = mask_root / f"{key}.png"
        anns = anns_by_image.get(image_id, [])
        if preserve_instances:
            instance_path = instance_root / f"{key}.png"
        if not target_path.exists() or (preserve_instances and not instance_path.exists()):
            union = np.zeros((image["height"], image["width"]), dtype=np.uint8)
            instance_map = np.zeros((image["height"], image["width"]), dtype=np.uint16)
            occupied = np.zeros_like(union, dtype=bool)
            for instance_id, ann in enumerate(anns, start=1):
                instance = polygon_mask(
                    ann.get("segmentation"), image["height"], image["width"]
                )
                union |= instance
                if preserve_instances:
                    overlap = instance.astype(bool) & occupied
                    instance_map[overlap] = np.uint16(65535)
                    available = instance.astype(bool) & ~occupied
                    instance_map[available] = np.uint16(instance_id)
                    occupied |= instance.astype(bool)
                    # Keep ambiguous pixels ignored even if a later polygon covers them.
                    instance_map[instance_map == 65535] = np.uint16(65535)
            if not target_path.exists():
                Image.fromarray(union * 255, mode="L").save(target_path, optimize=True)
            if preserve_instances and not instance_path.exists():
                Image.fromarray(instance_map).save(instance_path, optimize=True)
        centers = [c for ann in anns for c in polygon_centers(ann.get("segmentation"))]
        record = {
            "image_id": image_id,
            "file_name": filename,
            "image_path": str(image_path),
            "target_path": str(target_path),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "centers": centers,
        }
        if preserve_instances:
            record["instance_path"] = str(instance_path)
            record["spines"] = [ann.get("spine", []) for ann in anns]
        records.append(record)
    return records


def grouped_split(
    records: list[dict[str, Any]], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by physical filename so annotator views never leak between folds."""
    filenames = sorted({r["file_name"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(filenames)
    n_val = max(1, round(len(filenames) * val_fraction))
    val_files = set(filenames[:n_val])
    train = [r for r in records if r["file_name"] not in val_files]
    val = [r for r in records if r["file_name"] in val_files]
    if {r["file_name"] for r in train} & {r["file_name"] for r in val}:
        raise RuntimeError("Grouped split leaked a physical image across folds")
    return train, val


def view_balanced_sampler(
    records: list[dict[str, Any]], samples_per_image: int = 2
) -> WeightedRandomSampler:
    """Sample annotation views inversely to their count for each physical image."""
    view_counts = Counter(r["file_name"] for r in records)
    weights = [1.0 / view_counts[r["file_name"]] for r in records]
    num_samples = len(view_counts) * samples_per_image
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


class FilamentCropDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        patch_size: int = 768,
        include_instances: bool = False,
        spine_width: int = 7,
    ):
        self.records = records
        self.patch_size = patch_size
        self.include_instances = include_instances
        self.spine_width = spine_width

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("L")
        target = Image.open(record["target_path"]).convert("L")
        instance_map = (
            Image.open(record["instance_path"]) if self.include_instances else None
        )
        w, h = image.size
        p = min(self.patch_size, w, h)

        if record["centers"] and random.random() < 0.75:
            cx, cy = random.choice(record["centers"])
            jitter = p // 4
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)
            left = min(max(0, cx - p // 2), w - p)
            top = min(max(0, cy - p // 2), h - p)
        else:
            left = random.randint(0, max(0, w - p))
            top = random.randint(0, max(0, h - p))
        box = (left, top, left + p, top + p)
        image = image.crop(box)
        target = target.crop(box)
        if instance_map is not None:
            instance_map = instance_map.crop(box)

        x = np.asarray(image, dtype=np.float32).copy() / 255.0
        y = (np.asarray(target, dtype=np.uint8).copy() > 0).astype(np.float32)
        if not self.include_instances:
            return torch.from_numpy(x[None]), torch.from_numpy(y[None])

        ids = np.asarray(instance_map, dtype=np.uint16).copy().astype(np.int64)
        spine = Image.new("L", (p, p), 0)
        draw = ImageDraw.Draw(spine)
        for points in record.get("spines", []):
            if not isinstance(points, list) or len(points) < 2:
                continue
            coords = [
                (float(points[i]) - left, float(points[i + 1]) - top)
                for i in range(0, len(points) - 1, 2)
            ]
            if len(coords) >= 2:
                draw.line(coords, fill=255, width=self.spine_width, joint="curve")
            elif coords:
                cx, cy = coords[0]
                radius = max(1, self.spine_width // 2)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
        spine_target = (np.asarray(spine, dtype=np.uint8).copy() > 0).astype(np.float32)
        return (
            torch.from_numpy(x[None]),
            torch.from_numpy(y[None]),
            torch.from_numpy(spine_target[None]),
            torch.from_numpy(ids),
        )


def load_full_image(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        Image.open(record["image_path"]).convert("L"), dtype=np.uint8
    ).copy()


def gt_instances_for_image(
    annotation_json: str | Path, image_id: str
) -> list[np.ndarray]:
    """Decode instance masks only; class, box, and spine metadata are intentionally unused."""
    coco = json.loads(Path(annotation_json).read_text())
    image = next(img for img in coco["images"] if img["id"] == image_id)
    anns = [ann for ann in coco["annotations"] if ann["image_id"] == image_id]
    return [
        polygon_mask(ann.get("segmentation"), image["height"], image["width"]).astype(
            bool
        )
        for ann in anns
    ]
