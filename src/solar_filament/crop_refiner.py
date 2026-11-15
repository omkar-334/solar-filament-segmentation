from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from solar_filament.data import polygon_mask


CROP_SIZE = 256


def set_crop_size(size: int) -> None:
    """Change the refiner crop resolution for this process (train and inference must match)."""
    global CROP_SIZE
    CROP_SIZE = int(size)
MIN_CROP_SIDE = 96
CONTEXT_SCALE = 1.5


class CropUNet(nn.Module):
    """ResNet18 U-Net that refines a box-seeded grayscale crop."""

    def __init__(self, pretrained: bool = True, encoder_name: str = "resnet18"):
        super().__init__()
        if encoder_name == "resnet34":
            encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = _ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = _ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = _ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = _ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        value = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        value = self.dec3(torch.cat([self.up3(value), x2], dim=1))
        value = self.dec2(torch.cat([self.up2(value), x1], dim=1))
        value = self.dec1(torch.cat([self.up1(value), x0], dim=1))
        return self.head(self.up0(value))


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SeedCropDataset(Dataset):
    """Dynamically render grayscale, CLAHE, box seed, and instance target crops."""

    def __init__(self, samples: list[dict[str, Any]], cache_size: int = 8):
        self.samples = samples
        self.cache_size = cache_size
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.samples)

    def _read_gray(self, path: str) -> np.ndarray:
        image = self._image_cache.get(path)
        if image is None:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            self._image_cache[path] = image
            if len(self._image_cache) > self.cache_size:
                self._image_cache.popitem(last=False)
        else:
            self._image_cache.move_to_end(path)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = self._read_gray(sample["image_path"])
        x0, y0, side = sample["crop"]
        h, w = image.shape
        crop = np.zeros((side, side), dtype=np.uint8)
        ax0, ay0 = max(0, x0), max(0, y0)
        ax1, ay1 = min(w, x0 + side), min(h, y0 + side)
        if ax1 > ax0 and ay1 > ay0:
            crop[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = image[ay0:ay1, ax0:ax1]

        bx0, by0, bx1, by1 = sample["seed"]
        seed = np.zeros((side, side), dtype=np.uint8)
        sx0, sy0 = max(ax0, bx0), max(ay0, by0)
        sx1, sy1 = min(ax1, bx1), min(ay1, by1)
        if sx1 > sx0 and sy1 > sy0:
            seed[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = 255

        scale = CROP_SIZE / float(side)
        gray = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        clahe = cv2.resize(
            self._clahe.apply(crop), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
        )
        seed = cv2.resize(seed, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
        target = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        for polygon in sample.get("polygons", []):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = (points[:, 0] - x0) * scale
            points[:, 1] = (points[:, 1] - y0) * scale
            points = np.rint(points).astype(np.int32)
            cv2.fillPoly(target, [points], 1)
        soft_sets = sample.get("soft_polygon_sets")
        if soft_sets:
            # Each set is one annotator's polygons for this filament; the target is
            # the fraction of annotators covering each pixel.
            acc = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
            for polygons in soft_sets:
                layer = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
                for polygon in polygons:
                    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    points[:, 0] = (points[:, 0] - x0) * scale
                    points[:, 1] = (points[:, 1] - y0) * scale
                    cv2.fillPoly(layer, [np.rint(points).astype(np.int32)], 1)
                acc += layer
            target = acc / float(sample.get("n_views", len(soft_sets)))

        # Solar structures have no preferred orientation. Share flip/rotation
        # augmentations between the image channels, seed, and target.
        turns = random.randrange(4)
        if turns:
            gray = np.rot90(gray, turns).copy()
            clahe = np.rot90(clahe, turns).copy()
            seed = np.rot90(seed, turns).copy()
            target = np.rot90(target, turns).copy()
        if random.random() < 0.5:
            gray, clahe, seed, target = (
                np.ascontiguousarray(array[:, ::-1])
                for array in (gray, clahe, seed, target)
            )

        inputs = np.stack(
            [gray.astype(np.float32) / 255, clahe.astype(np.float32) / 255, seed > 0]
        ).astype(np.float32)
        return torch.from_numpy(inputs), torch.from_numpy(np.ascontiguousarray(target).astype(np.float32))[None]


def _polygons(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list) or not segmentation:
        return []
    if isinstance(segmentation[0], (int, float)):
        return [segmentation]
    return [poly for poly in segmentation if isinstance(poly, list) and len(poly) >= 6]


def _box_sample(
    image_path: str,
    bbox: tuple[float, float, float, float],
    polygons: list[list[float]],
    *,
    negative: bool = False,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    width, height = max(1.0, x1 - x0), max(1.0, y1 - y0)
    side = max(MIN_CROP_SIDE, int(np.ceil(max(width, height) * CONTEXT_SCALE)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    crop_x, crop_y = int(round(cx - side / 2)), int(round(cy - side / 2))
    return {
        "image_path": image_path,
        "crop": (crop_x, crop_y, side),
        "seed": (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
        "polygons": [] if negative else polygons,
    }


def build_medoid_crop_samples(
    coco: dict[str, Any],
    image_root: str | Path,
    train_records: list[dict[str, Any]],
    seed: int = 42,
    positive_crops_per_instance: int = 2,
) -> list[dict[str, Any]]:
    """Build medoid-annotator crops with jittered positive and hard-negative seeds."""
    from solar_filament.data import grouped_split

    del grouped_split  # Keep this builder independent after split records are passed.
    rng = random.Random(seed)
    image_root = Path(image_root)
    train_files = {record["file_name"] for record in train_records}
    images_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anns_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in coco["images"]:
        if image["file_name"] in train_files:
            images_by_file[image["file_name"]].append(image)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)

    samples: list[dict[str, Any]] = []
    for filename in sorted(images_by_file):
        versions = images_by_file[filename]
        image_path = str(image_root / filename)
        height, width = int(versions[0]["height"]), int(versions[0]["width"])
        semantic_masks = []
        version_annotations = []
        for version in versions:
            annotations = anns_by_image.get(version["id"], [])
            semantic = np.zeros((height, width), dtype=bool)
            for annotation in annotations:
                semantic |= polygon_mask(
                    annotation.get("segmentation"), height, width
                ).astype(bool)
            semantic_masks.append(semantic)
            version_annotations.append(annotations)
        if len(semantic_masks) == 1:
            medoid = 0
        else:
            scores = []
            for i, mask_a in enumerate(semantic_masks):
                pair_scores = []
                for j, mask_b in enumerate(semantic_masks):
                    if i == j:
                        continue
                    union = np.logical_or(mask_a, mask_b).sum()
                    pair_scores.append(
                        float(np.logical_and(mask_a, mask_b).sum()) / union
                        if union
                        else 1.0
                    )
                scores.append(float(np.mean(pair_scores)) if pair_scores else 1.0)
            medoid = int(np.argmax(scores))

        annotations = version_annotations[medoid]
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        union_mask = semantic_masks[medoid].astype(np.uint8)
        if not annotations:
            continue
        n_positive = 0
        for annotation in annotations:
            polygons = _polygons(annotation.get("segmentation"))
            if not polygons:
                continue
            bx, by, bw, bh = annotation.get("bbox", [0, 0, 0, 0])
            if bw < 4 or bh < 4:
                continue
            for _ in range(positive_crops_per_instance):
                scale = rng.uniform(0.80, 1.30)
                jside = max(8.0, max(bw, bh) * scale)
                cx = bx + bw / 2 + rng.uniform(-0.20, 0.20) * jside
                cy = by + bh / 2 + rng.uniform(-0.20, 0.20) * jside
                jittered = (
                    cx - jside / 2,
                    cy - jside / 2,
                    cx + jside / 2,
                    cy + jside / 2,
                )
                samples.append(_box_sample(image_path, jittered, polygons))
                n_positive += 1

        n_negative = max(1, n_positive // 3)
        gt_far = cv2.dilate(union_mask, np.ones((41, 41), np.uint8))
        dark_y, dark_x = np.where((gray < np.percentile(gray, 30)) & (gt_far == 0))
        bright_y, bright_x = np.where((gray > np.percentile(gray, 60)) & (gt_far == 0))
        for i in range(n_negative):
            side = rng.randint(MIN_CROP_SIDE, 384)
            use_dark = i % 5 < 3
            xs, ys = (dark_x, dark_y) if use_dark else (bright_x, bright_y)
            if len(xs) == 0:
                xs, ys = bright_x, bright_y
            if len(xs) == 0:
                continue
            cx, cy = int(xs[rng.randrange(len(xs))]), int(ys[rng.randrange(len(ys))])
            x0, y0 = cx - side // 2, cy - side // 2
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(width, x0 + side), min(height, y0 + side)
            if union_mask[ya:yb, xa:xb].any():
                continue
            seed_size = side * rng.uniform(0.50, 0.95)
            inset = (side - seed_size) / 2
            seed_box = (
                x0 + inset,
                y0 + inset,
                x0 + side - inset,
                y0 + side - inset,
            )
            samples.append(_box_sample(image_path, seed_box, [], negative=True))

    rng.shuffle(samples)
    return samples


def build_oof_proposal_crop_samples(
    coco: dict[str, Any],
    image_root: str | Path,
    train_records: list[dict[str, Any]],
    proposals_by_file: dict[str, list[dict[str, Any]]],
    seed: int = 42,
    max_proposals_per_image: int = 40,
    max_negatives_per_image: int = 16,
    soft_consensus: bool = False,
) -> list[dict[str, Any]]:
    """Build refiner crops from out-of-fold detector boxes and medoid masks.

    With ``soft_consensus`` every positive crop also carries, per annotator
    view, the polygons that view drew for the same filament (instances whose
    mask IoU with the medoid instance exceeds 0.2). Views that did not draw it
    contribute nothing, so the target is P(an annotator marks this pixel).

    OOF proposals are matched to the most-consistent annotator view for that
    physical image. Boxes that cover multiple reference instances are treated
    as hard negatives because the single-mask refiner cannot safely split them.
    A ground-truth box crop is also retained for every medoid instance so the
    refiner continues to see clean positive examples.
    """
    from solar_filament.data import polygon_mask

    rng = random.Random(seed)
    image_root = Path(image_root)
    train_files = {record["file_name"] for record in train_records}
    images_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anns_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in coco["images"]:
        if image["file_name"] in train_files:
            images_by_file[image["file_name"]].append(image)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)

    samples: list[dict[str, Any]] = []
    for filename in sorted(train_files):
        versions = images_by_file.get(filename, [])
        if not versions:
            continue
        height, width = int(versions[0]["height"]), int(versions[0]["width"])
        image_path = str(image_root / filename)
        semantic_masks = []
        version_annotations = []
        for version in versions:
            annotations = anns_by_image.get(version["id"], [])
            semantic = np.zeros((height, width), dtype=bool)
            for annotation in annotations:
                semantic |= polygon_mask(
                    annotation.get("segmentation"), height, width
                ).astype(bool)
            semantic_masks.append(semantic)
            version_annotations.append(annotations)

        if len(semantic_masks) <= 1:
            medoid = 0
        else:
            view_scores = []
            for index, mask in enumerate(semantic_masks):
                pair_scores = []
                for other_index, other in enumerate(semantic_masks):
                    if index == other_index:
                        continue
                    union = np.logical_or(mask, other).sum()
                    pair_scores.append(
                        float(np.logical_and(mask, other).sum()) / union
                        if union
                        else 1.0
                    )
                view_scores.append(float(np.mean(pair_scores)) if pair_scores else 1.0)
            medoid = int(np.argmax(view_scores))

        annotations = version_annotations[medoid]
        gt_masks = [
            polygon_mask(ann.get("segmentation"), height, width).astype(bool)
            for ann in annotations
        ]
        positive_counts = [0] * len(annotations)
        soft_sets_by_gt: list[list[list[list[float]]]] = [
            [_polygons(ann.get("segmentation"))] for ann in annotations
        ]
        if soft_consensus and len(version_annotations) > 1:
            for view_index, view_anns in enumerate(version_annotations):
                if view_index == medoid:
                    continue
                other_masks = [
                    polygon_mask(a.get("segmentation"), height, width).astype(bool)
                    for a in view_anns
                ]
                for gi, gmask in enumerate(gt_masks):
                    garea = int(gmask.sum())
                    polys: list[list[float]] = []
                    for a, om in zip(view_anns, other_masks):
                        inter = int(np.logical_and(gmask, om).sum())
                        if inter and inter / (garea + int(om.sum()) - inter) > 0.2:
                            polys.extend(_polygons(a.get("segmentation")))
                    soft_sets_by_gt[gi].append(polys)
        n_views = len(version_annotations)

        def _with_soft(sample: dict[str, Any], gi: int) -> dict[str, Any]:
            if soft_consensus:
                sample["soft_polygon_sets"] = soft_sets_by_gt[gi]
                sample["n_views"] = n_views
            return sample
        # Keep one clean seed per medoid instance in addition to realistic OOF boxes.
        for gi, annotation in enumerate(annotations):
            polygons = _polygons(annotation.get("segmentation"))
            bx, by, bw, bh = annotation.get("bbox", [0, 0, 0, 0])
            if polygons and bw >= 4 and bh >= 4:
                samples.append(_with_soft(
                    _box_sample(
                        image_path,
                        (float(bx), float(by), float(bx + bw), float(by + bh)),
                        polygons,
                    ), gi)
                )

        proposals = sorted(
            proposals_by_file.get(filename, []),
            key=lambda row: float(row.get("score", 0.0)),
            reverse=True,
        )[:max_proposals_per_image]
        negative_samples = []
        for proposal in proposals:
            raw_box = proposal.get("bbox", [])
            if len(raw_box) != 4:
                continue
            x0, y0, x1, y1 = map(float, raw_box)
            ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            ix1, iy1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            box_area = float((ix1 - ix0) * (iy1 - iy0))
            coverages = []
            for mask in gt_masks:
                area = int(mask.sum())
                overlap = int(mask[iy0:iy1, ix0:ix1].sum())
                coverages.append(overlap / max(1, area))
            matched = [i for i, coverage in enumerate(coverages) if coverage >= 0.25]
            if len(matched) == 1 and coverages[matched[0]] >= 0.35:
                gt_index = matched[0]
                if positive_counts[gt_index] < 3:
                    polygons = _polygons(annotations[gt_index].get("segmentation"))
                    if polygons:
                        samples.append(_with_soft(_box_sample(image_path, (x0, y0, x1, y1), polygons), gt_index))
                        positive_counts[gt_index] += 1
                continue

            # A proposal that merges multiple GT instances is also a rejection
            # example; the refiner produces only one instance mask per proposal.
            negative_samples.append(
                _box_sample(
                    image_path,
                    (x0, y0, x1, y1),
                    [],
                    negative=True,
                )
            )

        if len(negative_samples) > max_negatives_per_image:
            negative_samples = negative_samples[:max_negatives_per_image]
        samples.extend(negative_samples)

    rng.shuffle(samples)
    return samples


def crop_refiner_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return 0.5 * bce + 0.5 * dice


def make_inference_crop(
    gray: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Create one [gray, CLAHE, box] crop and return its full-image bounds."""
    h, w = gray.shape
    bx0, by0, bx1, by1 = bbox
    box_side = max(1.0, bx1 - bx0, by1 - by0)
    side = max(MIN_CROP_SIDE, int(np.ceil(box_side * CONTEXT_SCALE)))
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    patch = np.zeros((side, side), dtype=np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if xb > xa and yb > ya:
        patch[ya - y0 : yb - y0, xa - x0 : xb - x0] = gray[ya:yb, xa:xb]
    seed = np.zeros((side, side), dtype=np.uint8)
    sx0, sy0 = max(xa, int(np.floor(bx0))), max(ya, int(np.floor(by0)))
    sx1, sy1 = min(xb, int(np.ceil(bx1))), min(yb, int(np.ceil(by1)))
    if sx1 > sx0 and sy1 > sy0:
        seed[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = 255
    image = cv2.resize(patch, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    contrast = cv2.resize(
        cv2.createCLAHE(2.0, (8, 8)).apply(patch),
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    seed = cv2.resize(seed, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)
    tensor = np.stack(
        [image.astype(np.float32) / 255, contrast.astype(np.float32) / 255, seed > 0]
    ).astype(np.float32)
    return tensor, (x0, y0, side)
