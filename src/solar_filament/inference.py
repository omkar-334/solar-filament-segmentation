from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .metrics import connected_components
from .model import build_model
from .rle import encode_mask


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    count = int(np.ceil((length - overlap) / stride))
    return np.linspace(0, length - tile, count).round().astype(int).tolist()


@torch.inference_mode()
def predict_probability(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 1024,
    overlap: int = 256,
    batch_size: int = 4,
) -> np.ndarray:
    return predict_fields(model, image, device, tile_size, overlap, batch_size)[0]


@torch.inference_mode()
def predict_fields(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 1024,
    overlap: int = 256,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Predict blended foreground/spine probabilities and embedding fields."""
    h, w = image.shape
    tile = min(tile_size, h, w)
    xs, ys = tile_starts(w, tile, overlap), tile_starts(h, tile, overlap)
    locations = [(x, y) for y in ys for x in xs]
    img = torch.from_numpy(image.astype(np.float32) / 255.0).to(device=device)
    output_channels = int(getattr(model, "output_channels", 1))
    fields = torch.zeros((output_channels, h, w), device=device, dtype=torch.float32)
    weights = torch.zeros((h, w), device=device, dtype=torch.float32)
    win = torch.hann_window(tile, periodic=False, device=device).clamp_min(0.05)
    weight = torch.outer(win, win)

    for start in range(0, len(locations), batch_size):
        chunk = locations[start : start + batch_size]
        batch = torch.stack([
            img[y : y + tile, x : x + tile] for x, y in chunk
        ]).unsqueeze(1)
        batch = batch.contiguous(memory_format=torch.channels_last)
        use_amp = device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(batch)
        prediction = logits.float()
        if output_channels >= 2:
            prediction = torch.cat(
                (torch.sigmoid(prediction[:, :2]), prediction[:, 2:]), dim=1
            )
        else:
            prediction = torch.sigmoid(prediction)
        for k, (x, y) in enumerate(chunk):
            fields[:, y : y + tile, x : x + tile] += prediction[k] * weight[None]
            weights[y : y + tile, x : x + tile] += weight
    blended = (fields / weights.clamp_min(1e-6)[None]).cpu().numpy()
    mask_probability = blended[0]
    if output_channels < 10:
        return mask_probability, None, None
    return mask_probability, blended[1], blended[2:10]


def load_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        architecture=checkpoint.get("architecture", "small_unet"),
        base_channels=int(checkpoint.get("base_channels", 24)),
        pretrained_encoder=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, memory_format=torch.channels_last).eval()
    return model, checkpoint


def predict_instances(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    min_area: int = 64,
    tile_size: int = 1024,
    overlap: int = 256,
    batch_size: int = 4,
    merge_distance: int = 0,
    spine_threshold: float = 0.35,
    min_seed_area: int = 8,
    embedding_merge_distance: float = 0.8,
    max_seed_gap: float = 96.0,
    max_assignment_distance: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    prob, spine_prob, embeddings = predict_fields(
        model, image, device, tile_size, overlap, batch_size
    )
    if spine_prob is not None and embeddings is not None:
        labels = spine_seeded_instances(
            prob,
            spine_prob,
            embeddings,
            threshold=threshold,
            min_area=min_area,
            spine_threshold=spine_threshold,
            min_seed_area=min_seed_area,
            embedding_merge_distance=embedding_merge_distance,
            max_seed_gap=max_seed_gap,
            max_assignment_distance=max_assignment_distance,
        )
    else:
        labels = connected_components(
            prob,
            threshold=threshold,
            min_area=min_area,
            merge_distance=merge_distance,
        )
    return prob, labels


def spine_seeded_instances(
    foreground_probability: np.ndarray,
    spine_probability: np.ndarray,
    embeddings: np.ndarray,
    threshold: float = 0.4,
    min_area: int = 64,
    spine_threshold: float = 0.35,
    min_seed_area: int = 8,
    embedding_merge_distance: float = 0.8,
    max_seed_gap: float = 96.0,
    max_assignment_distance: float = 2.5,
) -> np.ndarray:
    """Use predicted spine components as instance seeds and embeddings to group pixels."""
    foreground = np.asarray(foreground_probability >= threshold, dtype=bool)
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.uint16)
    spine = np.asarray(spine_probability >= spine_threshold, dtype=bool)
    spine = ndimage.binary_closing(
        spine, structure=ndimage.generate_binary_structure(2, 2), iterations=2
    )
    seed_labels, seed_count = ndimage.label(
        spine, structure=ndimage.generate_binary_structure(2, 2)
    )
    seed_areas = np.bincount(seed_labels.ravel(), minlength=seed_count + 1)
    keep_seeds = np.flatnonzero(seed_areas >= min_seed_area)
    keep_seeds = keep_seeds[keep_seeds != 0]
    if not len(keep_seeds):
        return connected_components(
            foreground_probability,
            threshold=threshold,
            min_area=min_area,
        )

    embedding_map = np.asarray(embeddings, dtype=np.float32)
    seed_means = []
    seed_boxes = []
    seed_weights = []
    for seed_id in keep_seeds:
        ys, xs = np.where(seed_labels == seed_id)
        vectors = embedding_map[:, ys, xs].T
        seed_means.append(vectors.mean(axis=0))
        seed_boxes.append((ys.min(), xs.min(), ys.max(), xs.max()))
        seed_weights.append(len(ys))

    # Fragments of one predicted spine can form separate seed components. Merge only
    # nearby components whose learned embeddings indicate the same instance.
    parent = list(range(len(keep_seeds)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    means_array = np.stack(seed_means)
    for first in range(len(keep_seeds)):
        y0a, x0a, y1a, x1a = seed_boxes[first]
        for second in range(first + 1, len(keep_seeds)):
            if np.linalg.norm(means_array[first] - means_array[second]) > embedding_merge_distance:
                continue
            y0b, x0b, y1b, x1b = seed_boxes[second]
            dy = max(0, max(y0a, y0b) - min(y1a, y1b) - 1)
            dx = max(0, max(x0a, x0b) - min(x1a, x1b) - 1)
            if np.hypot(dy, dx) <= max_seed_gap:
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(keep_seeds)):
        groups.setdefault(find(index), []).append(index)
    prototypes = []
    for members in groups.values():
        weights = np.asarray([seed_weights[index] for index in members], dtype=np.float32)
        prototypes.append(
            np.average(means_array[members], axis=0, weights=weights)
        )
    prototypes_array = np.asarray(prototypes, dtype=np.float32)

    ys, xs = np.where(foreground)
    pixel_vectors = embedding_map[:, ys, xs].T
    assignments = np.empty(len(ys), dtype=np.int32)
    max_distance_sq = max_assignment_distance**2
    # Bound temporary memory for large full-disk foreground regions.
    for start in range(0, len(ys), 12000):
        stop = min(start + 12000, len(ys))
        vectors = pixel_vectors[start:stop]
        distances = (
            (vectors * vectors).sum(axis=1, keepdims=True)
            + (prototypes_array * prototypes_array).sum(axis=1)[None]
            - 2.0 * vectors @ prototypes_array.T
        )
        best = distances.argmin(axis=1)
        best_distances = distances[np.arange(len(best)), best]
        assignments[start:stop] = np.where(best_distances <= max_distance_sq, best + 1, 0)

    labels = np.zeros(foreground.shape, dtype=np.uint16)
    labels[ys, xs] = assignments.astype(np.uint16)
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    remap = np.zeros(len(areas), dtype=np.uint16)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return remap[labels]


def encode_instance(mask: np.ndarray) -> str:
    return encode_mask(mask)


def write_submission(
    test_image_dir: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    threshold: float = 0.6,
    min_area: int = 512,
    merge_distance: int = 10,
    batch_size: int = 4,
) -> None:
    import csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint_path, device)
    paths = sorted(Path(test_image_dir).glob("*.jpeg"))
    with Path(output_csv).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for path in paths:
            image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
            _, labels = predict_instances(
                model,
                image,
                device,
                threshold,
                min_area,
                batch_size=batch_size,
                merge_distance=merge_distance,
            )
            for instance_id in range(1, int(labels.max()) + 1):
                writer.writerow([
                    f"{path.stem}_{instance_id}",
                    encode_instance(labels == instance_id),
                ])
