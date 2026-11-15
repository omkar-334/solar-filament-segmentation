"""Encoding and auditing helpers for the Kaggle compressed COCO RLE format."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from pycocotools import mask as mask_utils


def encode_mask(mask: np.ndarray) -> str:
    """Encode a 2-D binary mask as the competition's compressed RLE counts."""
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"RLE mask must be 2-D, got shape {binary.shape}")
    binary = np.asfortranarray((binary != 0).astype(np.uint8, copy=False))
    rle = mask_utils.encode(binary)
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def decode_mask(counts: str, height: int, width: int) -> np.ndarray:
    """Decode compressed counts using the competition's implicit image shape."""
    if not counts or any(char in counts for char in ',"\'\n\r'):
        raise ValueError("RLE counts are empty or contain a CSV delimiter")
    encoded = {"size": [int(height), int(width)], "counts": counts.encode("ascii")}
    return mask_utils.decode(encoded).astype(bool, copy=False)


def audit_submission(
    path: str | Path,
    image_shapes: dict[str, tuple[int, int]],
    expected_image_ids: Iterable[str] | None = None,
) -> int:
    """Validate CSV schema, IDs, RLEs, uniqueness, dimensions and overlap.

    ``image_shapes`` maps image stems to (height, width). Images with no
    predictions are allowed to have no CSV rows, as required by the challenge.
    Returns the number of prediction rows.
    """
    import csv
    import re

    id_pattern = re.compile(r"^(?P<stem>.+)_(?P<instance>[1-9]\d*)$")
    seen_ids: set[str] = set()
    expected = set(expected_image_ids) if expected_image_ids is not None else None
    current_stem: str | None = None
    current_instance = 0
    closed_stems: set[str] = set()
    occupied: np.ndarray | None = None
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected submission columns: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            filament_id = row["filament_id"]
            match = id_pattern.fullmatch(filament_id or "")
            if match is None:
                raise ValueError(f"Malformed filament_id on row {row_number}: {filament_id!r}")
            if filament_id in seen_ids:
                raise ValueError(f"Duplicate filament_id on row {row_number}: {filament_id}")
            seen_ids.add(filament_id)
            stem = match.group("stem")
            if stem not in image_shapes:
                raise ValueError(f"Unknown image stem on row {row_number}: {stem}")
            if expected is not None and stem not in expected:
                raise ValueError(f"Unexpected image stem on row {row_number}: {stem}")
            if stem != current_stem:
                if stem in closed_stems:
                    raise ValueError(f"Rows for image {stem} are not contiguous")
                if current_stem is not None:
                    closed_stems.add(current_stem)
                current_stem = stem
                current_instance = 0
                occupied = np.zeros(image_shapes[stem], dtype=bool)
            instance_number = int(match.group("instance"))
            if instance_number != current_instance + 1:
                raise ValueError(
                    f"Instance IDs for {stem} must be sequential from 1; "
                    f"got {instance_number} on row {row_number}"
                )
            current_instance = instance_number
            height, width = image_shapes[stem]
            mask = decode_mask(row["segmentation_rle"], height, width)
            if not mask.any():
                raise ValueError(f"Empty mask on row {row_number}: {filament_id}")
            assert occupied is not None
            if np.any(occupied & mask):
                raise ValueError(f"Overlapping instance masks for image {stem}")
            occupied |= mask
    return len(seen_ids)
